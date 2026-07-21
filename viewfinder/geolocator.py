"""GeoCLIP wrapper: images/frames in, ranked GPS candidates out.

GeoCLIP (Vivanco Cepeda et al., NeurIPS 2023) embeds an image and a set of GPS
coordinates into a shared space, so geolocation becomes retrieval: score the
image against a gallery of ~100k coordinates that ship with the pip package and
take the best matches. No external dataset or API is involved.

This module exists rather than calling ``GeoCLIP.predict()`` directly because
that method has two properties that hurt on video:

* It takes a **file path** and opens it itself, so in-memory video frames would
  have to be written to disk and read back.
* It re-encodes the **entire 100k-point GPS gallery** on every call. Those
  embeddings never change, so for an N-frame video that is N-1 redundant passes
  over the location encoder -- the dominant cost for anything but very short
  clips.

:class:`GeoLocator` therefore encodes the gallery once, caches it, and scores
frames in batches against the cached embeddings. The maths is identical to
``GeoCLIP.predict``; only the redundant work is removed.

The CLIP ViT-L/14 backbone is downloaded from Hugging Face the first time a
model is constructed and cached on disk thereafter. That is the one-time setup
step -- every later run is fully offline.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import torch
from PIL import Image

from .aggregator import Candidate, FramePrediction

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Frames encoded per forward pass. CLIP ViT-L/14 at 224x224 is ~0.3 GB of
# activations per 8 frames on CPU, which keeps memory bounded on modest
# machines while still amortising the per-call overhead.
DEFAULT_BATCH_SIZE = 8


def resolve_device(requested: str | None = None) -> str:
    """Pick the torch device to run on.

    Args:
        requested: An explicit device string, or ``None``/``"auto"`` to detect.

    Returns:
        One of ``"cuda"``, ``"mps"`` or ``"cpu"``.
    """
    if requested and requested.lower() != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GeoLocator:
    """Loads GeoCLIP once and scores images against its cached GPS gallery."""

    def __init__(self, device: str | None = None, offline: bool = False) -> None:
        """Load the model and pre-encode the GPS gallery.

        Args:
            device: Torch device, or ``None``/``"auto"`` to detect one.
            offline: If True, forbid Hugging Face from reaching the network.
                Fails loudly if the CLIP backbone is not already cached, which
                is the desired behaviour for an air-gapped run.
        """
        if offline:
            # Must be set before transformers is touched by the geoclip import.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from geoclip import GeoCLIP  # imported lazily: pulls in transformers

        self.device = resolve_device(device)
        logger.info("Loading GeoCLIP on %s", self.device)

        self.model = GeoCLIP()
        self.model.eval()
        self.model.to(self.device)

        self.gps_gallery = self.model.gps_gallery.to(self.device)
        self._location_features = self._encode_gallery()

    @torch.inference_mode()
    def _encode_gallery(self) -> torch.Tensor:
        """Embed and L2-normalise every gallery coordinate, once.

        Returns:
            Tensor of shape ``(gallery_size, embed_dim)``.
        """
        logger.info("Encoding GPS gallery (%d coordinates)", len(self.gps_gallery))
        features = self.model.location_encoder(self.gps_gallery)
        return torch.nn.functional.normalize(features, dim=1)

    @torch.inference_mode()
    def predict_batch(self, images: Sequence[Image.Image], top_k: int = 5) -> list[list[Candidate]]:
        """Score a batch of images and return each one's top-k GPS candidates.

        Args:
            images: RGB PIL images.
            top_k: Number of candidates to return per image.

        Returns:
            One list of :class:`Candidate` per input image, best first.
        """
        if not images:
            return []

        pixel_values = self.model.image_encoder.preprocess_image(list(images)).to(self.device)

        image_features = self.model.image_encoder(pixel_values)
        image_features = torch.nn.functional.normalize(image_features, dim=1)

        # Identical to GeoCLIP.forward, but against the cached gallery features.
        logits = self.model.logit_scale.exp() * (image_features @ self._location_features.t())
        probs = logits.softmax(dim=-1)

        k = min(top_k, probs.shape[-1])
        top = torch.topk(probs, k, dim=-1)

        results: list[list[Candidate]] = []
        for row_indices, row_probs in zip(top.indices, top.values):
            coords = self.gps_gallery[row_indices].cpu()
            results.append(
                [
                    Candidate(lat=float(lat), lon=float(lon), score=float(p))
                    for (lat, lon), p in zip(coords, row_probs.cpu())
                ]
            )
        return results

    def predict_image(self, image: Image.Image, top_k: int = 5) -> list[Candidate]:
        """Score a single image. Convenience wrapper around :meth:`predict_batch`."""
        return self.predict_batch([image], top_k=top_k)[0]

    def predict_frames(
        self,
        frames: Sequence["FrameLike"],
        top_k: int = 5,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress: bool = False,
    ) -> list[FramePrediction]:
        """Score extracted video frames in batches.

        Args:
            frames: Objects exposing ``image``, ``index``, ``timestamp`` and
                ``blur_score`` -- i.e. :class:`video_processor.Frame`.
            top_k: Candidates to retain per frame.
            batch_size: Frames per forward pass.
            progress: Log progress after each batch.

        Returns:
            One :class:`FramePrediction` per frame, in input order.
        """
        predictions: list[FramePrediction] = []

        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            for frame, candidates in zip(chunk, self.predict_batch([f.image for f in chunk], top_k)):
                predictions.append(
                    FramePrediction(
                        candidates=candidates,
                        frame_index=frame.index,
                        timestamp=frame.timestamp,
                        blur_score=frame.blur_score,
                    )
                )
            if progress:
                logger.info("Geolocated %d/%d frames", len(predictions), len(frames))

        return predictions


class FrameLike:
    """Structural description of what :meth:`GeoLocator.predict_frames` needs.

    Documentation only -- :class:`video_processor.Frame` satisfies it.
    """

    image: Image.Image
    index: int
    timestamp: float
    blur_score: float


def is_image(path: str) -> bool:
    """Return True if ``path`` has a recognised still-image extension."""
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
