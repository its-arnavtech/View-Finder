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
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from .aggregator import Candidate, FramePrediction

logger = logging.getLogger(__name__)

# Re-exported from formats.py, the single source of truth for accepted inputs.
from .formats import IMAGE_EXTENSIONS  # noqa: F401
from .progress import report as _report_progress

# Optional custom GPS gallery, selectable without code changes.
GALLERY_ENV = "VIEWFINDER_GALLERY"
GALLERY_MODE_ENV = "VIEWFINDER_GALLERY_MODE"

# --- Fine-tuned location encoder ----------------------------------------
#
# Only the location encoder was ever trained (9.45 M of GeoCLIP's 314 M
# parameters). The CLIP vision tower, its projection head, the logit scale and
# the GPS gallery are all untouched stock GeoCLIP, so loading this swaps one
# MLP and changes nothing else about how an image is embedded.
#
# osv5m_reg_narrow_30 is an L2-SP run (regularisation strength 30) on OSV-5M.
# It scores 953.46 km median error on the independent 1000-image benchmark
# against stock GeoCLIP's 1330.86 km -- a 377.4 km improvement, and 38.5 km
# better than the previous best checkpoint (osv5m04, 991.98 km).
#
# Set VIEWFINDER_LOCATION_ENCODER to point elsewhere, or pass
# use_stock_geoclip=True / --use-stock-geoclip to skip it entirely.
LOCATION_ENCODER_ENV = "VIEWFINDER_LOCATION_ENCODER"
DEFAULT_LOCATION_ENCODER = Path(
    "finetune/checkpoints/osv5m_reg_narrow_30/location_encoder_best_external.pth"
)
FINETUNED_ENCODER_INFO = {
    "run": "osv5m_reg_narrow_30",
    "method": "L2-SP fine-tune on OSV-5M, regularisation strength 30",
    "external_median_error_km": 953.46,
    "stock_median_error_km": 1330.86,
    "benchmark": "benchmark-data/manifest.csv, 1000-image fixed sample",
}


def _resolve_location_encoder(explicit: str | os.PathLike[str] | None) -> Path:
    """Resolve the fine-tuned location encoder and fail if it is unavailable.

    Checked against the working directory first and then the repository root,
    so a run started from anywhere still finds the bundled checkpoint.

    Stock GeoCLIP is an explicit comparison mode. Silently falling back here
    would make the product's default depend on the launch directory or a
    missing local artifact while still producing plausible-looking answers.
    """
    candidate = explicit if explicit is not None else os.getenv(LOCATION_ENCODER_ENV)
    if candidate:
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(
            f"Fine-tuned location encoder not found: {path}. "
            "Restore the checkpoint or use --use-stock-geoclip intentionally."
        )

    repo_root = Path(__file__).resolve().parent.parent
    for base in (Path.cwd(), repo_root):
        path = (base / DEFAULT_LOCATION_ENCODER).resolve()
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Fine-tuned location encoder not found at "
        f"{(repo_root / DEFAULT_LOCATION_ENCODER).resolve()}. "
        "Restore the checkpoint or use --use-stock-geoclip intentionally."
    )
GALLERY_MODES = ("augment", "replace")
DEFAULT_GALLERY_MODE = "augment"

# Coordinates closer than this are treated as the same gallery entry when merging.
# 5 decimal places is ~1 m, far finer than the retrieval task needs.
_COORD_DECIMALS = 5

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


def load_gallery_coordinates(path: str | os.PathLike[str]) -> np.ndarray:
    """Load ``(lat, lon)`` pairs from a gallery built by the gallery-pipeline.

    Accepts a version directory (``.../gallery/v2``), a ``metadata.parquet``
    file, or a CSV with ``lat``/``lon`` columns.

    Only the *coordinates* are read, not the FAISS vectors. GeoCLIP scores a
    query image against embeddings of GPS coordinates produced by its own
    location encoder, so it can only ever predict a location that exists in its
    gallery. The pipeline's ``gallery.faiss`` holds OpenCLIP ViT-B-32 *image*
    embeddings, which live in a different vector space from GeoCLIP's CLIP
    ViT-L/14 image tower and are not comparable to it -- feeding them into
    GeoCLIP would produce meaningless similarities. What transfers usefully is
    the set of real, densely-sampled coordinates the pipeline discovered, which
    widens the set of places GeoCLIP is able to answer with.

    Args:
        path: Version directory, parquet file, or CSV file.

    Returns:
        Array of shape ``(n, 2)`` holding latitude and longitude.

    Raises:
        FileNotFoundError: If no readable metadata file is found.
        ValueError: If the file has no usable coordinate columns or no rows.
    """
    source = Path(path).expanduser()
    if source.is_dir():
        for candidate in ("metadata.parquet", "metadata.csv", "coordinates.csv"):
            if (source / candidate).is_file():
                source = source / candidate
                break
        else:
            raise FileNotFoundError(
                f"No metadata.parquet or metadata.csv found in gallery directory: {source}"
            )
    if not source.is_file():
        raise FileNotFoundError(f"Gallery metadata not found: {source}")

    if source.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "Reading a .parquet gallery requires pyarrow (pip install pyarrow), "
                "or export the gallery metadata to CSV instead."
            ) from exc
        table = pq.read_table(source)
        columns = {name.lower(): name for name in table.column_names}
        if "lat" not in columns or "lon" not in columns:
            raise ValueError(f"Gallery metadata lacks lat/lon columns: {table.column_names}")
        lats = table[columns["lat"]].to_pylist()
        lons = table[columns["lon"]].to_pylist()
    else:
        import csv

        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Gallery metadata is empty: {source}")
        fields = {name.lower(): name for name in rows[0]}
        if "lat" not in fields or "lon" not in fields:
            raise ValueError(f"Gallery metadata lacks lat/lon columns: {list(rows[0])}")
        lats = [row[fields["lat"]] for row in rows]
        lons = [row[fields["lon"]] for row in rows]

    coordinates = np.asarray(
        [(float(lat), float(lon)) for lat, lon in zip(lats, lons)], dtype=np.float64
    )
    if coordinates.size == 0:
        raise ValueError(f"Gallery metadata contains no coordinates: {source}")

    finite = np.isfinite(coordinates).all(axis=1)
    in_range = (
        (coordinates[:, 0] >= -90)
        & (coordinates[:, 0] <= 90)
        & (coordinates[:, 1] >= -180)
        & (coordinates[:, 1] <= 180)
    )
    valid = coordinates[finite & in_range]
    if valid.size == 0:
        raise ValueError(f"Gallery metadata contains no valid coordinates: {source}")
    if len(valid) != len(coordinates):
        logger.warning(
            "Dropped %d out-of-range coordinate(s) from %s", len(coordinates) - len(valid), source
        )
    return valid


def _silence_slow_processor_notice() -> None:
    """Drop GeoCLIP's harmless "slow image processor" notice from transformers.

    GeoCLIP's bundled ``ImageEncoder`` builds the CLIP processor with
    ``AutoProcessor.from_pretrained(...)`` and no ``use_fast`` argument, so
    transformers logs a one-time notice that it is defaulting to the slow
    processor. The fast and slow processors differ only in minor numerical ways,
    and we cannot pass ``use_fast`` through GeoCLIP's hardcoded call, so this
    filters just that one message rather than lowering transformers' log level
    and hiding genuine warnings.

    The filter is attached to the transformers logger and, crucially, to its
    handlers: the notice is emitted by a child logger whose records reach the
    parent's *handlers* on propagation, so a handler-level filter is what
    actually catches it.
    """
    import logging as _logging

    message_marker = "use_fast"

    class _DropUseFast(_logging.Filter):
        def filter(self, record: _logging.LogRecord) -> bool:
            try:
                return message_marker not in record.getMessage()
            except Exception:
                return True

    transformers_logger = _logging.getLogger("transformers")
    # Avoid stacking duplicate filters if a GeoLocator is constructed repeatedly.
    if getattr(transformers_logger, "_viewfinder_use_fast_filtered", False):
        return
    filt = _DropUseFast()
    transformers_logger.addFilter(filt)
    for handler in transformers_logger.handlers:
        handler.addFilter(filt)
    transformers_logger._viewfinder_use_fast_filtered = True  # type: ignore[attr-defined]


class _EncoderProvenance:
    """Which location encoder a result came from, for the output JSON."""

    def __init__(self, info: dict[str, Any]) -> None:
        self._info = dict(info)

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_encoder": "geoclip-stock (CLIP ViT-L/14, untouched)",
            "location_encoder": self._info,
        }


class GeoLocator:
    """Loads GeoCLIP once and scores images against its cached GPS gallery."""

    def __init__(
        self,
        device: str | None = None,
        offline: bool = False,
        gallery_path: str | os.PathLike[str] | None = None,
        gallery_mode: str | None = None,
        use_stock_geoclip: bool = False,
        location_encoder_path: str | os.PathLike[str] | None = None,
    ) -> None:
        """Load the model and pre-encode the GPS gallery.

        Args:
            device: Torch device, or ``None``/``"auto"`` to detect one.
            offline: If True, forbid Hugging Face from reaching the network.
                Fails loudly if the CLIP backbone is not already cached, which
                is the desired behaviour for an air-gapped run.
            gallery_path: Optional custom GPS gallery produced by the
                gallery-pipeline. Falls back to ``$VIEWFINDER_GALLERY``. When
                neither is set, GeoCLIP's bundled 100k-coordinate gallery is
                used unchanged.
            gallery_mode: ``"augment"`` (default) appends the custom
                coordinates to the bundled gallery, keeping global coverage
                while adding fine-grained options in the covered regions.
                ``"replace"`` swaps the gallery out entirely, which restricts
                every prediction to the custom coordinates. Falls back to
                ``$VIEWFINDER_GALLERY_MODE``.
            use_stock_geoclip: Skip the fine-tuned location encoder and keep
                GeoCLIP's original weights. For comparison and debugging.
            location_encoder_path: Explicit checkpoint to load instead of the
                bundled one. Falls back to ``$VIEWFINDER_LOCATION_ENCODER``.

        Raises:
            ValueError: If ``gallery_mode`` is not a recognised mode.
        """
        if offline:
            # Must be set before transformers is touched by the geoclip import.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from geoclip import GeoCLIP  # imported lazily: pulls in transformers

        # transformers is now imported, so its logger/handler exist and the
        # filter below will be in place before GeoCLIP builds the CLIP processor.
        _silence_slow_processor_notice()

        self.device = resolve_device(device)
        logger.info("Loading GeoCLIP on %s", self.device)

        self.model = GeoCLIP()

        # Swap in the fine-tuned location encoder before the model moves to the
        # device. Only this MLP changes; the CLIP tower, its projection head,
        # the logit scale and the GPS gallery stay stock.
        self.location_encoder_info: dict[str, Any] = {"source": "geoclip-stock"}
        if not use_stock_geoclip:
            checkpoint = _resolve_location_encoder(location_encoder_path)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            weights = state.get("location_encoder", state)
            # strict=True on purpose: a silent partial load would leave a
            # half-trained encoder that still runs and quietly scores worse.
            self.model.location_encoder.load_state_dict(weights, strict=True)
            self.location_encoder_info = {
                "source": "fine-tuned",
                "path": str(checkpoint),
                "trained_step": state.get("step"),
                **FINETUNED_ENCODER_INFO,
            }
            logger.info(
                "Loaded fine-tuned location encoder (%s, step %s): "
                "%.2f km external median vs stock %.2f km",
                FINETUNED_ENCODER_INFO["run"],
                state.get("step"),
                FINETUNED_ENCODER_INFO["external_median_error_km"],
                FINETUNED_ENCODER_INFO["stock_median_error_km"],
            )

        # Surfaced in the JSON output so a saved result always records which
        # location encoder produced it. Without this, a fine-tuned and a stock
        # run are indistinguishable after the fact.
        self.provenance = _EncoderProvenance(self.location_encoder_info)

        self.model.eval()
        self.model.to(self.device)

        gallery = self.model.gps_gallery
        self.gallery_info: dict[str, Any] = {
            "source": "geoclip-default",
            "mode": None,
            "bundled_coordinates": int(len(gallery)),
            "custom_coordinates": 0,
            "total_coordinates": int(len(gallery)),
        }

        resolved_path = gallery_path if gallery_path is not None else os.getenv(GALLERY_ENV)
        if resolved_path:
            mode = (gallery_mode or os.getenv(GALLERY_MODE_ENV) or DEFAULT_GALLERY_MODE).lower()
            if mode not in GALLERY_MODES:
                raise ValueError(
                    f"Unknown gallery mode {mode!r}; expected one of {', '.join(GALLERY_MODES)}"
                )
            gallery = self._build_gallery(gallery, resolved_path, mode)
            # Keep the wrapped model coherent for anything reading it directly.
            self.model.gps_gallery = gallery

        self.gps_gallery = gallery.to(self.device)
        self._location_features = self._encode_gallery()

    def _build_gallery(self, bundled: torch.Tensor, path: str | os.PathLike[str], mode: str) -> torch.Tensor:
        """Merge or replace the bundled GPS gallery with custom coordinates."""
        custom = load_gallery_coordinates(path)
        custom_tensor = torch.tensor(custom, dtype=bundled.dtype)

        if mode == "replace":
            merged = custom_tensor
        else:
            # Only the custom coordinates are deduplicated, and only against the
            # bundled set and themselves. The bundled gallery is passed through
            # untouched: it contains its own repeated coordinates, and dropping
            # those would change GeoCLIP's default behaviour rather than just
            # extending it, making an augmented run incomparable to a default one.
            bundled_cpu = bundled.cpu()
            # Compare both sides in the bundled gallery's dtype. GeoCLIP's gallery is
            # float32, whose precision at these magnitudes (~4e-6) is close to the
            # rounding step, so rounding a float64 copy of the same coordinate can
            # land on a different value and defeat the comparison.
            known = {
                tuple(row) for row in np.round(bundled_cpu.numpy(), _COORD_DECIMALS).tolist()
            }
            keep: list[int] = []
            for index, row in enumerate(
                np.round(custom_tensor.numpy(), _COORD_DECIMALS).tolist()
            ):
                key = tuple(row)
                if key not in known:
                    known.add(key)
                    keep.append(index)
            added = custom_tensor[keep] if keep else custom_tensor[:0]
            merged = torch.cat([bundled_cpu, added], dim=0)
            logger.debug(
                "%d of %d custom coordinates were already in the bundled gallery",
                len(custom_tensor) - len(added), len(custom_tensor),
            )

        self.gallery_info = {
            "source": str(path),
            "mode": mode,
            "bundled_coordinates": int(len(bundled)),
            "custom_coordinates": int(len(custom_tensor)),
            "custom_coordinates_added": int(len(merged) - len(bundled)) if mode == "augment" else int(len(merged)),
            "total_coordinates": int(len(merged)),
        }
        logger.info(
            "Custom GPS gallery (%s): %d bundled + %d custom -> %d coordinates",
            mode, len(bundled), len(custom_tensor), len(merged),
        )
        return merged

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
        progress: Any = False,
    ) -> list[FramePrediction]:
        """Score extracted video frames in batches.

        Args:
            frames: Objects exposing ``image``, ``index``, ``timestamp`` and
                ``blur_score`` -- i.e. :class:`video_processor.Frame`.
            top_k: Candidates to retain per frame.
            batch_size: Frames per forward pass.
            progress: ``True`` to log a line per batch, or a
                ``(done, total)`` callable to drive a progress bar or GUI.

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
            _report_progress(
                progress,
                len(predictions),
                len(frames),
                logger=logger,
                message="Geolocated %d/%d frames",
            )

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
