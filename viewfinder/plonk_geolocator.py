"""Probabilistic geolocation using the pretrained PLONK model family.

PLONK is an independent, generative alternative to GeoCLIP.  Its published
models were trained on substantially larger, global datasets:

* ``nicolas-dufour/PLONK_OSV_5M`` -- 5.1M street-view images from 225 countries
* ``nicolas-dufour/PLONK_YFCC`` -- geotagged, general-purpose Flickr imagery

Unlike a deterministic regressor, PLONK samples a distribution over the
Earth.  This adapter draws multiple samples for each image, geographically
clusters them, and exposes the modal cluster centroids as ordinary
``Candidate`` objects.  Cluster mass becomes the candidate score, so a diffuse
posterior is represented honestly instead of being collapsed into one
overconfident point.

The dependency and model weights are optional and loaded lazily.  Installing
them must not affect the existing GeoCLIP or retrieval predictors.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .aggregator import Candidate, FramePrediction, cluster_candidates
from .geolocator import resolve_device
from .progress import report as _report_progress


logger = logging.getLogger(__name__)

MODEL_OSV5M = "nicolas-dufour/PLONK_OSV_5M"
MODEL_YFCC = "nicolas-dufour/PLONK_YFCC"
SUPPORTED_MODELS = {
    "plonk-osv": MODEL_OSV5M,
    "plonk-yfcc": MODEL_YFCC,
}

DEFAULT_SAMPLES_PER_IMAGE = 32
DEFAULT_SAMPLE_CLUSTER_RADIUS_KM = 200.0
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEED = 20260726


class PlonkError(RuntimeError):
    """Raised when PLONK cannot be loaded or returns invalid coordinates."""


@dataclass(frozen=True)
class PlonkProvenance:
    """Model and sampling details attached to serialized predictions."""

    model_id: str
    training_dataset: str
    samples_per_image: int
    sample_cluster_radius_km: float
    stochastic: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "training_dataset": self.training_dataset,
            "samples_per_image": self.samples_per_image,
            "sample_cluster_radius_km": self.sample_cluster_radius_km,
            "probabilistic_sampling": self.stochastic,
        }


def _validate_samples(samples: Any) -> np.ndarray:
    """Return a finite ``(n, 2)`` latitude/longitude array in degrees."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise PlonkError(
            f"PLONK returned shape {values.shape}; expected a non-empty (n, 2) array"
        )
    if not np.isfinite(values).all():
        raise PlonkError("PLONK returned NaN or infinite coordinates")
    if np.any((values[:, 0] < -90.0) | (values[:, 0] > 90.0)):
        raise PlonkError("PLONK returned a latitude outside [-90, 90]")
    if np.any((values[:, 1] < -180.0) | (values[:, 1] > 180.0)):
        raise PlonkError("PLONK returned a longitude outside [-180, 180]")
    return values


def samples_to_candidates(
    samples: Any,
    *,
    top_k: int,
    cluster_radius_km: float = DEFAULT_SAMPLE_CLUSTER_RADIUS_KM,
) -> list[Candidate]:
    """Turn posterior samples into ranked geographic modes.

    Every raw sample has equal probability mass.  Connected-component
    clustering groups nearby samples, spherical centroids avoid antimeridian
    errors, and the fraction of samples in a component becomes its score.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not math.isfinite(cluster_radius_km) or cluster_radius_km <= 0:
        raise ValueError("cluster_radius_km must be a finite positive number")

    values = _validate_samples(samples)
    raw = [Candidate(float(lat), float(lon), 1.0) for lat, lon in values]
    clusters = cluster_candidates(raw, radius_km=cluster_radius_km)
    total = float(len(raw))
    candidates = [
        Candidate(
            lat=float(cluster.lat),
            lon=float(cluster.lon),
            score=float(cluster.size / total),
        )
        for cluster in clusters
    ]
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[:top_k]


class PlonkGeolocator:
    """Adapt one pretrained PLONK model to View Finder's predictor protocol."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        offline: bool = False,
        samples_per_image: int = DEFAULT_SAMPLES_PER_IMAGE,
        cluster_radius_km: float = DEFAULT_SAMPLE_CLUSTER_RADIUS_KM,
        seed: int = DEFAULT_SEED,
        pipeline: Any | None = None,
    ) -> None:
        if model_id not in SUPPORTED_MODELS.values():
            supported = ", ".join(sorted(SUPPORTED_MODELS.values()))
            raise ValueError(f"Unsupported PLONK model {model_id!r}; expected one of {supported}")
        if samples_per_image <= 0:
            raise ValueError("samples_per_image must be greater than zero")
        if not math.isfinite(cluster_radius_km) or cluster_radius_km <= 0:
            raise ValueError("cluster_radius_km must be a finite positive number")

        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.model_id = model_id
        self.device = resolve_device(device)
        self.samples_per_image = int(samples_per_image)
        self.cluster_radius_km = float(cluster_radius_km)
        self.seed = int(seed)

        if pipeline is None:
            try:
                from plonk import PlonkPipeline
            except ImportError as exc:
                raise PlonkError(
                    "PLONK support is optional. Install diff-plonk and its runtime "
                    "dependencies as documented in README.md."
                ) from exc

            logger.info("Loading %s on %s", self.model_id, self.device)
            try:
                pipeline = PlonkPipeline(self.model_id, device=self.device)
            except Exception as exc:
                raise PlonkError(f"Could not load {self.model_id}: {exc}") from exc
        self.pipeline = pipeline

        training_dataset = "OSV-5M" if model_id == MODEL_OSV5M else "YFCC-100M"
        self.provenance = PlonkProvenance(
            model_id=model_id,
            training_dataset=training_dataset,
            samples_per_image=self.samples_per_image,
            sample_cluster_radius_km=self.cluster_radius_km,
        )

    def _generator(self, offset: int = 0) -> Any:
        """Build a deterministic Torch generator for reproducible sampling."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - PLONK requires torch
            raise PlonkError("PLONK requires torch") from exc

        generator_device = self.device if self.device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(self.seed + int(offset))
        return generator

    def predict_image(
        self,
        image: Image.Image,
        top_k: int = 5,
        *,
        sample_offset: int = 0,
    ) -> list[Candidate]:
        """Sample PLONK's posterior and return its highest-mass location modes."""
        sample_count = max(self.samples_per_image, top_k)
        try:
            samples = self.pipeline(
                image.convert("RGB"),
                batch_size=sample_count,
                generator=self._generator(sample_offset),
            )
        except Exception as exc:
            raise PlonkError(f"{self.model_id} inference failed: {exc}") from exc
        return samples_to_candidates(
            samples,
            top_k=top_k,
            cluster_radius_km=self.cluster_radius_km,
        )

    def predict_batch(
        self,
        images: Sequence[Image.Image],
        top_k: int = 5,
    ) -> list[list[Candidate]]:
        """Predict images independently so each receives a full posterior sample."""
        return [
            self.predict_image(image, top_k=top_k, sample_offset=index)
            for index, image in enumerate(images)
        ]

    def predict_frames(
        self,
        frames: Sequence["FrameLike"],
        top_k: int = 5,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress: Any = False,
    ) -> list[FramePrediction]:
        """Predict video frames while bounding feature-extractor memory."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        predictions: list[FramePrediction] = []
        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            for local_index, frame in enumerate(chunk):
                candidates = self.predict_image(
                    frame.image,
                    top_k=top_k,
                    sample_offset=start + local_index,
                )
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
                message="PLONK-geolocated %d/%d frames",
            )
        return predictions


class FrameLike:
    """Structural protocol documented without importing the video module."""

    image: Image.Image
    index: int
    timestamp: float
    blur_score: float
