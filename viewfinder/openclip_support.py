"""Shared OpenCLIP model loading, used wherever a gallery embedding is decoded.

Extracted from ``retrieval_geolocator`` when the retrieval predictor was shelved
into ``experiments/retrieval/`` (see the README's "Retrieval and ensemble" note).
These helpers were never retrieval-specific: the local-evidence hybrid's landmark
matcher (:mod:`local_landmarks`) reads a Wikimedia Commons gallery built with the
same ``open_clip:<model>:<checkpoint>`` manifest convention and needs exactly the
same loader. Keeping them here lets the active hybrid path stand alone while the
shelved experiment imports the identical code rather than a copy of it.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Manifests name their embedding space as "open_clip:<model>:<checkpoint>".
OPEN_CLIP_PREFIX = "open_clip"

# Loading OpenCLIP weights is expensive and a video run would otherwise repeat it
# per call. Double-checked locking, matching reverse_geocode.py's discipline.
_model_cache: dict[tuple[str, str, str], tuple[Any, Any]] = {}
_model_lock = threading.Lock()


class GalleryError(RuntimeError):
    """Raised when a gallery directory is missing, malformed, or inconsistent."""


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split ``open_clip:<model>:<checkpoint>`` into ``(model, checkpoint)``.

    Raises:
        GalleryError: If the spec is empty, malformed, or not an OpenCLIP model.
    """
    parts = str(spec).split(":")
    if len(parts) != 3 or not all(parts):
        raise GalleryError(
            f"Gallery manifest has an unparseable embedding_model {spec!r}; "
            f"expected '{OPEN_CLIP_PREFIX}:<model>:<checkpoint>'"
        )
    framework, model_name, pretrained = parts
    if framework != OPEN_CLIP_PREFIX:
        raise GalleryError(
            f"Gallery was built with {framework!r}, but queries can only be encoded "
            f"with OpenCLIP. Matching across different embedding spaces would return "
            f"meaningless results."
        )
    return model_name, pretrained


def load_model(model_name: str, pretrained: str, device: str) -> tuple[Any, Any]:
    """Load (and process-cache) an OpenCLIP model and its preprocessing transform."""
    key = (model_name, pretrained, device)
    cached = _model_cache.get(key)
    if cached is not None:
        return cached

    with _model_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            return cached

        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise GalleryError(
                "Encoding gallery queries requires open-clip-torch "
                "(pip install open-clip-torch)"
            ) from exc

        logger.info("Loading OpenCLIP %s/%s on %s", model_name, pretrained, device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        _model_cache[key] = (model, preprocess)
        return model, preprocess


def resolve_device(requested: str | None = None) -> str:
    """Pick the torch device to run on."""
    import torch

    if requested and requested.lower() != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Backwards-compatible aliases for the private names this code had while it lived
# in retrieval_geolocator, so the shelved experiment imports unchanged.
_parse_model_spec = parse_model_spec
_load_model = load_model
