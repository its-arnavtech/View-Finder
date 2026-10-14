"""Monocular depth estimation behind one small interface.

Every estimator returns *relative disparity*: an HxW float32 array at the input
resolution where larger means nearer. Absolute scale cannot be recovered from
one photo, so nothing downstream assumes it.

``depth-anything-v2-small`` is the default and is only ever loaded from the
local Hugging Face cache. A missing model raises :class:`DepthModelUnavailable`
naming the explicit setup command instead of downloading silently, matching how
View Finder treats every other model. ``heuristic`` needs no model at all and
exists for tests and air-gapped smoke runs; it is never substituted implicitly.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEPTH_ANYTHING_SMALL = "depth-anything/Depth-Anything-V2-Small-hf"
# The only files the engine reads; `world setup` downloads exactly these.
MODEL_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")

ESTIMATORS = ("depth-anything-v2-small", "heuristic")
DEFAULT_ESTIMATOR = "depth-anything-v2-small"


class DepthModelUnavailable(RuntimeError):
    """The depth model is not cached locally (or its runtime is missing)."""


class DepthEstimator(Protocol):
    name: str

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 RGB -> HxW float32 relative disparity (larger = nearer)."""
        ...

    def describe(self) -> dict[str, str]:
        ...


def _resolve_device(device: str | None) -> str:
    import torch

    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise DepthModelUnavailable("--device cuda was requested but no CUDA GPU is available")
    return device


class DepthAnythingEstimator:
    """Depth Anything V2 (Small, Apache-2.0) through ``transformers``."""

    name = "depth-anything-v2-small"

    def __init__(self, model_id: str = DEPTH_ANYTHING_SMALL, device: str | None = "auto") -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise DepthModelUnavailable(
                f"The depth runtime is missing ({exc.name}); install viewfinder/requirements.txt"
            ) from exc

        self.model_id = model_id
        self.device = _resolve_device(device)
        try:
            # use_fast=False pins the processor the model was validated with
            # (PDD-REVIEW §1) and silences transformers' "slow processor" notice.
            self._processor = AutoImageProcessor.from_pretrained(
                model_id, local_files_only=True, use_fast=False
            )
            model = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=True)
        except (OSError, ValueError) as exc:
            raise DepthModelUnavailable(
                f"Depth model {model_id} is not installed. Run `view-finder world setup` "
                "once with network access to download it (~100 MB)."
            ) from exc
        self._model = model.to(self.device).eval()
        # One model instance is shared by the server's worker and any caller;
        # torch modules are not safe to call concurrently on one device.
        self._lock = threading.Lock()
        logger.info("Loaded %s on %s", model_id, self.device)

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        height, width = rgb.shape[:2]
        with self._lock, torch.inference_mode():
            inputs = self._processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
            predicted = self._model(**inputs).predicted_depth  # 1 x h' x w', short side 518
            disparity = torch.nn.functional.interpolate(
                predicted.unsqueeze(1).float(),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )[0, 0]
        result = disparity.cpu().numpy().astype(np.float32)
        # Bicubic overshoot can dip just below zero at the sky boundary.
        np.maximum(result, 0.0, out=result)
        return result

    def describe(self) -> dict[str, str]:
        return {"estimator": self.name, "model_id": self.model_id, "device": self.device}


class HeuristicDepthEstimator:
    """A model-free prior: a ground plane receding toward the horizon.

    Rows near the bottom are near and rows near the top are far, gently
    modulated by local detail (textured regions tend to be closer than
    smooth, hazy ones). It is crude by design. It produces a valid, tilted
    scene for tests and offline smoke runs, not a reconstruction.
    """

    name = "heuristic"

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        # Disparity of a ground plane seen from eye height is linear in the
        # row below the horizon; above it everything sits far away.
        horizon = 0.35
        ground = np.clip((rows - horizon) / (1.0 - horizon), 0.0, 1.0)
        base = np.broadcast_to(0.08 + 0.92 * ground, (height, width)).astype(np.float32)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        detail = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), max(2.0, width / 200))
        detail /= float(detail.max()) or 1.0
        return (base * (0.85 + 0.15 * detail)).astype(np.float32)

    def describe(self) -> dict[str, str]:
        return {"estimator": self.name, "model_id": "", "device": "cpu"}


_cache: dict[tuple[str, str, str], DepthEstimator] = {}
_cache_lock = threading.Lock()


def load_estimator(
    name: str = DEFAULT_ESTIMATOR,
    *,
    device: str | None = "auto",
    model_id: str | None = None,
) -> DepthEstimator:
    """Return a (process-cached) estimator by name."""
    key_name = (name or DEFAULT_ESTIMATOR).strip().lower()
    if key_name not in ESTIMATORS:
        raise ValueError(f"Unknown depth model {name!r}; choose one of {', '.join(ESTIMATORS)}")
    if key_name == "heuristic":
        return HeuristicDepthEstimator()
    resolved_id = model_id or DEPTH_ANYTHING_SMALL
    key = (key_name, resolved_id, device or "auto")
    with _cache_lock:
        cached = _cache.get(key)
        if cached is None:
            cached = DepthAnythingEstimator(resolved_id, device)
            _cache[key] = cached
        return cached


def download_model(model_id: str = DEPTH_ANYTHING_SMALL) -> list[str]:
    """Explicitly fetch the depth model into the Hugging Face cache."""
    from huggingface_hub import hf_hub_download

    return [hf_hub_download(model_id, filename) for filename in MODEL_FILES]


def model_cached(model_id: str = DEPTH_ANYTHING_SMALL) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover
        return False
    return all(
        isinstance(try_to_load_from_cache(model_id, filename), str) for filename in MODEL_FILES
    )
