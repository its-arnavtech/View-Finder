"""Learned inpainting for content the photo never captured.

A single photo shows nothing behind its foreground objects and nothing outside
its frame, yet a viewer who moves or turns sees both. Rule-based fills (blurs,
colour profiles, nearest-pixel copies) read as smears. LaMa (Suvorov et al.,
WACV 2022) continues textures and structures (rock, foliage, masonry, water)
into large missing regions, which is what those areas need.

The model is the OpenCV Model Zoo's ONNX export of LaMa (Apache-2.0, ~93 MB),
run with ONNX Runtime. Like the depth model, it is only ever loaded from the
local cache; ``view-finder world setup`` downloads it. Its input is fixed at
512 x 512, so images are resized in and out and only the masked pixels are
taken from the result: known pixels are never altered.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

LAMA_REPO = "opencv/inpainting_lama"
LAMA_FILE = "inpainting_lama_2025jan.onnx"
LAMA_SIZE = 512


class InpainterUnavailable(RuntimeError):
    """The inpainting model or its runtime is not installed."""


class Inpainter(Protocol):
    name: str

    def inpaint(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill ``mask`` (HxW bool, True = unknown) in ``rgb`` (HxWx3 uint8)."""
        ...

    def describe(self) -> dict[str, str]: ...


class LamaInpainter:
    name = "lama"

    def __init__(self, model_path: str | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise InpainterUnavailable(
                "onnxruntime is not installed; install viewfinder/requirements-world.txt"
            ) from exc
        path = model_path or cached_model_path()
        if path is None:
            raise InpainterUnavailable(
                "The inpainting model is not installed. Run `view-finder world setup` once "
                "with network access to download it (~93 MB)."
            )
        logger.info("Loading the LaMa inpainting model…")
        options = ort.SessionOptions()
        options.log_severity_level = 3  # the export carries harmless unused-initializer warnings
        # Full graph optimisation took ~100 s to build the session (measured
        # on a Core Ultra 9 185H) for no faster inference; basic takes ~12 s.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # ORT defaults to one thread per physical core. On hybrid CPUs that
        # includes slow efficiency cores, which every step then waits for:
        # passes took 10-58 s instead of ~3 s with six threads.
        options.intra_op_num_threads = max(1, min(6, (os.cpu_count() or 2) // 2))
        self._session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        self._inputs = [i.name for i in self._session.get_inputs()]
        # One session is shared by the server worker and any caller.
        self._lock = threading.Lock()
        logger.info("Loaded LaMa inpainting model")

    def inpaint(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return rgb.copy()
        height, width = rgb.shape[:2]
        small = cv2.resize(rgb, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_AREA)
        # Area-resize the mask so any partly unknown 512-px cell counts as unknown.
        small_mask = cv2.resize(mask.astype(np.float32), (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_AREA) > 0.01
        image = (small.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        hole = small_mask.astype(np.float32)[None, None]
        image = image * (1.0 - hole)
        with self._lock:
            out = self._session.run(None, {self._inputs[0]: image, self._inputs[1]: hole})[0]
        filled = np.clip(out[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        filled = cv2.resize(filled, (width, height), interpolation=cv2.INTER_CUBIC)
        result = rgb.copy()
        result[mask] = filled[mask]
        return result

    def describe(self) -> dict[str, str]:
        return {"inpainter": self.name, "model": f"{LAMA_REPO}/{LAMA_FILE}"}


def cached_model_path() -> str | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover
        return None
    path = try_to_load_from_cache(LAMA_REPO, LAMA_FILE)
    return path if isinstance(path, str) else None


def download_model() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(LAMA_REPO, LAMA_FILE)


_cache: dict[str, LamaInpainter] = {}
_cache_lock = threading.Lock()


def load_inpainter() -> LamaInpainter:
    """Process-cached LaMa inpainter; raises InpainterUnavailable if not installed."""
    with _cache_lock:
        if "lama" not in _cache:
            _cache["lama"] = LamaInpainter()
        return _cache["lama"]
