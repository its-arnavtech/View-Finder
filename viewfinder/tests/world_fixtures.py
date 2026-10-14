"""Synthetic scenes with known geometry for the World Engine tests."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

WIDTH, HEIGHT = 320, 240
HORIZON_ROW = 72  # rows above are sky
BOX = (130, 190, 110, 200)  # x0, x1, y0, y1 of a near box occluder


def scene_rgb() -> np.ndarray:
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    rgb[:HORIZON_ROW] = (110, 160, 230)  # sky
    rng = np.random.default_rng(1)
    ground = rng.integers(60, 120, size=(HEIGHT - HORIZON_ROW, WIDTH, 1))
    rgb[HORIZON_ROW:] = np.concatenate([ground * 0.6, ground, ground * 0.4], axis=2).astype(np.uint8)
    x0, x1, y0, y1 = BOX
    rgb[y0:y1, x0:x1] = (200, 40, 40)
    return rgb


def scene_disparity() -> np.ndarray:
    """Sky = 0; ground disparity linear in the row (a true plane); a near box."""
    disparity = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    rows = np.arange(HORIZON_ROW, HEIGHT, dtype=np.float32)
    disparity[HORIZON_ROW:] = (0.05 + 0.95 * (rows - HORIZON_ROW) / (HEIGHT - HORIZON_ROW - 1))[:, None]
    x0, x1, y0, y1 = BOX
    disparity[y0:y1, x0:x1] = 1.4
    return disparity


def scene_jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(scene_rgb()).save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


class SceneEstimator:
    """Returns the synthetic disparity, resized to whatever it is given."""

    name = "synthetic"

    def __init__(self) -> None:
        self.calls = 0

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        self.calls += 1
        height, width = rgb.shape[:2]
        return cv2.resize(scene_disparity(), (width, height), interpolation=cv2.INTER_NEAREST)

    def describe(self) -> dict[str, str]:
        return {"estimator": self.name, "model_id": "", "device": "cpu"}
