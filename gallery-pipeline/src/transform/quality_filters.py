"""Dependency-light image quality signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def laplacian_variance(path: str | Path) -> float:
    """Estimate sharpness with a discrete Laplacian, without requiring OpenCV."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((512, 512)), dtype=np.float32)
    center = pixels[1:-1, 1:-1]
    laplacian = (
        -4.0 * center
        + pixels[:-2, 1:-1]
        + pixels[2:, 1:-1]
        + pixels[1:-1, :-2]
        + pixels[1:-1, 2:]
    )
    return float(np.var(laplacian))


def passes_blur_filter(path: str | Path, threshold: float) -> tuple[bool, float]:
    score = laplacian_variance(path)
    return score >= threshold, score

