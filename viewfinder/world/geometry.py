"""Depth conditioning and grid sampling.

Units: the scene is normalised so the nearest surface sits at depth 1 and the
farthest non-sky surface at ``far``. "Disparity" below always means ``1/z`` in
those units, so it lies in ``[1/far, 1]`` and is (to within ``1/far``) the
network's normalised output. Every threshold is expressed in disparity because
the parallax a moving viewer sees is proportional to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Intrinsics


def normalize_disparity(
    disparity: np.ndarray,
    exclude: np.ndarray | None = None,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    """Robustly rescale relative disparity to ``[0, 1]``.

    Percentiles are taken over pixels not in ``exclude`` so that an infinitely
    distant sky does not compress the range of everything else.
    """
    values = disparity[~exclude] if exclude is not None and (~exclude).any() else disparity
    low, high = np.percentile(values, [low_percentile, high_percentile])
    if not np.isfinite(high - low) or high - low < 1e-6:
        return np.full(disparity.shape, 0.5, dtype=np.float32)
    return np.clip((disparity - low) / (high - low), 0.0, 1.0).astype(np.float32)


def smooth_disparity(disparity: np.ndarray, long_side: int) -> np.ndarray:
    """Edge-preserving smoothing: flattens texture noise, keeps boundaries sharp."""
    diameter = 5 if long_side <= 1200 else 7
    return cv2.bilateralFilter(disparity.astype(np.float32), diameter, 0.06, diameter)


def to_scene_disparity(normalized: np.ndarray, far: float) -> np.ndarray:
    """Map ``[0, 1]`` normalised disparity onto ``[1/far, 1]`` scene disparity.

    Values above 1 (frame extrapolation can produce them) stay above 1, which
    places them nearer than the nearest photographed surface.
    """
    return (normalized * (1.0 - 1.0 / far) + 1.0 / far).astype(np.float32)


@dataclass(frozen=True)
class Grid:
    """A regular vertex grid laid over a (padded) image."""

    width: int  # vertices across
    height: int  # vertices down
    u: np.ndarray  # (height, width) continuous pixel x of each vertex
    v: np.ndarray  # (height, width) continuous pixel y of each vertex
    ui: np.ndarray  # (height, width) integer pixel column sampled
    vi: np.ndarray  # (height, width) integer pixel row sampled
    image_width: int
    image_height: int

    @property
    def stride(self) -> float:
        return self.image_width / max(1, self.width - 1)

    def sample(self, array: np.ndarray) -> np.ndarray:
        """Point-sample a per-pixel array at the vertices.

        Point sampling, not area averaging: averaging across an occlusion
        boundary invents a depth halfway between two surfaces, which is
        exactly the "flying pixel" the mesh is trying to avoid.
        """
        return array[self.vi, self.ui]

    def uv(self) -> np.ndarray:
        return np.stack([self.u / self.image_width, self.v / self.image_height], axis=-1)


def make_grid(image_width: int, image_height: int, grid_long: int) -> Grid:
    if image_width >= image_height:
        width = grid_long
        height = max(2, round(grid_long * image_height / image_width))
    else:
        height = grid_long
        width = max(2, round(grid_long * image_width / image_height))
    width = min(width, image_width)
    height = min(height, image_height)
    xs = np.linspace(0.5, image_width - 0.5, width, dtype=np.float64)
    ys = np.linspace(0.5, image_height - 0.5, height, dtype=np.float64)
    u, v = np.meshgrid(xs, ys)
    ui = np.clip(np.floor(u).astype(np.int64), 0, image_width - 1)
    vi = np.clip(np.floor(v).astype(np.int64), 0, image_height - 1)
    return Grid(width, height, u, v, ui, vi, image_width, image_height)


def unproject(grid: Grid, intrinsics: Intrinsics, depth: np.ndarray) -> np.ndarray:
    """(height, width, 3) camera-space positions for per-vertex depth ``z``."""
    rays = intrinsics.rays(grid.u, grid.v)
    return (rays * depth[..., None]).astype(np.float32)


def near_side_of_jumps(disparity: np.ndarray, jump: float) -> np.ndarray:
    """Vertices on the *near* side of a disparity jump to a 4-neighbour.

    These are the foreground silhouettes: when the viewer moves, the region
    just behind them is what becomes visible, so it is what the background
    layer must cover.
    """
    near = np.zeros(disparity.shape, dtype=bool)
    horizontal = disparity[:, 1:] - disparity[:, :-1]
    near[:, :-1] |= horizontal < -jump
    near[:, 1:] |= horizontal > jump
    vertical = disparity[1:, :] - disparity[:-1, :]
    near[:-1, :] |= vertical < -jump
    near[1:, :] |= vertical > jump
    return near


def navigation_bounds(depth: np.ndarray, valid: np.ndarray) -> dict[str, list[float]]:
    """Box around the original viewpoint where single-view geometry holds up.

    Scaled by the near field (20th-percentile depth): moving further than a
    fraction of the distance to the closest surfaces is where the missing
    sides of objects start to show.
    """
    values = depth[valid]
    near_field = float(np.percentile(values, 20)) if values.size else 1.0
    half = [max(0.05, near_field * k) for k in (0.35, 0.2, 0.6)]
    return {
        "center": [0.0, 0.0, 0.0],
        "half_extents": [round(h, 4) for h in half],
        "near_field_depth": round(near_field, 4),
    }
