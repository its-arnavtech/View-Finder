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
    """Vertex grid with a fixed *vertex budget* (a 4:3 grid of ``grid_long``).

    Budgeting by count rather than long side keeps panoramas from
    collapsing to a few rows, and keeps the mobile profile's grid within
    16-bit indices whatever the aspect ratio.
    """
    budget = grid_long * round(grid_long * 0.75)
    aspect = image_width / image_height
    width = min(round((budget * aspect) ** 0.5), round(2.5 * grid_long))
    height = max(2, round(budget / max(width, 1)))
    width = max(2, min(width, round(2.5 * grid_long), image_width))
    height = min(height, round(2.5 * grid_long), image_height)
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


def guided_filter(guide: np.ndarray, source: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-aware filtering of ``source`` steered by ``guide`` (He et al. 2010).

    Used to snap the depth network's soft, slightly offset boundaries onto the
    photo's actual colour edges. Without it, near/far classification along a
    silhouette flips pixel by pixel and leaves a striped halo of background
    texels glued to the foreground's edge. Each output pixel is a local
    linear function of the guide, so depth edges land where colour edges are.

    Args:
        guide: HxW float32 guidance (luminance in [0, 1]).
        source: HxW float32 signal to filter (disparity).
        radius: box radius in pixels.
        eps: regularisation. Larger values follow the guide's edges less and
            copy less of its texture into the depth.
    """
    size = (2 * radius + 1, 2 * radius + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, cv2.CV_32F, size, borderType=cv2.BORDER_REFLECT)

    guide = guide.astype(np.float32)
    source = source.astype(np.float32)
    mean_g, mean_s = box(guide), box(source)
    var_g = box(guide * guide) - mean_g * mean_g
    cov_gs = box(guide * source) - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return (box(a) * guide + box(b)).astype(np.float32)


def sharpen_depth_edges(
    pixel_disparity: np.ndarray,
    rgb: np.ndarray,
    jump: float,
    stride: float,
    ramp_px: float | None = None,
) -> np.ndarray:
    """Turn soft occlusion ramps in a per-pixel disparity map into steps,
    placed on the photo's colour edges.

    Monocular depth draws an occlusion boundary as a ramp several pixels
    wide, and not quite where the colour edge is. On the mesh grid, every
    step along a ramp exceeded ``jump``, so a ramp became a run of edge cells
    extended at slightly different depths ("Venetian blind" stripes from the
    side). Splitting a ramp by depth alone puts the cut a pixel or two off
    the colour edge, which leaves shards of background glued to the
    foreground. Silhouettes are cut per pixel, so the mesh and the mask both
    read this one sharpened map.

    1. A pixel is on a ramp when its gradient is steeper than
       ``jump / stride`` (the grid's edge test, per pixel). Ground planes
       never qualify.
    2. Each ramp pixel is assigned to the near or the far side by *colour*:
       whichever local mean colour, of the confidently near or the
       confidently far pixels around it, it is closer to. Depth decides only
       where colour cannot (no confident pixels on one side).
    3. It takes that side's disparity (the neighbourhood's near or far value).

    The neighbourhood must span a whole ramp, or the snap targets are
    mid-ramp values, but in clutter a wide one mixes neighbouring objects.
    ``ramp_px`` defaults to 0.6% of the image's long side (the network's
    ramps after upsampling) and is never less than 1.5 grid cells.
    """
    values = pixel_disparity.astype(np.float32)
    gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    steep = np.hypot(gx, gy) > 0.5 * jump / max(stride, 1.0)
    if not steep.any():
        return values
    ramp = cv2.dilate(steep.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if ramp_px is None:
        ramp_px = 0.006 * max(values.shape)
    size = 2 * int(np.ceil(max(2.0, 1.5 * stride, ramp_px))) + 1
    window = np.ones((size, size), np.uint8)
    near = cv2.dilate(values, window)
    far = cv2.erode(values, window)
    spread = near - far
    snap = ramp & (spread > jump)
    if not snap.any():
        return values

    # Confident pixels: in the outer quarter of the local range on each side.
    near_core = (values >= near - 0.25 * spread).astype(np.float32)
    far_core = (values <= far + 0.25 * spread).astype(np.float32)
    colour = rgb.astype(np.float32)
    box = (2 * size + 1, 2 * size + 1)

    def local_mean(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weight = cv2.boxFilter(mask, -1, box, normalize=False)
        total = cv2.boxFilter(colour * mask[..., None], -1, box, normalize=False)
        return total / np.maximum(weight, 1e-6)[..., None], weight

    near_colour, near_weight = local_mean(near_core)
    far_colour, far_weight = local_mean(far_core)
    to_near = np.sum((colour - near_colour) ** 2, axis=2)
    to_far = np.sum((colour - far_colour) ** 2, axis=2)
    by_colour = to_near < to_far
    by_depth = values >= (near + far) / 2.0
    decidable = (near_weight > 0.5) & (far_weight > 0.5)
    is_near = np.where(decidable, by_colour, by_depth)

    out = values.copy()
    out[snap] = np.where(is_near[snap], near[snap], far[snap])
    return out
