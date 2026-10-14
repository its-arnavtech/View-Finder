"""Scene segmentation into layers: sky, foreground and the hidden background.

The background layer is a simplified form of layered-depth "3D photo
inpainting" (Shih et al., CVPR 2020). Behind every foreground silhouette, a
second surface is built with inpainted colour and extrapolated depth, so that
moving the camera reveals a plausible continuation instead of a hole.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Below this provisional disparity a top-connected region is sky. Depth
# Anything outputs exactly 0 for open sky (measured; PDD-REVIEW §1), so the
# threshold only has to absorb haze and bicubic ringing at the boundary.
SKY_DISPARITY = 0.025
MIN_SKY_FRACTION = 0.005


def detect_sky(normalized_disparity: np.ndarray, threshold: float = SKY_DISPARITY) -> np.ndarray:
    """Boolean sky mask: very low disparity, connected to the top border."""
    height, width = normalized_disparity.shape
    candidate = (normalized_disparity < threshold).astype(np.uint8)
    if not candidate[0].any():
        return np.zeros((height, width), dtype=bool)
    kernel_size = max(3, (max(height, width) // 200) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    count, labels = cv2.connectedComponents(candidate, connectivity=4)
    top_labels = np.unique(labels[0][candidate[0] > 0])
    sky = np.isin(labels, top_labels[top_labels > 0])
    # Close pinholes (birds, wires) inside the sky so they don't become
    # isolated shards of geometry floating in front of it.
    sky = cv2.morphologyEx(sky.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    if sky.mean() < MIN_SKY_FRACTION:
        return np.zeros((height, width), dtype=bool)
    return sky


def sky_colors(rgb: np.ndarray, sky: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(zenith, horizon) RGB colours in ``[0, 1]`` taken from the photo.

    With sky visible: the mean of its top and bottom tenths by row. Without
    (indoors, close-ups, the heuristic estimator): the image's top band for
    the horizon and a slightly darker version of it for the zenith.
    """
    pixels = rgb.astype(np.float32) / 255.0
    rows = np.nonzero(sky.any(axis=1))[0]
    if rows.size >= 2:
        span = max(1, (rows[-1] - rows[0] + 1) // 10)
        top = sky[: rows[0] + span]
        bottom_start = rows[-1] - span + 1
        zenith = pixels[: rows[0] + span][top].mean(axis=0)
        horizon = pixels[bottom_start : rows[-1] + 1][sky[bottom_start : rows[-1] + 1]].mean(axis=0)
        return zenith, horizon
    band = pixels[: max(1, pixels.shape[0] // 12)].reshape(-1, 3)
    horizon = band.mean(axis=0)
    return horizon * 0.85, horizon


@dataclass(frozen=True)
class BackgroundLayer:
    region: np.ndarray  # (gh, gw) bool: vertices the layer covers
    disparity: np.ndarray  # (gh, gw) scene disparity behind the occluders
    rgb: np.ndarray  # (h, w, 3) uint8 texture (half the padded image size)


def occluder_pixels(pixel_disparity: np.ndarray, radius: int, jump: float) -> np.ndarray:
    """Pixels that are nearer than something within ``radius`` by more than ``jump``.

    These are exactly the pixels that hide background: the ones whose colour
    the background layer must replace. Background-side pixels stay untouched
    and keep their real colour.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    farthest_nearby = cv2.erode(pixel_disparity.astype(np.float32), kernel)
    return pixel_disparity - farthest_nearby > jump


def background_sources(pixel_disparity: np.ndarray, radius: int, jump: float) -> np.ndarray:
    """Pixels that may donate colour to the hidden background.

    A pixel qualifies when nothing within ``2 * radius`` is farther than it by
    more than ``jump``, i.e. it is not on (or near) the near side of a
    silhouette. This excludes the occluder's own rim, so the occluder can
    never bleed its colour into the region it hides.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4 * radius + 1, 4 * radius + 1))
    farthest_nearby = cv2.erode(pixel_disparity.astype(np.float32), kernel)
    return pixel_disparity - farthest_nearby <= jump


def propagate_background(
    rgb: np.ndarray, unknown: np.ndarray, sources: np.ndarray
) -> np.ndarray:
    """Give every ``unknown`` pixel the colour of its nearest ``sources`` pixel.

    One labelled distance transform (Voronoi fill) replaces an iterative flood:
    colour streaks in from the background side of each silhouette, continuing
    its texture. It is about 50x faster than the flood it replaced and avoids
    the flat grey that diffusion inpainting (Telea) produces.
    """
    if not unknown.any() or not sources.any():
        return rgb.copy()
    # Zero pixels are the seeds; every non-zero pixel is labelled with its
    # nearest seed. With DIST_LABEL_PIXEL, seeds are labelled 1..n in raster
    # order, which is the order of ``rgb[sources]``.
    seeds = np.where(sources, 0, 255).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        seeds, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    palette = rgb[sources]
    filled = rgb.copy()
    filled[unknown] = palette[labels[unknown] - 1]
    return filled


def background_layer(
    rgb: np.ndarray,
    pixel_disparity: np.ndarray,
    disparity: np.ndarray,
    sky_grid: np.ndarray,
    near_grid: np.ndarray,
    dilation_px: int,
    jump: float,
) -> BackgroundLayer:
    """Build the surface hidden behind foreground silhouettes.

    Args:
        rgb: padded image, HxWx3 uint8.
        pixel_disparity: padded per-pixel scene disparity.
        disparity: per-vertex scene disparity on the grid.
        sky_grid: per-vertex sky mask.
        near_grid: vertices on the near side of a disparity jump.
        dilation_px: how far behind a silhouette the layer extends, in pixels.
        jump: the profile's disparity jump threshold.
    """
    grid_height, grid_width = disparity.shape
    image_height, image_width = rgb.shape[:2]
    stride = image_width / max(1, grid_width - 1)
    cells = max(1, int(np.ceil(dilation_px / stride)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cells + 1, 2 * cells + 1))
    region = cv2.dilate(near_grid.astype(np.uint8), kernel).astype(bool) & ~sky_grid

    # Background is by definition farther, i.e. lower disparity. A min filter
    # (grey erosion) on disparity carries the far side of each edge across
    # the region the occluder hides. A box blur afterwards removes the blocky
    # plateaus the filter leaves.
    hidden = cv2.erode(disparity.astype(np.float32), kernel)
    hidden = cv2.blur(hidden, (cells + 1, cells + 1))
    hidden = np.minimum(hidden, disparity)
    layer_disparity = np.where(region, hidden, disparity).astype(np.float32)

    # Colour, at half resolution: the layer's texture is stored at half size
    # anyway, and the fill is 4x cheaper there.
    half = (max(1, image_width // 2), max(1, image_height // 2))
    small_rgb = cv2.resize(rgb, half, interpolation=cv2.INTER_AREA)
    small_disparity = cv2.resize(pixel_disparity, half, interpolation=cv2.INTER_NEAREST)
    radius = max(1, dilation_px // 2)
    unknown = occluder_pixels(small_disparity, radius, jump)
    sources = background_sources(small_disparity, radius, jump) & ~unknown
    layer_rgb = propagate_background(small_rgb, unknown, sources)
    return BackgroundLayer(region=region, disparity=layer_disparity, rgb=layer_rgb)
