"""Equirectangular geometry: the 360° canvas the surround is generated on.

Directions use the engine's camera frame (glTF: +Y up, looking down -Z).
Azimuth is 0 straight ahead and positive toward +X (right); elevation is
positive up. An equirectangular image maps azimuth linearly to columns
(-180° at the left edge) and elevation linearly to rows (+90° at the top).

The surround is generated through ordinary perspective *views* of this
canvas, because image models are trained on perspective photos, not on
equirectangular projections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .camera import Intrinsics


@lru_cache(maxsize=4)
def equirect_directions(width: int, height: int) -> np.ndarray:
    """(height, width, 3) unit directions at the pixel centres (cached and
    read-only: a 4096x2048 canvas is ~100 MB, and every view needs it)."""
    azimuth = (np.arange(width, dtype=np.float32) + 0.5) / width * 2 * np.pi - np.pi
    elevation = np.pi / 2 - (np.arange(height, dtype=np.float32) + 0.5) / height * np.pi
    a, e = np.meshgrid(azimuth, elevation)
    directions = np.stack([np.sin(a) * np.cos(e), np.sin(e), -np.cos(a) * np.cos(e)], axis=-1)
    directions.setflags(write=False)
    return directions


def direction_to_equirect(directions: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Float pixel coordinates (columns, rows) of unit ``directions``."""
    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    azimuth = np.arctan2(x, -z)
    elevation = np.arcsin(np.clip(y, -1.0, 1.0))
    columns = (azimuth + np.pi) / (2 * np.pi) * width - 0.5
    rows = (np.pi / 2 - elevation) / np.pi * height - 0.5
    return columns.astype(np.float32), rows.astype(np.float32)


def sample_equirect(
    image: np.ndarray, directions: np.ndarray, interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Sample an equirectangular ``image`` along ``directions``. Azimuth wraps;
    elevation is clamped, so straight down never blends in the zenith row."""
    height, width = image.shape[:2]
    columns, rows = direction_to_equirect(directions, width, height)
    rows = np.clip(rows, 0.0, height - 1.001).astype(np.float32)
    return cv2.remap(image, columns, rows, interpolation, borderMode=cv2.BORDER_WRAP)


def project_to_photo(camera: Intrinsics, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Photo pixel (u, v) of each direction, and whether it lies in the frame.
    The inverse of :meth:`Intrinsics.rays`."""
    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    if camera.projection == "cylindrical":
        horizontal = np.maximum(np.hypot(x, z), 1e-9)
        u = camera.cx + camera.fx * np.arctan2(x, -z)
        v = camera.cy - camera.fy * y / horizontal
        ahead = np.ones_like(u, dtype=bool)
    else:
        depth = np.maximum(-z, 1e-9)
        u = camera.cx + camera.fx * x / depth
        v = camera.cy - camera.fy * y / depth
        ahead = z < -1e-6
    inside = ahead & (u >= 0) & (u <= camera.width - 1) & (v >= 0) & (v <= camera.height - 1)
    return u.astype(np.float32), v.astype(np.float32), inside


def photo_to_equirect(
    rgb: np.ndarray, camera: Intrinsics, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """The photo placed on an equirectangular canvas at its true field of
    view. Returns (image, known): pixels outside the photo are black and not
    known."""
    directions = equirect_directions(width, height)
    u, v, inside = project_to_photo(camera, directions)
    # Area-like sampling when the canvas is coarser than the photo.
    scale = camera.fx * (2 * np.pi / width)
    source = rgb if scale <= 1.5 else cv2.GaussianBlur(rgb, (0, 0), 0.5 * scale)
    image = cv2.remap(source, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    image[~inside] = 0
    return image, inside


@dataclass(frozen=True)
class View:
    """A square perspective view of the panorama."""

    yaw_deg: float
    pitch_deg: float
    fov_deg: float
    size: int

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(forward, right, up) unit vectors in the camera frame."""
        yaw, pitch = math.radians(self.yaw_deg), math.radians(self.pitch_deg)
        forward = np.array(
            [math.sin(yaw) * math.cos(pitch), math.sin(pitch), -math.cos(yaw) * math.cos(pitch)]
        )
        # Right stays horizontal, also when looking straight up or down.
        right = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
        up = np.cross(right, forward)
        return forward, right, up / np.linalg.norm(up)

    def directions(self) -> np.ndarray:
        """(size, size, 3) unit directions through the view's pixel centres."""
        forward, right, up = self.basis()
        half = math.tan(math.radians(self.fov_deg) / 2)
        t = ((np.arange(self.size, dtype=np.float32) + 0.5) / self.size * 2 - 1) * half
        x, y = np.meshgrid(t, -t)
        rays = forward[None, None] + x[..., None] * right[None, None] + y[..., None] * up[None, None]
        return (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).astype(np.float32)

    def project(self, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """View pixel coordinates (i, j) of ``directions``, and whether each
        falls inside the view."""
        forward, right, up = self.basis()
        depth = directions @ forward
        half = math.tan(math.radians(self.fov_deg) / 2)
        safe = np.maximum(depth, 1e-6)
        i = ((directions @ right) / safe / half + 1) / 2 * self.size - 0.5
        j = (1 - (directions @ up) / safe / half) / 2 * self.size - 0.5
        inside = (depth > 1e-6) & (i >= 0) & (i <= self.size - 1) & (j >= 0) & (j <= self.size - 1)
        return i.astype(np.float32), j.astype(np.float32), inside


def render_view(panorama: np.ndarray, view: View, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """The panorama as seen through ``view``."""
    return sample_equirect(panorama, view.directions(), interpolation)


def photo_disparity_to_equirect(
    disparity: np.ndarray, camera: Intrinsics, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """The photo's disparity on the equirectangular canvas, as *inverse
    distance* (a pinhole's disparity is inverse depth along its axis, which
    differs from inverse distance by the ray's cosine)."""
    directions = equirect_directions(width, height)
    u, v, inside = project_to_photo(camera, directions)
    sampled = cv2.remap(disparity.astype(np.float32), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if camera.projection == "cylindrical":
        cosine = np.sqrt(np.maximum(1.0 - directions[..., 1] ** 2, 0.0))
    else:
        cosine = np.maximum(-directions[..., 2], 0.0)
    out = np.where(inside, sampled * cosine, 0.0).astype(np.float32)
    return out, inside


def _fit_affine(raw: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """``reference ≈ a * raw + b`` by trimmed least squares, with ``a > 0``."""
    a, b = 1.0, 0.0
    keep = np.ones(raw.shape, dtype=bool)
    for _ in range(3):
        x, y = raw[keep], reference[keep]
        if x.size < 50 or np.ptp(x) < 1e-6:
            break
        a, b = np.polyfit(x, y, 1)
        residual = np.abs(a * raw + b - reference)
        keep = residual <= np.percentile(residual, 80)
    if not np.isfinite(a) or a <= 1e-6:
        # A degenerate overlap (all sky, all one depth): match medians instead.
        scale = float(np.median(reference)) / max(float(np.median(raw)), 1e-6)
        return max(scale, 1e-6), 0.0
    return float(a), float(b)


def stitch_disparity(
    panorama: np.ndarray,
    known_disparity: np.ndarray,
    known: np.ndarray,
    views: list["View"],
    estimate,
    sky_threshold: float = 0.02,
    anchor: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse-distance disparity for the whole panorama.

    Each view's relative depth (``estimate(rgb) -> disparity``, affine-
    invariant, as monocular networks give it) is aligned by scale and shift
    to what is already known where they overlap (first the photo itself),
    then fills the pixels still unknown. Disparity is fitted in each view's
    own inverse-*depth* space and converted to inverse distance, so views
    looking different ways agree.

    ``anchor`` pixels of ``known_disparity`` guide the alignment where no
    view has been placed yet, but are not kept: the rendered depth of
    surfaces that were repainted, right in scale but torn in detail.
    """
    height, width = known_disparity.shape
    directions_eq = equirect_directions(width, height)
    # Views are blended where they overlap, weighted toward each view's
    # centre, so neighbouring views meet without a step (a step would tear
    # the mesh open along every view boundary).
    total = np.zeros((height, width), np.float32)
    weights = np.zeros((height, width), np.float32)
    sky_votes = np.zeros((height, width), np.float32)

    def current() -> tuple[np.ndarray, np.ndarray]:
        estimate_now = np.where(weights > 0, total / np.maximum(weights, 1e-6), 0.0)
        return np.where(known, known_disparity, estimate_now).astype(np.float32), known | (weights > 0)

    for view in views:
        directions = view.directions()
        forward, _, _ = view.basis()
        cosine = np.maximum(directions @ forward, 1e-3)
        rgb = render_view(panorama, view)
        raw = estimate(rgb).astype(np.float32)
        if raw.shape != rgb.shape[:2]:
            raw = cv2.resize(raw, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        span = float(np.percentile(raw, 99)) or 1.0
        sky = raw < sky_threshold * span
        result, filled = current()
        if anchor is not None:
            guide = anchor & ~filled
            result = np.where(guide, known_disparity, result)
            filled = filled | guide
        reference = render_view(result, view) / cosine
        overlap = render_view(filled.astype(np.uint8), view, cv2.INTER_NEAREST).astype(bool)
        usable = overlap & ~sky & (reference > 1e-4)
        a, b = _fit_affine(raw[usable], reference[usable]) if usable.sum() >= 50 else (1.0 / span, 0.0)
        aligned = np.clip(a * raw + b, 0.0, None) * cosine
        aligned[sky] = 0.0
        # Accumulate into the canvas with a weight that falls to zero at the
        # view's border.
        i, j, inside = view.project(directions_eq)
        border = np.minimum(np.minimum(i, j), np.minimum(view.size - 1 - i, view.size - 1 - j))
        weight = np.where(inside, np.clip(border / (0.25 * view.size), 0.0, 1.0) ** 2, 0.0).astype(np.float32)
        sampled = cv2.remap(aligned.astype(np.float32), i, j, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sky_view = cv2.remap(sky.astype(np.float32), i, j, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        total += sampled * weight
        sky_votes += sky_view * weight
        weights += weight
    result, _ = current()
    # Where the views mostly called it sky, it is sky: averaging a sky (0)
    # with a near value would put a hazy shell at a made-up distance.
    sky_mask = ~known & (sky_votes > 0.5 * np.maximum(weights, 1e-6))
    result[sky_mask] = 0.0
    return result


#: Sky colours are summarised by this many clusters, and a pixel this close
#: to one of them (CIE Lab distance) is sky-coloured.
SKY_CLUSTERS = 12
SKY_COLOUR_DISTANCE = 12.0


def separate_sky(
    disparity: np.ndarray,
    rgb: np.ndarray,
    editable: np.ndarray | None = None,
    band_fraction: float = 0.03,
) -> np.ndarray:
    """``disparity`` (equirectangular, 0 = sky) with sky-coloured pixels at
    the edge of the land above the horizon turned back into sky.

    A monocular depth network gives the sky around and through a tree's crown
    (and beside a post, a roof, a rock) the object's own depth: one flat
    plateau, so no depth test finds it. Meshed, that halo of sky is glued to
    the object, and from a few metres away it detaches as patches of
    shifted cloud and confetti in the crown. Colour separates them: the
    confident sky (far from any land) is summarised as a palette, and land
    pixels near the sky that match it are sky. Calling a little land sky
    costs little (it shows at infinity, where far land nearly is); sky glued
    to near land tears as soon as you move. Indoors there is no confident
    sky, and nothing changes.

    ``editable`` limits which pixels may change (default all).
    """
    height, width = disparity.shape
    land = disparity > 0
    band = max(2, int(band_fraction * width))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    sky_core = ~cv2.dilate(land.astype(np.uint8), kernel).astype(bool)
    if sky_core.sum() < 500:
        return disparity
    above = (np.arange(height) < height // 2)[:, None]
    rim = land & cv2.dilate((~land).astype(np.uint8), kernel).astype(bool) & above
    if editable is not None:
        rim &= editable
    if not rim.any():
        return disparity
    if rgb.shape[:2] != disparity.shape:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    samples = lab[sky_core]
    if len(samples) > 20000:
        samples = samples[np.random.default_rng(0).choice(len(samples), 20000, replace=False)]
    clusters = min(SKY_CLUSTERS, len(samples))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centres = cv2.kmeans(samples, clusters, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    pixels = lab[rim]
    nearest = np.min(np.linalg.norm(pixels[:, None, :] - centres[None, :, :], axis=-1), axis=1)
    out = disparity.copy()
    rows, columns = np.nonzero(rim)
    to_sky = nearest < SKY_COLOUR_DISTANCE
    out[rows[to_sky], columns[to_sky]] = 0.0
    return out


def blend_view(
    panorama: np.ndarray,
    known: np.ndarray,
    protected: np.ndarray,
    image: np.ndarray,
    alpha: np.ndarray,
    view: View,
    margin: float = 0.04,
) -> np.ndarray:
    """Write a generated ``image`` (seen through ``view``) into ``panorama``
    in place, and return the newly filled mask.

    * Unknown pixels in the view's interior take the image and become known.
    * Known pixels not ``protected`` (the photo itself is) are blended by
      ``alpha`` (view space, 0..1), tapered to nothing at the view's border,
      so the band the model repainted fades into what was there.
    * The border ``margin`` fills nothing new: models are least reliable at
      their edges, and the next view overlaps there anyway.
    """
    height, width = panorama.shape[:2]
    directions = equirect_directions(width, height)
    i, j, inside = view.project(directions)
    edge = max(1.0, margin * view.size)
    distance_to_border = np.minimum(np.minimum(i, j), np.minimum(view.size - 1 - i, view.size - 1 - j))
    fill = inside & (distance_to_border >= edge) & ~known
    blend = inside & known & ~protected
    if not (fill.any() or blend.any()):
        return fill
    sampled = cv2.remap(image, i, j, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    weight = cv2.remap(alpha.astype(np.float32), i, j, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    taper = np.clip(distance_to_border / edge, 0.0, 1.0)
    weight = (weight * taper * taper * (3 - 2 * taper))[blend][:, None]
    panorama[blend] = (panorama[blend] * (1.0 - weight) + sampled[blend] * weight).astype(panorama.dtype)
    panorama[fill] = sampled[fill].astype(panorama.dtype)
    known |= fill
    return fill


def paste_view(
    panorama: np.ndarray,
    known: np.ndarray,
    image: np.ndarray,
    view: View,
    margin: float = 0.04,
) -> np.ndarray:
    """Write ``image`` (seen through ``view``) into the unknown pixels of
    ``panorama`` in place, and return the newly filled mask.

    A thin border of each view is skipped: image models are least reliable at
    their edges, and the next view overlaps there anyway.
    """
    height, width = panorama.shape[:2]
    directions = equirect_directions(width, height)
    i, j, inside = view.project(directions)
    edge = margin * view.size
    inside &= (i >= edge) & (i <= view.size - 1 - edge) & (j >= edge) & (j <= view.size - 1 - edge)
    target = inside & ~known
    if not target.any():
        return target
    sampled = cv2.remap(image, i, j, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    panorama[target] = sampled[target]
    known |= target
    return target
