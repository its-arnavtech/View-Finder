"""The world around the photo: frame extension, ground, sky dome and fog."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Fraction of the frame, measured in from each edge, whose disparity gradient
# is extrapolated across the pad.
_EDGE_BAND = 0.06


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _pad_weight(height: int, width: int, pad_y: int, pad_x: int) -> np.ndarray:
    """0 inside the original frame, rising smoothly to 1 at the outer border."""
    ys = np.arange(height + 2 * pad_y, dtype=np.float32)
    xs = np.arange(width + 2 * pad_x, dtype=np.float32)
    dy = np.maximum(np.maximum(pad_y - ys, ys - (pad_y + height - 1)), 0.0) / max(pad_y, 1)
    dx = np.maximum(np.maximum(pad_x - xs, xs - (pad_x + width - 1)), 0.0) / max(pad_x, 1)
    return _smoothstep(np.maximum(dy[:, None], dx[None, :]))


def _extrapolate_disparity(disparity: np.ndarray, pad_y: int, pad_x: int) -> np.ndarray:
    """Pad disparity by continuing each border's local gradient.

    Disparity of a plane is affine in pixel coordinates, so linear
    extrapolation continues a floor, wall or ceiling the way the real surface
    would. Replicating the edge value would instead build a vertical curtain.
    The gradient fades out across the pad so curved surfaces don't run away.
    """
    height, width = disparity.shape
    band_y = max(2, int(height * _EDGE_BAND))
    band_x = max(2, int(width * _EDGE_BAND))
    out = cv2.copyMakeBorder(disparity, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
    sigma = max(1.0, width / 60.0)

    def slope_rows(edge: np.ndarray, inner: np.ndarray, rows: int) -> np.ndarray:
        slope = (edge - inner) / rows
        return cv2.GaussianBlur(slope[None, :], (0, 0), sigma)[0]

    top = slope_rows(disparity[0], disparity[band_y], band_y)
    bottom = slope_rows(disparity[-1], disparity[-1 - band_y], band_y)
    left = slope_rows(disparity[:, 0], disparity[:, band_x], band_x)
    right = slope_rows(disparity[:, -1], disparity[:, -1 - band_x], band_x)

    def ramp(n: int) -> np.ndarray:
        # Distance travelled with a gradient that fades linearly to zero.
        t = np.arange(1, n + 1, dtype=np.float32)
        return t - t * t / (2.0 * n)

    if pad_y:
        steps = ramp(pad_y)
        out[:pad_y, pad_x : pad_x + width] += (steps[::-1, None] * top[None, :])
        out[pad_y + height :, pad_x : pad_x + width] += (steps[:, None] * bottom[None, :])
    if pad_x:
        steps = ramp(pad_x)
        out[pad_y : pad_y + height, :pad_x] += (left[:, None] * steps[None, ::-1])
        out[pad_y : pad_y + height, pad_x + width :] += (right[:, None] * steps[None, :])
    # Corners: average the two neighbouring continuations by re-blurring.
    corner = cv2.GaussianBlur(out, (0, 0), max(pad_x, pad_y, 1) / 2.0)
    corner_mask = np.zeros_like(out, dtype=bool)
    corner_mask[:pad_y, :pad_x] = corner_mask[:pad_y, -pad_x or None :] = True
    corner_mask[-pad_y or None :, :pad_x] = corner_mask[-pad_y or None :, -pad_x or None :] = True
    out[corner_mask] = corner[corner_mask]
    return np.clip(out, 0.0, 1.3).astype(np.float32)


@dataclass(frozen=True)
class PaddedCanvas:
    rgb: np.ndarray
    disparity: np.ndarray  # normalised [0, ~1.3]
    sky: np.ndarray
    pad_x: int
    pad_y: int
    original: np.ndarray  # bool mask of the original frame within the canvas


def pad_size(height: int, width: int, pad_fraction: float) -> tuple[int, int]:
    """(pad_y, pad_x) added on each side."""
    return int(round(height * pad_fraction)), int(round(width * pad_fraction))


def extend_frame(
    rgb: np.ndarray,
    disparity: np.ndarray,
    sky: np.ndarray,
    pad_fraction: float,
    fade_to: np.ndarray | None = None,
) -> PaddedCanvas:
    """Pad the photo on every side with a soft, depth-consistent continuation.

    Colour goes mirrored → blurred → ``fade_to`` (the dome's colour in each
    pad pixel's direction) across the pad, so that when the viewer turns past
    the photo's edge, the edge dissolves into the surrounding sky and ground
    instead of ending in a hard wall.
    """
    height, width = disparity.shape
    pad_y, pad_x = pad_size(height, width, pad_fraction)
    weight = _pad_weight(height, width, pad_y, pad_x)[..., None]

    mirrored = cv2.copyMakeBorder(rgb, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT_101)
    blurred = cv2.GaussianBlur(mirrored, (0, 0), max(4.0, max(height, width) / 40.0))
    canvas = mirrored * (1.0 - weight) + blurred * weight
    if fade_to is not None:
        fade_weight = 0.9 * weight**2
        canvas = canvas * (1.0 - fade_weight) + fade_to.astype(np.float32) * fade_weight
    canvas_rgb = np.clip(canvas, 0, 255).astype(np.uint8)

    sky_padded = cv2.copyMakeBorder(
        sky.astype(np.uint8), pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
    ).astype(bool)
    padded_disparity = _extrapolate_disparity(disparity, pad_y, pad_x)
    padded_disparity[sky_padded] = 0.0

    original = np.zeros(sky_padded.shape, dtype=bool)
    original[pad_y : pad_y + height, pad_x : pad_x + width] = True
    return PaddedCanvas(canvas_rgb, padded_disparity, sky_padded, pad_x, pad_y, original)


@dataclass(frozen=True)
class GroundPlane:
    accepted: bool
    normal: np.ndarray  # unit, pointing up (toward the camera side)
    offset: float  # plane: normal . p + offset = 0
    inlier_ratio: float
    tilt_deg: float
    color: np.ndarray  # RGB in [0, 1]
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "normal": [round(float(v), 4) for v in self.normal],
            "offset": round(float(self.offset), 4),
            "camera_height": round(abs(float(self.offset)), 4),
            "inlier_ratio": round(self.inlier_ratio, 3),
            "tilt_deg": round(self.tilt_deg, 2),
            "reason": self.reason,
        }


MIN_GROUND_INLIERS = 0.45
MAX_GROUND_TILT_DEG = 40.0
# A real ground has (almost) nothing visible beneath it. A plane fitted to a
# car dashboard, a table top or a balcony rail does, and extending such a
# plane would slice a flat slab through the real scene behind it.
MAX_BELOW_GROUND = 0.05


def fit_ground(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    scene_points: np.ndarray | None = None,
    iterations: int = 200,
    seed: int = 0,
) -> GroundPlane:
    """RANSAC a ground plane through points sampled from the lower frame.

    Accepted only if enough points agree, the plane faces roughly up (its
    normal within 40 degrees of camera-up), and no more than 5% of
    ``scene_points`` (the whole visible scene) lie clearly below it.
    Close-ups, aerial shots and photos taken over a foreground surface fail
    the gates, and then no ground is invented.
    """
    up = np.array([0.0, 1.0, 0.0])
    fallback_color = colors.mean(axis=0) if len(colors) else np.array([0.4, 0.4, 0.4])
    if len(points) < 30:
        return GroundPlane(False, up, 0.0, 0.0, 0.0, fallback_color, "too few points")
    rng = np.random.default_rng(seed)
    if len(points) > 4000:
        keep = rng.choice(len(points), 4000, replace=False)
        points, colors = points[keep], colors[keep]
    tolerance = 0.03 * float(np.median(-points[:, 2]))
    best_inliers = np.zeros(len(points), dtype=bool)
    best_normal, best_offset = up, 0.0
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        if normal @ up < 0:
            normal = -normal
        if np.degrees(np.arccos(np.clip(normal @ up, -1, 1))) > MAX_GROUND_TILT_DEG:
            continue
        offset = -float(normal @ a)
        inliers = np.abs(points @ normal + offset) < tolerance
        if inliers.sum() > best_inliers.sum():
            best_inliers, best_normal, best_offset = inliers, normal, offset
    ratio = float(best_inliers.mean())
    if ratio == 0.0:
        return GroundPlane(False, up, 0.0, 0.0, 0.0, fallback_color, "no up-facing plane")

    # Least-squares refinement on the inliers.
    inlier_points = points[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vt = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vt[-1]
    if normal @ up < 0:
        normal = -normal
    offset = -float(normal @ centroid)
    tilt = float(np.degrees(np.arccos(np.clip(normal @ up, -1, 1))))
    color = np.median(colors[best_inliers], axis=0)
    below = 0.0
    if scene_points is not None and len(scene_points):
        heights = scene_points @ normal + offset
        below = float(np.mean(heights < -3.0 * tolerance))
    accepted = (
        ratio >= MIN_GROUND_INLIERS
        and tilt <= MAX_GROUND_TILT_DEG
        and offset > 0
        and below <= MAX_BELOW_GROUND
    )
    if accepted:
        reason = "ok"
    elif offset <= 0:
        reason = "plane is above the camera"
    elif tilt > MAX_GROUND_TILT_DEG:
        reason = f"plane tilted {tilt:.0f} deg from camera-up"
    elif below > MAX_BELOW_GROUND:
        reason = f"{below:.0%} of the scene lies below the fitted plane (a foreground surface, not ground)"
    else:
        reason = f"only {ratio:.0%} of lower-frame points lie on a plane"
    return GroundPlane(accepted, normal, offset, ratio, tilt, color, reason)


def fog_density(far: float, fogged_at_far: float = 0.6) -> float:
    """Density for squared-exponential fog ``1 - exp(-(d*x)^2)`` hitting
    ``fogged_at_far`` at distance ``far``. Squared keeps the near field clear
    (~1.4% at far/8) while still hiding the mesh/dome seam."""
    return float(np.sqrt(-np.log(1.0 - fogged_at_far)) / far)


def _to_rgba8(colors: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
    alpha = np.full((len(rgb), 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=1)


def dome_colors(
    directions: np.ndarray,
    up: np.ndarray,
    zenith: np.ndarray,
    horizon: np.ndarray,
    ground: np.ndarray,
) -> np.ndarray:
    """Dome colour (RGB in [0, 1]) seen along unit ``directions`` (..., 3)."""
    sine = (directions @ (up / np.linalg.norm(up)))[..., None]
    colors = horizon + (zenith - horizon) * _smoothstep(sine / 0.6)
    return colors + (ground - colors) * _smoothstep(-sine / 0.25)


def dome_mesh(
    radius: float,
    segments: int,
    up: np.ndarray,
    zenith: np.ndarray,
    horizon: np.ndarray,
    ground: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A UV sphere around the origin, vertex-coloured by elevation about ``up``."""
    up = up / np.linalg.norm(up)
    # Orthonormal basis with `up` as the pole.
    helper = np.array([0.0, 0.0, -1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    east = np.cross(helper, up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)

    rings = max(6, segments // 2)
    elevations = np.linspace(-np.pi / 2, np.pi / 2, rings + 1)
    azimuths = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    elevation, azimuth = np.meshgrid(elevations, azimuths, indexing="ij")
    direction = (
        np.cos(elevation)[..., None] * (np.cos(azimuth)[..., None] * east + np.sin(azimuth)[..., None] * north)
        + np.sin(elevation)[..., None] * up
    )
    positions = (direction * radius).reshape(-1, 3).astype(np.float32)
    colors = dome_colors(direction.reshape(-1, 3), up, zenith, horizon, ground)
    columns = segments + 1
    a = (np.arange(rings)[:, None] * columns + np.arange(segments)[None, :]).reshape(-1)
    indices = np.stack([a, a + columns, a + 1, a + 1, a + columns, a + columns + 1], axis=1)
    return positions, _to_rgba8(colors), indices.reshape(-1).astype(np.uint32)


def ground_disc_mesh(
    plane: GroundPlane,
    radius: float,
    segments: int,
    horizon: np.ndarray,
    sink: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A disc on the ground plane under the camera, fading to the horizon colour.

    Sunk by ``sink`` along the normal so the photo's own ground (which lies on
    the same plane) always wins the depth test instead of z-fighting.
    """
    normal = plane.normal
    center = -normal * (plane.offset + sink)  # foot of the camera on the sunk plane
    helper = np.array([0.0, 0.0, -1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    east = np.cross(helper, normal)
    east /= np.linalg.norm(east)
    north = np.cross(normal, east)
    radii = radius * np.array([0.0, 0.08, 0.2, 0.4, 0.7, 1.0])
    angles = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    r, a = np.meshgrid(radii, angles, indexing="ij")
    positions = center + r[..., None] * (np.cos(a)[..., None] * east + np.sin(a)[..., None] * north)
    t = _smoothstep((r / radius) / 0.7).reshape(-1, 1)
    colors = plane.color + (horizon - plane.color) * t
    columns = segments + 1
    rings = len(radii) - 1
    base = (np.arange(rings)[:, None] * columns + np.arange(segments)[None, :]).reshape(-1)
    indices = np.stack(
        [base, base + 1, base + columns, base + 1, base + columns + 1, base + columns], axis=1
    )
    return (
        positions.reshape(-1, 3).astype(np.float32),
        _to_rgba8(colors),
        indices.reshape(-1).astype(np.uint32),
    )
