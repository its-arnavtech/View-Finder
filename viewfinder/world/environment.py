"""The world around the photo: frame extension, ground, sky dome and fog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .camera import Intrinsics

# Fraction of the frame, measured in from each edge, whose disparity gradient
# is extrapolated across the pad.
_EDGE_BAND = 0.06


def fast_blur(image: np.ndarray, sigma: float, border: int = cv2.BORDER_REFLECT) -> np.ndarray:
    """Large-sigma Gaussian blur computed at reduced resolution.

    A sigma-25 blur on a full canvas costs hundreds of milliseconds. Its
    result has no detail finer than ~sigma pixels, so blurring a copy
    downsampled by ~sigma/3 and upsampling it again looks identical and is
    an order of magnitude cheaper.
    """
    factor = int(max(1, min(8, sigma // 3)))
    if factor == 1:
        return cv2.GaussianBlur(image, (0, 0), sigma, borderType=border)
    height, width = image.shape[:2]
    small = cv2.resize(
        image, (max(1, width // factor), max(1, height // factor)), interpolation=cv2.INTER_AREA
    )
    small = cv2.GaussianBlur(small, (0, 0), sigma / factor, borderType=border)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


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
    if pad_x and pad_y:
        corner = fast_blur(out, max(pad_x, pad_y) / 2.0)
        corner_mask = np.zeros_like(out, dtype=bool)
        corner_mask[:pad_y, :pad_x] = corner_mask[:pad_y, -pad_x:] = True
        corner_mask[-pad_y:, :pad_x] = corner_mask[-pad_y:, -pad_x:] = True
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
    pad: tuple[int, int],
    fade_to: np.ndarray | None = None,
) -> PaddedCanvas:
    """Pad the photo on every side with a soft, depth-consistent continuation.

    Colour goes mirrored → blurred → ``fade_to`` (the dome's colour in each
    pad pixel's direction) across the pad, so that when the viewer turns past
    the photo's edge, the edge dissolves into the surrounding sky and ground
    instead of ending in a hard wall.
    """
    height, width = disparity.shape
    pad_y, pad_x = pad
    weight = _pad_weight(height, width, pad_y, pad_x)[..., None]

    mirrored = cv2.copyMakeBorder(rgb, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT_101)
    blurred = fast_blur(mirrored, max(4.0, max(height, width) / 40.0))
    canvas = mirrored * (1.0 - weight) + blurred * weight
    if fade_to is not None:
        fade_weight = 0.9 * weight**2
        canvas = canvas * (1.0 - fade_weight) + fade_to.astype(np.float32) * fade_weight
    canvas_rgb = np.clip(canvas, 0, 255).astype(np.uint8)
    # The original frame is copied back exactly: the blends above are no-ops
    # there in theory, but float round-trips must never alter the photo.
    canvas_rgb[pad_y : pad_y + height, pad_x : pad_x + width] = rgb

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


def direction_angles(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(azimuth, elevation) of camera-frame unit directions.

    Azimuth is 0 straight ahead (-Z) and positive toward +X (right);
    elevation is positive up. Matches the viewer's yaw/pitch convention.
    """
    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    return np.arctan2(x, -z), np.arcsin(np.clip(y, -1.0, 1.0))


def _reflect(coordinate: np.ndarray, size: int) -> np.ndarray:
    """Ping-pong ``coordinate`` into [0, size - 1] (mirror tiling)."""
    period = 2.0 * (size - 1)
    folded = np.mod(coordinate, period)
    return np.where(folded > size - 1, period - folded, folded)


def _wrap_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that wraps horizontally (azimuth is periodic)."""
    margin = int(3 * sigma) + 1
    wrapped = np.concatenate([image[:, -margin:], image, image[:, :margin]], axis=1)
    return fast_blur(wrapped, sigma)[:, margin:-margin]


def environment_map(
    rgb: np.ndarray,
    camera: "Intrinsics",
    size: tuple[int, int],
    zenith: np.ndarray,
    ground: np.ndarray,
) -> np.ndarray:
    """A 360° equirectangular surround synthesised from the photo.

    Straight ahead it reproduces the photo. Past each side edge the photo is
    mirrored outward and blurred more the farther you turn. Behind the
    viewer, the right-edge and left-edge continuations cross-fade so the wrap
    is seamless. Above and below the frame it fades into the photo's zenith
    and ground tones. This is an *ambient* continuation, deliberately
    blurred, not a claim about what was really there. A full 360° panorama
    covers every azimuth itself, so it has no back arc to invent.
    """
    height, width = rgb.shape[:2]
    env_width, env_height = size
    fx, fy = camera.fx, camera.fy
    cx, cy = width / 2.0, height / 2.0
    cylindrical = camera.projection == "cylindrical"
    half_h = min(np.pi, cx / fx) if cylindrical else float(np.arctan(cx / fx))
    half_v = float(np.arctan(cy / fy))

    azimuth = (np.arange(env_width, dtype=np.float32) + 0.5) / env_width * 2 * np.pi - np.pi
    elevation = np.pi / 2 - (np.arange(env_height, dtype=np.float32) + 0.5) / env_height * np.pi
    a, e = np.meshgrid(azimuth, elevation)

    # Unwrap azimuth so the back arc runs continuously from the right edge
    # (half_h) to the left edge (2*pi - half_h).
    unwrapped = np.where(a < -half_h, a + 2 * np.pi, a)
    inside = unwrapped <= half_h
    back_span = max(2 * np.pi - 2 * half_h, 1e-6)
    beyond = np.clip(unwrapped - half_h, 0.0, back_span)
    pixels_per_radian = width / (2 * half_h)

    clipped = np.clip(a, -half_h, half_h)
    columns_right = _reflect((width - 1) - beyond * pixels_per_radian, width)
    columns_left = _reflect((back_span - beyond) * pixels_per_radian, width)
    tan_e = np.tan(np.clip(e, -1.5, 1.5))
    if cylindrical:
        # Columns are linear in azimuth; rows follow tan(elevation) directly.
        columns_in = np.clip(cx + fx * clipped, 0, width - 1)
        rows = np.clip(cy - fy * tan_e, 0, height - 1)
    else:
        in_cos = np.cos(clipped)
        columns_in = cx + fx * np.tan(clipped)
        rows = np.clip(cy - fy * tan_e / np.where(inside, in_cos, np.cos(half_h)), 0, height - 1)

    def remap(columns: np.ndarray) -> np.ndarray:
        return cv2.remap(
            rgb, columns.astype(np.float32), rows.astype(np.float32),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float32)

    right, left = remap(np.where(inside, columns_in, columns_right)), remap(columns_left)
    t = np.where(inside, 0.0, beyond / back_span)[..., None]
    env = right * (1.0 - t) + left * t

    # Blur by how far outside the photo each direction is.
    excess = np.maximum(
        np.where(inside, 0.0, np.minimum(beyond, back_span - beyond)),
        np.maximum(np.abs(e) - half_v, 0.0),
    )
    soft = _wrap_blur(env, max(1.0, env_width / 400.0))
    heavy = _wrap_blur(env, env_width / 40.0)
    w = _smoothstep(excess / np.radians(40.0))[..., None]
    env = soft * (1.0 - w) + heavy * w

    # Above/below the frame, fade to the photo's sky and ground tones.
    up_t = _smoothstep((e - half_v) / max(1e-3, np.pi / 2 - half_v))[..., None]
    down_t = _smoothstep((-e - half_v) / max(1e-3, np.pi / 2 - half_v))[..., None]
    env = env + (zenith * 255.0 - env) * up_t * 0.85
    env = env + (ground * 255.0 - env) * down_t * 0.85
    return np.clip(env, 0, 255).astype(np.uint8)


def sample_environment(environment: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Environment colour (float RGB 0..255) along camera-frame unit directions."""
    env_height, env_width = environment.shape[:2]
    azimuth, elevation = direction_angles(directions)
    columns = ((azimuth + np.pi) / (2 * np.pi) * env_width - 0.5).astype(np.float32)
    # Azimuth wraps (BORDER_WRAP below), elevation must not: clamp rows so
    # straight down never blends in the zenith row across the pole.
    rows = np.clip((np.pi / 2 - elevation) / np.pi * env_height - 0.5, 0, env_height - 1).astype(np.float32)
    return cv2.remap(
        environment, columns, rows, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
    ).astype(np.float32)


def environment_dome(radius: float, segments: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UV sphere in the camera frame, UV-mapped to an equirectangular texture."""
    rings = max(8, segments // 2)
    elevations = np.linspace(np.pi / 2, -np.pi / 2, rings + 1)
    azimuths = np.linspace(-np.pi, np.pi, segments + 1)
    e, a = np.meshgrid(elevations, azimuths, indexing="ij")
    directions = np.stack([np.sin(a) * np.cos(e), np.sin(e), -np.cos(a) * np.cos(e)], axis=-1)
    uvs = np.stack([(a + np.pi) / (2 * np.pi), (np.pi / 2 - e) / np.pi], axis=-1)
    columns = segments + 1
    base = (np.arange(rings)[:, None] * columns + np.arange(segments)[None, :]).reshape(-1)
    indices = np.stack(
        [base, base + columns, base + 1, base + 1, base + columns, base + columns + 1], axis=1
    )
    return (
        (directions * radius).reshape(-1, 3).astype(np.float32),
        uvs.reshape(-1, 2).astype(np.float32),
        indices.reshape(-1).astype(np.uint32),
    )


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
    environment: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A disc on the ground plane under the camera.

    Sunk by ``sink`` along the normal so the photo's own ground (which lies on
    the same plane) always wins the depth test instead of z-fighting. With an
    ``environment`` map, each vertex takes the surround's colour in its
    direction from the camera, so ground, frame padding and surround all
    agree. Without one, it fades from the plane's colour to the horizon.
    """
    normal = plane.normal
    center = -normal * (plane.offset + sink)  # foot of the camera on the sunk plane
    helper = np.array([0.0, 0.0, -1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    east = np.cross(helper, normal)
    east /= np.linalg.norm(east)
    north = np.cross(normal, east)
    radii = radius * np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.0])
    angles = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    r, a = np.meshgrid(radii, angles, indexing="ij")
    positions = center + r[..., None] * (np.cos(a)[..., None] * east + np.sin(a)[..., None] * north)
    if environment is not None:
        flat = positions.reshape(-1, 3)
        directions = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-9)
        colors = sample_environment(environment, directions[None]).reshape(-1, 3) / 255.0
    else:
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
