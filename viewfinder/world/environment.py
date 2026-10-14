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


def _pad_distances(height: int, width: int, pad_y: int, pad_x: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row and per-column distance into the pad, 0 at the photo edge and 1
    at the outer border (0 everywhere inside the photo)."""
    ys = np.arange(height + 2 * pad_y, dtype=np.float32)
    xs = np.arange(width + 2 * pad_x, dtype=np.float32)
    dy = np.maximum(np.maximum(pad_y - ys, ys - (pad_y + height - 1)), 0.0) / max(pad_y, 1)
    dx = np.maximum(np.maximum(pad_x - xs, xs - (pad_x + width - 1)), 0.0) / max(pad_x, 1)
    return dy, dx


def _pad_weight(height: int, width: int, pad_y: int, pad_x: int) -> np.ndarray:
    """0 inside the original frame, rising smoothly to 1 at the outer border."""
    dy, dx = _pad_distances(height, width, pad_y, pad_x)
    return _smoothstep(np.maximum(dy[:, None], dx[None, :]))


def _box_gaussian(image: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    """Gaussian blur along one axis, approximated by three box passes. Its
    cost does not depend on ``sigma``; a true Gaussian with sigma ~100 px
    needs a ~600 px kernel."""
    width = max(1, int(round(np.sqrt(4.0 * sigma * sigma + 1.0))))  # 3 boxes ~ sigma
    size = (1, width) if axis == 0 else (width, 1)
    out = image
    for _ in range(3):
        out = cv2.blur(out, size, borderType=cv2.BORDER_REPLICATE)
    return out


def _progressive_blur(image: np.ndarray, distance: np.ndarray, sigmas: tuple[float, ...], axis: int) -> np.ndarray:
    """Blur along one ``axis`` (0 = vertical, 1 = horizontal) with a strength
    that grows with ``distance`` (0..1, broadcastable to the image)."""
    levels = [image.astype(np.float32)]
    levels += [_box_gaussian(levels[0], sigma, axis) for sigma in sigmas]
    position = np.broadcast_to(np.clip(distance, 0.0, 1.0) * (len(levels) - 1), image.shape[:2])
    # Linear interpolation between the two nearest levels (tent weights).
    out = np.zeros_like(levels[0])
    for index, level in enumerate(levels):
        out += level * np.clip(1.0 - np.abs(position - index), 0.0, 1.0)[..., None]
    return out


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

    Colour goes stretched edge → blurred → ``fade_to`` (the dome's colour in each
    pad pixel's direction) across the pad, so that when the viewer turns past
    the photo's edge, the edge dissolves into the surrounding sky and ground
    instead of ending in a hard wall.
    """
    height, width = disparity.shape
    pad_y, pad_x = pad
    weight = _pad_weight(height, width, pad_y, pad_x)[..., None]
    dy, dx = _pad_distances(height, width, pad_y, pad_x)

    # Edge pixels are stretched outward, never mirrored: a mirrored border
    # puts a symmetric copy of the photo's edge right beside it, and that
    # symmetry is instantly visible. Stretching has its own artefact (every
    # row of a side pad is one colour, so it reads as stripes), so each pad
    # is blurred *along* its edge, more the farther out it is: side pads
    # vertically, top and bottom pads horizontally.
    stretched = cv2.copyMakeBorder(rgb, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
    long_side = max(height, width)
    sigmas = (long_side / 50.0, long_side / 18.0, long_side / 8.0)
    # Only the pad strips need it: side strips (full height) and end strips
    # (full width). Inside the frame the photo is copied back below.
    sides = stretched.astype(np.float32)
    ends = stretched.astype(np.float32)
    total_w = width + 2 * pad_x
    total_h = height + 2 * pad_y
    for columns in (slice(0, pad_x), slice(total_w - pad_x, total_w)) if pad_x else ():
        sides[:, columns] = _progressive_blur(stretched[:, columns], dx[None, columns], sigmas, axis=0)
    for rows in (slice(0, pad_y), slice(total_h - pad_y, total_h)) if pad_y else ():
        ends[rows, :] = _progressive_blur(stretched[rows, :], dy[rows, None], sigmas, axis=1)
    # In the corners both apply; weight by which edge is nearer.
    side_share = (dx[None, :] / np.maximum(dx[None, :] + dy[:, None], 1e-6))[..., None]
    canvas = sides * side_share + ends * (1.0 - side_share)
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


def _wrap_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that wraps horizontally (azimuth is periodic)."""
    margin = int(3 * sigma) + 1
    wrapped = np.concatenate([image[:, -margin:], image, image[:, :margin]], axis=1)
    return fast_blur(wrapped, sigma)[:, margin:-margin]


def _row_profiles(rgb: np.ndarray, sigma_fraction: float) -> np.ndarray:
    """Per-row median colours of vertical strips, smoothed along rows.

    Shape (6, height, 3): left edge strip, left third, centre third, right
    third, right edge strip, whole frame. A profile answers "what colour is
    the scene at this elevation on this side", which is all a surround can
    honestly claim. The edge strips (the outer 4%) make the surround start
    out exactly matching the photo's border.
    """
    height, width = rgb.shape[:2]
    edge = max(1, int(round(width * 0.04)))
    thirds = np.linspace(0, width, 4).astype(int)
    strips = [
        rgb[:, :edge],
        rgb[:, thirds[0] : thirds[1]],
        rgb[:, thirds[1] : thirds[2]],
        rgb[:, thirds[2] : thirds[3]],
        rgb[:, width - edge :],
        rgb,
    ]
    profiles = np.stack([np.median(strip, axis=1) for strip in strips]).astype(np.float32)
    sigma = max(1.0, height * sigma_fraction)
    return np.stack(
        [cv2.GaussianBlur(p[:, None, :], (0, 0), sigmaX=0.1, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)[:, 0, :]
         for p in profiles]
    )


def _angle_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """(azimuth, elevation) at the pixel centres of an equirectangular image."""
    azimuth = (np.arange(width, dtype=np.float32) + 0.5) / width * 2 * np.pi - np.pi
    elevation = np.pi / 2 - (np.arange(height, dtype=np.float32) + 0.5) / height * np.pi
    return np.meshgrid(azimuth, elevation)


def _ambient(
    rgb: np.ndarray, a: np.ndarray, e: np.ndarray, fy: float, half_h: float, half_v: float
) -> np.ndarray:
    """Profile-based surround colour for every (azimuth, elevation)."""
    height = rgb.shape[0]
    cy = height / 2.0
    sharp_profiles = _row_profiles(rgb, 1 / 30)
    soft_profiles = _row_profiles(rgb, 1 / 6)
    row_index = np.clip(cy - fy * np.tan(np.clip(e, -1.5, 1.5)), 0, height - 1)
    lower = np.floor(row_index).astype(int)
    upper = np.minimum(lower + 1, height - 1)
    frac = (row_index - lower)[..., None]

    def sample(profiles):
        return profiles[:, lower] * (1 - frac) + profiles[:, upper] * frac

    l_edge, left, centre, right, r_edge, whole = sample(sharp_profiles)
    _, left_soft, _, right_soft, _, whole_soft = sample(soft_profiles)

    unwrapped = np.where(a < -half_h, a + 2 * np.pi, a)
    inside = unwrapped <= half_h
    back_span = max(2 * np.pi - 2 * half_h, 1e-6)
    beyond = np.clip(unwrapped - half_h, 0.0, back_span)
    across = np.clip((a + half_h) / (2 * half_h), 0.0, 1.0)[..., None]
    ahead = np.where(
        across < 0.5,
        left + (centre - left) * (across / 0.5),
        centre + (right - centre) * ((across - 0.5) / 0.5),
    )

    # Going around: the edge strip, then that side's third, then the whole
    # frame, blurred vertically more the farther you turn so a feature at the
    # border (a cloud, a lamp) diffuses instead of running round as a stripe.
    def around(edge_profile, side, side_soft, distance):
        d = distance[..., None]
        out = edge_profile + (side - edge_profile) * _smoothstep(d / np.radians(12.0))
        out = out + (side_soft - out) * _smoothstep(d / np.radians(45.0))
        return out + (whole_soft - out) * _smoothstep((d - np.radians(20.0)) / np.radians(70.0))

    behind_right = around(r_edge, right, right_soft, beyond)
    behind_left = around(l_edge, left, left_soft, back_span - beyond)
    t = (beyond / back_span)[..., None]
    behind = behind_right * (1 - t) + behind_left * t
    ambient = np.where(inside[..., None], ahead, behind)

    # Above and below the photo, side-to-side structure would converge into
    # beams at the poles: hand over to the whole-frame profile there.
    vertical_excess = np.maximum(np.abs(e) - half_v, 0.0)
    to_whole = _smoothstep(vertical_excess / np.radians(15.0))[..., None]
    return ambient + (whole_soft - ambient) * to_whole


def environment_map(
    rgb: np.ndarray,
    camera: "Intrinsics",
    size: tuple[int, int],
    zenith: np.ndarray,
    ground: np.ndarray,
) -> np.ndarray:
    """A 360° equirectangular surround synthesised from the photo.

    * Ahead (inside the photo's field of view) it is the photo itself, softly
      blurred. It is mostly hidden behind the mesh and shows only through
      gaps and at the edges, where it must match.
    * Everywhere else it is built from *vertical colour profiles*: the
      per-row median colour of the photo's border strips and thirds, carried
      around the viewer, blurred more with distance and blended into the
      whole-frame profile behind them. There is no copied image structure,
      so turning around never shows mirrored "Rorschach" shapes (an earlier
      design mirrored the photo outward, and symmetry is the first thing the
      eye catches).
    * Above and below the photo it converges fully to the zenith and ground
      tones, so looking straight up or down shows a smooth pole, not the
      photo's top row stretched into rays.

    The ambient part is smooth by construction, so it is computed at a
    quarter of the resolution and upsampled.
    """
    height, width = rgb.shape[:2]
    env_width, env_height = size
    fx, fy = camera.fx, camera.fy
    cx, cy = width / 2.0, height / 2.0
    cylindrical = camera.projection == "cylindrical"
    half_h = min(np.pi, cx / fx) if cylindrical else float(np.arctan(cx / fx))
    half_v = float(np.arctan(cy / fy))

    a, e = _angle_grid(env_width, env_height)
    tan_e = np.tan(np.clip(e, -1.5, 1.5))

    # -- ahead: the photo, softly blurred ------------------------------------
    clipped = np.clip(a, -half_h, half_h)
    if cylindrical:
        columns = np.clip(cx + fx * clipped, 0, width - 1)
        rows_photo = cy - fy * tan_e
    else:
        columns = cx + fx * np.tan(clipped)
        rows_photo = cy - fy * tan_e / np.cos(clipped)
    soft = cv2.GaussianBlur(rgb, (0, 0), max(1.0, width / 200.0))
    photo = cv2.remap(
        soft, columns.astype(np.float32), np.clip(rows_photo, 0, height - 1).astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)

    # -- elsewhere: profiles, at quarter resolution ---------------------------
    coarse_w, coarse_h = max(64, env_width // 4), max(32, env_height // 4)
    ca, ce = _angle_grid(coarse_w, coarse_h)
    ambient = cv2.resize(
        _ambient(rgb, ca, ce, fy, half_h, half_v).astype(np.float32),
        (env_width, env_height),
        interpolation=cv2.INTER_LINEAR,
    )

    # -- blend: the photo where it exists, profiles beyond it -----------------
    unwrapped = np.where(a < -half_h, a + 2 * np.pi, a)
    inside = unwrapped <= half_h
    back_span = max(2 * np.pi - 2 * half_h, 1e-6)
    beyond = np.clip(unwrapped - half_h, 0.0, back_span)
    excess = np.maximum(
        np.where(inside, 0.0, np.minimum(beyond, back_span - beyond)),
        np.maximum(np.abs(e) - half_v, 0.0),
    )
    # A short hand-over: just outside the frame the edge-strip profile
    # already matches the border, so the photo is not stretched outward.
    w = _smoothstep(excess / np.radians(4.0))[..., None]
    env = photo * (1.0 - w) + ambient * w

    # -- poles: converge fully to the sky / ground tones ---------------------
    span = max(1e-3, np.pi / 2 - half_v)
    up_t = _smoothstep((e - half_v) / span)[..., None]
    down_t = _smoothstep((-e - half_v) / span)[..., None]
    env = env + (zenith * 255.0 - env) * up_t
    env = env + (ground * 255.0 - env) * down_t
    env = _wrap_blur(env.astype(np.float32), max(1.0, env_width / 256.0))
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
