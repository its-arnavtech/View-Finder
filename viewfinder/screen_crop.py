"""Detect a screen (or framed picture) inside a photo and crop to just its content.

Motivation
----------
A photo *of a display* is not a photo of a place. Handed one, the pipeline is
asked "where is this laptop sitting in a room", not "where is the scene on the
screen", and it answers accordingly: the room, the bezel, the desk, and the
operating system's clock and news tiles all reach the model, while the scene
itself survives CLIP's ``Resize(224) + CenterCrop(224)`` at roughly a hundred
pixels of vertical resolution.

Cropping to the display fixes the *input*. It does not guarantee the answer --
the scene still has to be something the model or the gallery knows.

Why this is opt-out rather than always-on
-----------------------------------------
Cropping an ordinary photograph would be actively harmful: it would throw away
most of the frame on the strength of a false detection. So detection is
deliberately conservative and every gate below must pass. When any of them
fails the original image is returned untouched and the reason is recorded, so a
no-op is always explainable rather than silent.

The gates
---------
``area``
    The quad must cover between ``MIN_AREA_FRACTION`` and ``MAX_AREA_FRACTION``
    of the frame. Too small is a window, a sign, or noise. Too large means the
    photo essentially *is* the quad and there is nothing worth cropping away.
``aspect``
    Displays and framed pictures are rectangles in a bounded range of aspect
    ratios. This rejects doors, table edges and long thin highlights.
``contrast``
    A screen looks materially different from what surrounds it. Note this is a
    two-sided test on |inside - outside|, not "inside is brighter": a lit screen
    in a dim room and a dark wallpaper against a bright wall are both real, and
    an earlier one-sided brightness test would have rejected the second.
``rectangularity``
    The contour must actually fill its own quad. A four-point approximation of
    a blobby shape passes the other gates surprisingly often; this does not.
``edge support``
    The strongest gate, and the one that makes automatic use safe. A real bezel
    produces a detected edge along essentially its whole perimeter; a quad
    hallucinated from scenery has corners that happen to line up but sides that
    are only partly supported by any real gradient.
``angles`` / ``opposite sides``
    A physical rectangle under mild perspective keeps near-right corners and
    near-equal opposite sides. Natural quads drift well outside both.

Those last three were added after measurement, not designed in: the first four
gates alone cropped 18 of 400 real photographs (4.5%), which is far too high for
something that runs without being asked. With the full set, measured on 900
photographs from the Im2GPS benchmark set, the rate is **4 in 900 (0.44%)** while
both a real and a synthetic screen photo still crop.

Re-run that measurement if you touch any threshold here. It is easy to recover a
real screen by loosening a gate and silently give back the false-positive rate:
raising the edge-support tolerance to make a real photograph detect at all took
the rate from 0.44% to 2.5% until the aspect, area and angle gates were tightened
to compensate.

What this does *not* fix
------------------------
Cropping fixes the input, not the model. On the photograph this was built for --
a Windows lock screen showing the Penang Second Bridge -- the crop succeeds and
the prediction does not move at all, because the overlaid clock and date
dominate what is left. ``--remove-text`` is what shifts that one, and even then
the answer lands in East Asia rather than Malaysia, which no preprocessing can
repair: the gallery holds zero photographs within 972 km of Penang.

Nested quads
------------
A laptop photographed head-on yields two concentric quads: the outer lid and
the display inside the bezel. Both pass the gates, and the outer one is larger,
so a plain largest-quad rule returns the lid and keeps the bezel. Detection
therefore descends into a contained quad when one covers enough of its parent
to be its screen rather than an element drawn on it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# --- Gates ---------------------------------------------------------------
# A quad smaller than this is not the subject of the photograph. Someone
# photographing a screen for its content fills the frame with it; below this the
# rectangle is incidental (a TV across a room), and cropping to it would answer
# a question nobody asked.
MIN_AREA_FRACTION = 0.20
# Above this the quad is essentially the whole frame; cropping gains nothing
# and risks shaving a real photo's edges.
MAX_AREA_FRACTION = 0.94
# Displays, monitors, TVs and framed pictures live in this range: 4:3 is 1.33,
# 16:10 is 1.60, 16:9 is 1.78, and the widest mass-market ultrawide is 21:9 at
# 2.37. Nothing rectangular in a photograph is a display at 2.7 or 3.3, so the
# ceiling sits just above the real one. Phones held in portrait are excluded
# deliberately: they are usually photographed at an angle severe enough that
# the warp does more harm than good.
MIN_ASPECT = 1.05
MAX_ASPECT = 2.45
# Minimum relative difference in mean luminance between the quad's interior and
# the frame outside it. Two-sided: either can be the brighter one.
MIN_CONTRAST_RATIO = 0.10
# The contour's own area must be at least this fraction of its bounding quad.
MIN_RECTANGULARITY = 0.85
# Fraction of the quad's perimeter that must coincide with a detected edge.
# A bezel scores near 1.0; scenery quads rarely clear 0.8.
MIN_EDGE_SUPPORT = 0.92
# How far from the fitted quad an edge may sit and still count as supporting it,
# as a fraction of the working image's long side. This is a tolerance, not a
# threshold on evidence: a real screen boundary is never perfectly straight
# (lens distortion, and the quad is a hull approximation), so it drifts from the
# fitted line by roughly a percent of the frame. Tuned at 5 px on a synthetic
# rectangle it scored a real photograph's genuine screen edge at 0.22 and
# rejected it; the same edge scores 0.95 at this tolerance.
EDGE_SUPPORT_TOLERANCE_RATIO = 0.020
MIN_EDGE_SUPPORT_TOLERANCE_PX = 5
# Corner angles must stay this close to square, in degrees. Someone
# photographing a screen to read it stands in front of it; 10 degrees is
# already a noticeably oblique shot.
MAX_ANGLE_DEVIATION = 10.0
# Opposite sides may differ by at most this ratio (perspective foreshortening).
MAX_OPPOSITE_SIDE_RATIO = 1.22
# A contained quad this much of its parent's area is taken to be the display
# inside a bezel rather than a tile drawn on the display.
NESTED_AREA_RATIO = 0.55

# Detection runs on a downscaled copy; the warp uses the full-resolution source.
WORKING_LONG_SIDE = 1100
# Canny hysteresis, applied to a median-blurred grayscale image.
CANNY_LOW, CANNY_HIGH = 50, 160
# Polygon-approximation tolerances, as a fraction of the hull perimeter, tried
# in order until one yields four convex points. Tight first, so a crisp
# rectangle is not over-simplified before a looser value is reached.
APPROX_EPSILON_RATIOS = (0.010, 0.020, 0.030, 0.045, 0.060, 0.080)


@dataclass
class ScreenCropResult:
    """Outcome of one screen-detection pass."""

    image: Image.Image = field(repr=False)
    """The cropped image, or the original object when nothing was cropped."""

    cropped: bool = False
    reason: str = "not_attempted"
    """Why the decision went the way it did. Always set, including on success."""

    area_fraction: float = 0.0
    aspect_ratio: float = 0.0
    contrast_ratio: float = 0.0
    quad: list[list[int]] | None = None
    """The detected corners in source coordinates, clockwise from top-left."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "cropped": self.cropped,
            "reason": self.reason,
            "area_fraction": round(self.area_fraction, 4),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "contrast_ratio": round(self.contrast_ratio, 4),
            "quad": self.quad,
        }


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points clockwise from the top-left.

    Uses coordinate sums and differences, which is orientation-stable for the
    moderate perspective a hand-held photo produces.
    """
    points = points.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = points.sum(axis=1)
    ordered[0] = points[np.argmin(total)]   # top-left  : smallest x+y
    ordered[2] = points[np.argmax(total)]   # bottom-right
    diff = np.diff(points, axis=1).ravel()
    ordered[1] = points[np.argmin(diff)]    # top-right : smallest y-x
    ordered[3] = points[np.argmax(diff)]    # bottom-left
    return ordered


def _edge_map(gray: np.ndarray) -> np.ndarray:
    """Union of two edge detectors.

    Neither alone is reliable here: Canny finds a bezel against a plain wall,
    while adaptive thresholding survives the window glare that flattens local
    gradients on a glossy screen.
    """
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 6
    )
    return cv2.bitwise_or(edges, cv2.Canny(adaptive, CANNY_LOW, CANNY_HIGH))


def _approx_quad(contour: np.ndarray) -> np.ndarray | None:
    """Reduce a contour to a convex four-point polygon, or None.

    Two departures from the textbook single ``approxPolyDP`` call, both needed
    by real photographs rather than by synthetic ones:

    * The **convex hull** is approximated instead of the raw contour. A screen
      boundary interrupted by a reflection, a webcam notch or the tile strip
      traces as a non-convex 8-9 vertex shape that no epsilon reduces to a
      clean quad; its hull is exactly the rectangle wanted.
    * A **sweep of epsilon values** rather than one fixed ratio. The right
      tolerance depends on how noisy the boundary is, and a single value that
      works for a crisp synthetic bezel misses a hand-held photo.

    Loosening candidate *generation* is safe because it decides nothing: every
    candidate still has to clear the gates in :func:`_evaluate`.
    """
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0:
        return None
    for ratio in APPROX_EPSILON_RATIOS:
        approx = cv2.approxPolyDP(hull, ratio * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad_area = cv2.contourArea(approx)
            if quad_area <= 0:
                return None
            # Rectangularity is measured against the hull, not the raw contour:
            # the question is whether the enclosing shape is a rectangle.
            if cv2.contourArea(hull) / quad_area < MIN_RECTANGULARITY:
                return None
            return approx
    return None


def _candidate_quads(edges: np.ndarray) -> list[np.ndarray]:
    """Return convex, genuinely rectangle-shaped four-point contours, largest first."""
    # Close small gaps so a bezel broken by a reflection still forms one contour.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # A screen photographed close-up runs off the side of the frame, so its
    # boundary is an open curve and never becomes a quad. Drawing the frame's
    # own border into the edge map lets such a shape close against it. The
    # border rectangle itself becomes a full-frame contour, which the
    # MAX_AREA_FRACTION gate discards.
    bordered = closed.copy()
    cv2.rectangle(bordered, (0, 0), (bordered.shape[1] - 1, bordered.shape[0] - 1), 255, 1)

    contours, _ = cv2.findContours(bordered, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads: list[np.ndarray] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
        approx = _approx_quad(contour)
        if approx is not None:
            quads.append(approx)
    return quads


def _contrast_ratio(gray: np.ndarray, quad: np.ndarray) -> float:
    """Relative difference in mean luminance inside vs outside the quad."""
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [quad.reshape(4, 2).astype(np.int32)], 255)
    inside_pixels = gray[mask == 255]
    outside_pixels = gray[mask == 0]
    if inside_pixels.size == 0 or outside_pixels.size == 0:
        return 0.0
    inside = float(inside_pixels.mean())
    outside = float(outside_pixels.mean())
    denominator = max(inside, outside, 1.0)
    return abs(inside - outside) / denominator


def _edge_support(edges: np.ndarray, corners: np.ndarray) -> float:
    """Fraction of the quad's outline that lies on a detected edge.

    This is what separates a bezel from four scenery features that happen to
    form a rectangle: the bezel is an actual continuous boundary in the image.
    """
    outline = np.zeros(edges.shape, dtype=np.uint8)
    cv2.polylines(outline, [corners.astype(np.int32)], True, 255, thickness=1)
    outline_pixels = int(np.count_nonzero(outline))
    if outline_pixels == 0:
        return 0.0
    # Dilate the edges, not the outline, so a slightly off polygon approximation
    # is tolerated without inflating the denominator.
    tolerance = max(
        MIN_EDGE_SUPPORT_TOLERANCE_PX,
        int(round(EDGE_SUPPORT_TOLERANCE_RATIO * max(edges.shape))),
    )
    kernel_size = tolerance * 2 + 1
    near_edge = cv2.dilate(edges, np.ones((kernel_size, kernel_size), np.uint8))
    hit = int(np.count_nonzero(cv2.bitwise_and(outline, near_edge)))
    return hit / outline_pixels


def _corner_angles(corners: np.ndarray) -> list[float]:
    """Interior angle at each corner, in degrees."""
    angles: list[float] = []
    for i in range(4):
        previous = corners[(i - 1) % 4] - corners[i]
        following = corners[(i + 1) % 4] - corners[i]
        norms = np.linalg.norm(previous) * np.linalg.norm(following)
        cosine = float(np.dot(previous, following) / norms) if norms > 0 else 0.0
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return angles


def _evaluate(
    quad: np.ndarray, gray: np.ndarray, edges: np.ndarray
) -> tuple[bool, str, dict[str, float]]:
    """Apply every gate to one candidate quad."""
    height, width = gray.shape[:2]
    frame_area = float(height * width)
    corners = _order_corners(quad)

    area_fraction = cv2.contourArea(corners) / frame_area
    top = float(np.linalg.norm(corners[1] - corners[0]))
    bottom = float(np.linalg.norm(corners[2] - corners[3]))
    left = float(np.linalg.norm(corners[3] - corners[0]))
    right = float(np.linalg.norm(corners[2] - corners[1]))
    quad_width = max(top, bottom)
    quad_height = max(left, right)
    aspect = quad_width / quad_height if quad_height > 0 else 0.0
    contrast = _contrast_ratio(gray, corners)
    support = _edge_support(edges, corners)
    angles = _corner_angles(corners)
    worst_angle = max(abs(angle - 90.0) for angle in angles)
    horizontal_ratio = max(top, bottom) / max(min(top, bottom), 1e-6)
    vertical_ratio = max(left, right) / max(min(left, right), 1e-6)

    metrics = {
        "area_fraction": area_fraction,
        "aspect_ratio": aspect,
        "contrast_ratio": contrast,
        "edge_support": support,
        "worst_angle_deviation": worst_angle,
    }

    if area_fraction < MIN_AREA_FRACTION:
        return False, f"quad_too_small({area_fraction:.3f})", metrics
    if area_fraction > MAX_AREA_FRACTION:
        return False, f"quad_fills_frame({area_fraction:.3f})", metrics
    if not MIN_ASPECT <= aspect <= MAX_ASPECT:
        return False, f"aspect_out_of_range({aspect:.2f})", metrics
    if contrast < MIN_CONTRAST_RATIO:
        return False, f"insufficient_contrast({contrast:.3f})", metrics
    if worst_angle > MAX_ANGLE_DEVIATION:
        return False, f"corners_not_square({worst_angle:.0f}deg)", metrics
    if max(horizontal_ratio, vertical_ratio) > MAX_OPPOSITE_SIDE_RATIO:
        return False, f"opposite_sides_unequal({max(horizontal_ratio, vertical_ratio):.2f})", metrics
    if support < MIN_EDGE_SUPPORT:
        return False, f"weak_edge_support({support:.2f})", metrics
    return True, "ok", metrics


def _contains(outer: np.ndarray, inner: np.ndarray) -> bool:
    """True when every corner of ``inner`` lies inside ``outer``."""
    polygon = outer.reshape(4, 2).astype(np.float32)
    return all(
        cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0
        for x, y in inner.reshape(4, 2)
    )


def _select(
    quads: list[np.ndarray], gray: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray | None, str, dict]:
    """Pick the innermost passing quad, descending through nested rectangles."""
    passing: list[tuple[np.ndarray, dict]] = []
    first_failure = "no_quadrilateral_found"
    for quad in quads:
        ok, reason, metrics = _evaluate(quad, gray, edges)
        if ok:
            passing.append((quad, metrics))
        elif first_failure == "no_quadrilateral_found":
            first_failure = reason

    if not passing:
        return None, first_failure, {}

    passing.sort(key=lambda item: item[1]["area_fraction"], reverse=True)
    chosen, metrics = passing[0]
    # Descend: a quad inside the current one covering most of it is the display
    # within a bezel, not a widget drawn on the display.
    changed = True
    while changed:
        changed = False
        for quad, candidate_metrics in passing:
            if quad is chosen:
                continue
            ratio = candidate_metrics["area_fraction"] / max(metrics["area_fraction"], 1e-9)
            if NESTED_AREA_RATIO <= ratio < 0.995 and _contains(chosen, quad):
                chosen, metrics = quad, candidate_metrics
                changed = True
                break
    return chosen, "ok", metrics


def detect_and_crop(image: Image.Image, enabled: bool = True) -> ScreenCropResult:
    """Crop to a detected screen, or return the image unchanged.

    Args:
        image: RGB PIL image.
        enabled: When False, returns immediately (keeps callers branch-free).

    Returns:
        A :class:`ScreenCropResult`. ``cropped`` is False whenever any gate
        failed, and ``reason`` says which one.
    """
    if not enabled:
        return ScreenCropResult(image=image, reason="disabled")

    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if min(height, width) < 200:
        return ScreenCropResult(image=image, reason="image_too_small")

    scale = min(1.0, WORKING_LONG_SIDE / float(max(height, width)))
    working = (
        cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else rgb
    )
    gray = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)

    try:
        edges = _edge_map(gray)
        quads = _candidate_quads(edges)
    except cv2.error as exc:  # pragma: no cover - malformed input
        logger.debug("screen detection failed: %s", exc)
        return ScreenCropResult(image=image, reason=f"detection_error:{exc}")

    if not quads:
        return ScreenCropResult(image=image, reason="no_quadrilateral_found")

    chosen, reason, metrics = _select(quads, gray, edges)
    if chosen is None:
        return ScreenCropResult(image=image, reason=reason)

    # Map the quad back to full resolution and warp from the original pixels, so
    # the crop keeps every bit of detail the source had.
    corners = _order_corners(chosen) / scale
    top = float(np.linalg.norm(corners[1] - corners[0]))
    bottom = float(np.linalg.norm(corners[2] - corners[3]))
    left = float(np.linalg.norm(corners[3] - corners[0]))
    right = float(np.linalg.norm(corners[2] - corners[1]))
    out_w = int(round(max(top, bottom)))
    out_h = int(round(max(left, right)))
    if out_w < 64 or out_h < 64:
        return ScreenCropResult(image=image, reason="cropped_region_too_small")

    destination = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    warped = cv2.warpPerspective(rgb, transform, (out_w, out_h), flags=cv2.INTER_CUBIC)

    logger.info(
        "screen crop: cropped to a %dx%d region covering %.0f%% of the frame "
        "(aspect %.2f, contrast %.2f)",
        out_w, out_h, metrics["area_fraction"] * 100,
        metrics["aspect_ratio"], metrics["contrast_ratio"],
    )
    return ScreenCropResult(
        image=Image.fromarray(warped),
        cropped=True,
        reason="ok",
        area_fraction=metrics["area_fraction"],
        aspect_ratio=metrics["aspect_ratio"],
        contrast_ratio=metrics["contrast_ratio"],
        quad=[[int(x), int(y)] for x, y in corners],
    )
