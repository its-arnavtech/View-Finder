"""Tests for screen detection.

The gates exist to make automatic cropping safe, so the tests that matter most
are the negative ones: an ordinary photograph must come back untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from viewfinder import screen_crop
from viewfinder.screen_crop import ScreenCropResult, detect_and_crop


def _noise_photo(width: int = 900, height: int = 700, seed: int = 0) -> Image.Image:
    """A textured image with no rectangle in it, standing in for a real photo."""
    rng = np.random.default_rng(seed)
    # Smooth gradients plus noise: busy enough to generate contours, but with
    # no straight continuous boundary for a quad to latch onto.
    yy, xx = np.mgrid[0:height, 0:width]
    base = (np.sin(xx / 40.0) * 40 + np.cos(yy / 30.0) * 40 + 128).astype(np.float64)
    noisy = np.clip(base[..., None] + rng.normal(0, 18, (height, width, 3)), 0, 255)
    return Image.fromarray(noisy.astype(np.uint8))


def _screen_photo(
    width: int = 1200,
    height: int = 900,
    inset: int = 120,
    bezel: int = 18,
) -> Image.Image:
    """A bright rectangular 'display' on a plain surround, with a dark bezel."""
    photo = Image.new("RGB", (width, height), (225, 225, 222))
    draw = ImageDraw.Draw(photo)
    draw.rectangle([inset, inset, width - inset, height - inset], fill=(12, 12, 14))
    draw.rectangle(
        [inset + bezel, inset + bezel, width - inset - bezel, height - inset - bezel],
        fill=(40, 90, 150),
    )
    return photo


# --- negative cases: must not crop ---------------------------------------


def test_ordinary_photo_is_left_alone():
    image = _noise_photo()
    result = detect_and_crop(image)
    assert result.cropped is False
    assert result.image is image


def test_disabled_returns_immediately():
    image = _noise_photo()
    result = detect_and_crop(image, enabled=False)
    assert result.cropped is False
    assert result.reason == "disabled"
    assert result.image is image


def test_tiny_image_is_skipped():
    result = detect_and_crop(Image.new("RGB", (80, 60), "white"))
    assert result.cropped is False
    assert result.reason == "image_too_small"


def test_a_rectangle_filling_the_frame_is_not_cropped():
    """Nothing to gain, and shaving a real photo's edges is a real cost."""
    photo = _screen_photo(inset=6, bezel=2)
    result = detect_and_crop(photo)
    assert result.cropped is False


def test_a_small_rectangle_is_not_cropped():
    photo = Image.new("RGB", (1200, 900), (225, 225, 222))
    ImageDraw.Draw(photo).rectangle([500, 400, 640, 500], fill=(20, 20, 20))
    result = detect_and_crop(photo)
    assert result.cropped is False


# --- positive case -------------------------------------------------------


def test_a_screen_photo_is_cropped_to_the_display():
    photo = _screen_photo()
    result = detect_and_crop(photo)
    assert result.cropped is True
    assert result.reason == "ok"
    # The crop must be materially smaller than the original but not degenerate.
    assert result.image.size[0] < photo.size[0]
    assert result.image.size[0] > photo.size[0] * 0.4
    assert screen_crop.MIN_AREA_FRACTION < result.area_fraction < screen_crop.MAX_AREA_FRACTION


def test_result_serializes_for_the_json_output():
    payload = detect_and_crop(_screen_photo()).as_dict()
    assert payload["cropped"] is True
    assert payload["reason"] == "ok"
    assert len(payload["quad"]) == 4
    assert all(len(corner) == 2 for corner in payload["quad"])


# --- gate behaviour ------------------------------------------------------


def test_edge_support_separates_a_real_boundary_from_a_coincidence():
    """The gate that makes auto mode safe; assert it actually discriminates."""
    photo = _screen_photo()
    gray = np.array(photo.convert("L"))
    edges = screen_crop._edge_map(gray)

    real = screen_crop._order_corners(
        np.array([[120, 120], [1080, 120], [1080, 780], [120, 780]], dtype=np.float32)
    )
    # Same size and shape, but sitting on blank surround with no boundary.
    coincidental = screen_crop._order_corners(
        np.array([[10, 10], [700, 10], [700, 500], [10, 500]], dtype=np.float32)
    )
    assert screen_crop._edge_support(edges, real) > screen_crop.MIN_EDGE_SUPPORT
    assert screen_crop._edge_support(edges, coincidental) < screen_crop.MIN_EDGE_SUPPORT


def test_corner_angles_of_a_square_are_right_angles():
    corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    assert all(abs(angle - 90.0) < 1e-3 for angle in screen_crop._corner_angles(corners))


def test_corner_ordering_is_clockwise_from_top_left():
    scrambled = np.array([[100, 100], [0, 100], [100, 0], [0, 0]], dtype=np.float32)
    ordered = screen_crop._order_corners(scrambled)
    assert ordered[0].tolist() == [0, 0]
    assert ordered[1].tolist() == [100, 0]
    assert ordered[2].tolist() == [100, 100]
    assert ordered[3].tolist() == [0, 100]


@pytest.mark.parametrize("seed", range(6))
def test_no_false_positives_across_varied_textures(seed):
    """A crop here would silently discard most of a real photograph."""
    assert detect_and_crop(_noise_photo(seed=seed)).cropped is False
