import io
import math

import numpy as np
import pytest
from PIL import Image

from viewfinder.world.camera import DEFAULT_FOCAL_35MM, intrinsics_for
from viewfinder.world.ingest import ImageInputError, load_source


def _jpeg(image: Image.Image, exif: Image.Exif | None = None) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif if exif is not None else Image.Exif())
    return buffer.getvalue()


def test_exif_orientation_is_applied_before_use():
    # Stored landscape with "rotate 90 CW" -> must come out portrait.
    image = Image.new("RGB", (400, 200), (10, 200, 10))
    image.paste((250, 0, 0), (0, 0, 40, 200))  # red stripe on the stored left edge
    exif = Image.Exif()
    exif[0x0112] = 6
    source = load_source(_jpeg(image, exif), working_long_side=4000)
    assert source.rgb.shape[:2] == (400, 200)
    # Orientation 6: the stored left edge becomes the top.
    assert source.rgb[5, 100, 0] > 200 and source.rgb[-5, 100, 0] < 100


def test_focal_length_is_read_and_resizing_keeps_aspect():
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 50
    source = load_source(_jpeg(Image.new("RGB", (3000, 1500)), exif), working_long_side=1024)
    assert source.focal_35mm == 50.0
    # Area budget of a 1024x768 frame, aspect preserved.
    height, width = source.rgb.shape[:2]
    assert width / height == pytest.approx(2.0, abs=0.01)
    assert width * height == pytest.approx(1024 * 768, rel=0.01)
    assert (source.source_width, source.source_height) == (3000, 1500)


def test_four_by_three_photos_keep_the_classic_working_size():
    source = load_source(_jpeg(Image.new("RGB", (4032, 3024))), working_long_side=1024)
    assert source.rgb.shape == (768, 1024, 3)


def test_panoramas_keep_a_usable_height():
    source = load_source(_jpeg(Image.new("RGB", (8000, 1000))), working_long_side=1024)
    height, width = source.rgb.shape[:2]
    assert width == 2048 and height == 256  # long side capped at 2x, height not a sliver


def test_missing_or_corrupt_focal_is_none():
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 0  # EXIF "unknown"
    assert load_source(_jpeg(Image.new("RGB", (100, 80)), exif), 1024).focal_35mm is None


def test_transparency_is_composited_on_white():
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    source = load_source(buffer.getvalue(), 1024)
    assert source.rgb.mean() > 250


@pytest.mark.parametrize("payload", [b"not an image at all", b"\xff\xd8\xff\xe0 truncated"])
def test_non_images_raise_input_error(payload):
    with pytest.raises(ImageInputError):
        load_source(payload, 1024)


def test_tiny_images_and_missing_files_are_rejected(tmp_path):
    with pytest.raises(ImageInputError, match="at least"):
        load_source(_jpeg(Image.new("RGB", (8, 8))), 1024)
    with pytest.raises(ImageInputError, match="not found"):
        load_source(tmp_path / "missing.jpg", 1024)


def test_intrinsics_default_and_exif_fov():
    default = intrinsics_for(4000, 3000)
    expected = math.degrees(2 * math.atan(36 / (2 * DEFAULT_FOCAL_35MM)))
    assert default.hfov_deg == pytest.approx(expected, rel=1e-6)
    assert default.focal_source == "default"
    tele = intrinsics_for(4000, 3000, focal_35mm=100)
    assert tele.hfov_deg < default.hfov_deg and tele.focal_source == "exif"
    # Portrait: the 36 mm frame side maps to the long (vertical) side.
    portrait = intrinsics_for(3000, 4000)
    assert portrait.vfov_deg == pytest.approx(expected, rel=1e-6)


def test_wide_images_are_cylindrical_panoramas():
    cinema = intrinsics_for(2390, 1000)
    assert cinema.projection == "pinhole"
    pano = intrinsics_for(3000, 1000, focal_35mm=26)  # EXIF focal is ignored for panoramas
    assert pano.projection == "cylindrical" and pano.focal_source == "panorama"
    assert pano.vfov_deg == pytest.approx(60.0, abs=0.1)
    assert 150 < pano.hfov_deg < 360 and not pano.wraps
    full = intrinsics_for(8000, 1000)
    assert full.hfov_deg == pytest.approx(360.0, abs=0.01) and full.wraps
    assert full.max_pad_x() == 0  # nothing to extend: the edges meet


def test_cylindrical_rays_are_linear_in_azimuth():
    pano = intrinsics_for(4000, 1000)
    centre = pano.rays(np.array([2000.0]), np.array([500.0]))[0]
    np.testing.assert_allclose(centre, [0, 0, -1], atol=1e-9)
    # A quarter of the horizontal span to the right is a quarter of hfov in azimuth.
    ray = pano.rays(np.array([3000.0]), np.array([500.0]))[0]
    assert math.degrees(math.atan2(ray[0], -ray[2])) == pytest.approx(pano.hfov_deg / 4, rel=1e-6)
    assert np.hypot(ray[0], ray[2]) == pytest.approx(1.0)  # unit cylinder


def test_rays_follow_gltf_convention():
    camera = intrinsics_for(200, 100)
    center = camera.rays(np.array([100.0]), np.array([50.0]))[0]
    np.testing.assert_allclose(center, [0, 0, -1])
    top_left = camera.rays(np.array([0.0]), np.array([0.0]))[0]
    assert top_left[0] < 0 and top_left[1] > 0  # left is -X, up is +Y
    padded = camera.padded(10, 5)
    np.testing.assert_allclose(padded.rays(np.array([110.0]), np.array([55.0]))[0], center)
