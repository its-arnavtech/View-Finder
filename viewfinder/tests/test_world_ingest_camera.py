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
    assert source.rgb.shape == (512, 1024, 3)
    assert (source.source_width, source.source_height) == (3000, 1500)


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


def test_rays_follow_gltf_convention():
    camera = intrinsics_for(200, 100)
    center = camera.rays(np.array([100.0]), np.array([50.0]))[0]
    np.testing.assert_allclose(center, [0, 0, -1])
    top_left = camera.rays(np.array([0.0]), np.array([0.0]))[0]
    assert top_left[0] < 0 and top_left[1] > 0  # left is -X, up is +Y
    padded = camera.padded(10, 5)
    np.testing.assert_allclose(padded.rays(np.array([110.0]), np.array([55.0]))[0], center)
