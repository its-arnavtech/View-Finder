import numpy as np
import pytest

from viewfinder.world import build_world, inpaint
from viewfinder.world.cli import run as world_cli
from viewfinder.world.environment import extend_frame
from viewfinder.world.geometry import near_side_of_jumps
from viewfinder.world.inpaint import InpainterUnavailable, LamaInpainter
from viewfinder.world.layers import background_layer, occluder_pixels

from world_fixtures import SceneEstimator, scene_jpeg

MAGENTA = (255, 0, 255)


class PaintInpainter:
    """Fills every masked pixel with one colour and records each mask."""

    name = "paint"

    def __init__(self, color=MAGENTA):
        self.color = np.array(color, np.uint8)
        self.masks = []

    def inpaint(self, rgb, mask):
        self.masks.append(mask.copy())
        out = rgb.copy()
        out[mask] = self.color
        return out

    def describe(self):
        return {"inpainter": self.name}


def _lama_installed() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return inpaint.cached_model_path() is not None


def test_outpainted_pad_keeps_the_photo_and_fades_only_its_outer_part():
    height, width = 60, 80
    plane = np.broadcast_to(np.linspace(0.2, 0.8, height, dtype=np.float32)[:, None], (height, width)).copy()
    rgb = np.full((height, width, 3), 120, np.uint8)
    fade = np.zeros((height + 20, width + 26, 3), np.float32)  # pads: 10 rows, 13 columns
    painter = PaintInpainter()
    canvas = extend_frame(rgb, plane, np.zeros_like(plane, bool), (10, 13), fade_to=fade, inpainter=painter)
    (mask,) = painter.masks
    assert mask.sum() == mask.size - height * width and not mask[10:70, 13:93].any()  # exactly the pad
    assert (canvas.rgb[10:70, 13:93] == 120).all()  # photo untouched
    assert tuple(canvas.rgb[40, 11]) == MAGENTA  # beside the photo: outpainted texture, unfaded
    assert canvas.rgb[0, 0].max() < 30  # outer corner still fades to the (black) dome


def test_inpainted_background_hides_the_occluder_and_replaces_only_hidden_pixels():
    rgb = np.zeros((48, 96, 3), np.uint8)
    rgb[:, :40] = (0, 200, 0)  # far background, left
    rgb[:, 40:] = (200, 0, 0)  # near box, right
    pixel_disparity = np.where(np.arange(96)[None, :] < 40, 0.1, 0.9).astype(np.float32) * np.ones((48, 1))
    grid_disparity = np.ascontiguousarray(pixel_disparity[::4, ::4])
    sky = np.zeros_like(grid_disparity, bool)
    near = near_side_of_jumps(grid_disparity, 0.05)
    painter = PaintInpainter((0, 0, 255))
    layer = background_layer(rgb, pixel_disparity, grid_disparity, sky, near, 8, 0.05, inpainter=painter)
    unknown = occluder_pixels(pixel_disparity, 4, 0.05)
    (mask,) = painter.masks
    # The model sees no occluder pixel near the edge, not even those it need not replace.
    assert (mask >= unknown).all() and mask.sum() > unknown.sum()
    assert (layer.rgb[unknown] == (0, 0, 255)).all()
    assert (layer.rgb[~unknown] == rgb[~unknown]).all()


def test_build_world_uses_the_inpainter_and_records_it():
    painter = PaintInpainter()
    result = build_world(scene_jpeg(), profile="mobile", estimator=SceneEstimator(), inpainter=painter)
    assert result.manifest["inpainting"] == {"inpainter": "paint"}
    assert len(painter.masks) == 2  # frame pad, then the hidden background
    plain = build_world(scene_jpeg(), profile="mobile", estimator=SceneEstimator())
    assert plain.manifest["inpainting"] == {"inpainter": "none"}


def test_missing_lama_model_says_how_to_install_it(monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.setattr(inpaint, "cached_model_path", lambda: None)
    with pytest.raises(InpainterUnavailable, match="world setup"):
        LamaInpainter()


def test_cli_inpaint_auto_degrades_and_lama_insists(tmp_path, monkeypatch):
    def missing():
        raise InpainterUnavailable("not installed")

    monkeypatch.setattr(inpaint, "load_inpainter", missing)
    photo = tmp_path / "scene.jpg"
    photo.write_bytes(scene_jpeg())
    common = ["build", str(photo), "--depth-model", "heuristic", "--quality", "mobile"]
    assert world_cli(common) == 0
    assert world_cli(common + ["--inpaint", "lama"]) == 1


@pytest.mark.skipif(not _lama_installed(), reason="LaMa is not installed (run `view-finder world setup`)")
def test_lama_fills_a_hole_and_never_touches_known_pixels():
    rng = np.random.default_rng(0)
    rgb = np.clip(np.array([60, 140, 60]) + rng.normal(0, 6, (300, 400, 3)), 0, 255).astype(np.uint8)
    mask = np.zeros((300, 400), bool)
    mask[100:200, 150:250] = True
    rgb[mask] = (255, 0, 0)  # junk under the mask must be ignored
    out = inpaint.load_inpainter().inpaint(rgb, mask)
    assert (out[~mask] == rgb[~mask]).all()
    assert np.abs(out[mask].mean(axis=0) - (60, 140, 60)).max() < 25
