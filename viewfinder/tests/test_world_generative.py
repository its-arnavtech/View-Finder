import numpy as np
import pytest

from viewfinder.world import build_world, generate, surround
from viewfinder.world.camera import intrinsics_for
from viewfinder.world.cli import run as world_cli
from viewfinder.world.generate import GeneratorUnavailable, complete_panorama, match_colours, view_plan
from viewfinder.world.gltf import read_glb
from viewfinder.world.panorama import (
    View,
    direction_to_equirect,
    equirect_directions,
    paste_view,
    photo_to_equirect,
    project_to_photo,
    render_view,
    stitch_disparity,
)
from viewfinder.world.scene import GENERIC, NEGATIVE, Scene

from world_fixtures import SceneEstimator, scene_jpeg


class FlatBackend:
    """Fills every hole with one colour and records the prompts it was given."""

    name = "flat"

    def __init__(self, color=(40, 160, 90)):
        self.color = np.array(color, np.uint8)
        self.prompts = []

    def inpaint(self, image, mask, prompt, negative, seed, strength=0.99):
        self.prompts.append((prompt, negative))
        out = image.copy()
        out[mask] = self.color
        return out

    def describe(self):
        return {"generator": self.name}


@pytest.fixture
def small_canvas(monkeypatch):
    monkeypatch.setattr(surround, "PANORAMA_SIZE", (512, 256))
    monkeypatch.setattr(surround, "DEPTH_SIZE", (256, 128))
    monkeypatch.setattr(surround, "DEPTH_VIEW_SIZE", 96)


def test_directions_round_trip_through_equirect_pixels():
    directions = equirect_directions(64, 32)
    columns, rows = direction_to_equirect(directions, 64, 32)
    assert np.allclose(columns, np.arange(64)[None, :], atol=1e-3)
    assert np.allclose(rows, np.arange(32)[:, None], atol=1e-3)
    assert np.allclose(np.linalg.norm(directions, axis=-1), 1.0, atol=1e-5)


def test_views_project_their_own_rays_back_to_their_pixels():
    for view in (View(0, 0, 100, 48), View(135, 0, 100, 48), View(0, 90, 100, 48), View(-60, -90, 100, 48)):
        i, j, _ = view.project(view.directions())
        assert np.allclose(i, np.arange(48)[None, :], atol=1e-3)
        assert np.allclose(j, np.arange(48)[:, None], atol=1e-3)


def test_photo_sits_ahead_at_its_true_field_of_view():
    rgb = np.zeros((60, 80, 3), np.uint8)
    rgb[:, :40] = (255, 0, 0)  # left half red
    rgb[:, 40:] = (0, 0, 255)  # right half blue
    camera = intrinsics_for(80, 60)
    panorama, known = photo_to_equirect(rgb, camera, 720, 360)
    columns = np.nonzero(known.any(axis=0))[0]
    span_deg = (columns[-1] - columns[0] + 1) / 720 * 360
    assert span_deg == pytest.approx(camera.hfov_deg, abs=1.5)
    assert abs((columns[0] + columns[-1]) / 2 - 359.5) <= 1  # centred straight ahead
    assert (panorama[180, 350][0] > 200) and (panorama[180, 370][2] > 200)  # left stays left
    # A view at the photo's field of view reproduces it.
    view = render_view(panorama, View(0, 0, camera.hfov_deg, 80))
    assert view[30, 10, 0] > 200 and view[30, 70, 2] > 200


def test_paste_fills_only_unknown_pixels():
    panorama = np.zeros((64, 128, 3), np.uint8)
    known = np.zeros((64, 128), bool)
    known[:, :64] = True
    panorama[known] = 7
    view = View(0, 0, 120, 64)
    filled = paste_view(panorama, known, np.full((64, 64, 3), 200, np.uint8), view)
    assert filled.any() and not (filled & (np.arange(128)[None, :] < 64)).any()
    assert (panorama[:, :64] == 7).all()


def test_complete_panorama_fills_everything_and_keeps_the_photo():
    rgb = np.full((60, 80, 3), 222, np.uint8)
    camera = intrinsics_for(80, 60)
    panorama, known = photo_to_equirect(rgb, camera, 512, 256)
    photo_pixels = panorama[known].copy()
    backend = FlatBackend()
    indoor = Scene("a living room", "indoor", "lit by indoor lights", 0.9)
    result = complete_panorama(panorama, known, indoor, backend, view_size=128)
    assert result.generated.sum() == (~known).sum()
    assert (panorama[known] == photo_pixels).all()  # the photo is never repainted
    assert panorama[~known].min() > 20  # every invented pixel was filled
    prompts = [p for p, _ in backend.prompts]
    assert len(prompts) == len(view_plan(128))
    assert all("no people" in p for p in prompts)
    assert "ceiling" in prompts[-2] and "floor" in prompts[-1]  # up, then down, indoors
    assert all("people" in n and "haze" in n for _, n in backend.prompts)
    foggy = Scene("a forest", "nature", "in fog", 0.9)
    assert foggy.negative() == NEGATIVE  # a foggy photo may stay foggy


def test_generated_views_are_colour_matched_to_their_surroundings():
    reference = np.full((32, 32, 3), (100, 120, 140), np.uint8)
    generated = np.full((32, 32, 3), (160, 60, 200), np.uint8)
    generated[16:] += 10  # some texture so the gain is defined
    reference[16:] += 10
    region = np.zeros((32, 32), bool)
    region[:, :8] = True
    matched = match_colours(generated, reference, region)
    assert np.abs(matched[region].astype(int) - reference[region]).max() <= 2
    assert np.abs(matched[:, 20:].astype(int) - reference[:, 20:]).max() <= 2  # the shift applies everywhere


def test_outdoor_prompts_look_at_sky_and_ground():
    assert "sky" in GENERIC.prompt("up") and "ground" in GENERIC.prompt("down")
    assert "empty" in GENERIC.prompt() and not GENERIC.enclosed


def test_stitched_depth_recovers_the_scene_from_affine_views():
    """Each view's estimate is the true inverse depth under a different
    scale and shift; stitching must undo them using the overlaps."""
    width, height = 256, 128
    directions = equirect_directions(width, height)
    # A room-like world: distance varies smoothly with direction.
    truth = (0.5 + 0.3 * np.cos(np.arctan2(directions[..., 0], -directions[..., 2]))).astype(np.float32)
    known = np.zeros((height, width), bool)
    known[40:88, 112:144] = True  # the "photo" straight ahead
    views = view_plan(96)
    calls = []

    def estimate(rgb):
        view = views[len(calls)]
        calls.append(view)
        cosine = view.directions() @ view.basis()[0]
        inverse_depth = render_view(truth, view) / np.maximum(cosine, 1e-3)
        return (3.0 + len(calls)) * inverse_depth + 0.2 * len(calls)

    result = stitch_disparity(np.zeros((height, width, 3), np.uint8), np.where(known, truth, 0), known, views, estimate)
    assert len(calls) == len(views)
    assert np.median(np.abs(result - truth) / truth) < 0.05


def test_world_with_a_generated_surround(small_canvas, monkeypatch):
    from viewfinder.world import explore

    # The synthetic scene is too cramped for a station to clear its walls; one
    # 3 m in exercises the export.
    monkeypatch.setattr(explore, "plan_stations", lambda *a, **k: [explore.Station(np.array([0.0, 0.0, -3.0]))])
    backend = FlatBackend()
    result = build_world(
        scene_jpeg(), profile="mobile", estimator=SceneEstimator(), generator=backend, classify=lambda rgb: GENERIC
    )
    manifest = result.manifest
    assert manifest["surround"]["generator"] == "flat"
    assert manifest["surround"]["scene"]["place"] == GENERIC.place
    assert manifest["source"]["padded"] == manifest["source"]["working"]  # no frame padding
    glb = read_glb(result.glb)
    names = [mesh["name"] for mesh in glb.document["meshes"]]
    assert "surround" in names and "ground" not in names
    # One station bubble (mobile), with its own dome, both bound to its station.
    assert "walk0" in names and "walk0_dome" in names
    station = manifest["overlays"]["walk0"]["station"]
    assert manifest["overlays"]["walk0_dome"]["station"] == station and station[2] < 0  # into the photo
    assert [round(v, 2) for v in station[::2]] in manifest["navigation"]["stations"]
    # The photo's own mesh stops at the edge ring, where the surround fades in.
    camera = intrinsics_for(*manifest["source"]["working"])
    foreground = glb.accessor(glb.mesh("foreground")["primitives"][0]["attributes"]["POSITION"])
    u, v, _ = project_to_photo(camera, foreground / np.linalg.norm(foreground, axis=1, keepdims=True))
    ring = surround.ring_pixels(camera)
    assert u.min() >= ring - 2 and u.max() <= camera.width - 1 - ring + 2
    assert v.min() >= ring - 2 and v.max() <= camera.height - 1 - ring + 2
    plain = build_world(scene_jpeg(), profile="mobile", estimator=SceneEstimator())
    assert plain.manifest["surround"] == {"generator": "none"}


def test_surround_mesh_leaves_the_photo_to_the_photo_layer(small_canvas):
    camera = intrinsics_for(320, 240)
    rgb = np.full((240, 320, 3), 128, np.uint8)
    disparity = np.full((240, 320), 0.5, np.float32)
    built = surround.build_surround(rgb, camera, disparity, FlatBackend(), lambda image: np.ones(image.shape[:2], np.float32))
    layer = surround.surround_layer(built, camera, (64, 32), jump=1.0, far=40.0)
    assert layer is not None
    directions = layer.positions / np.linalg.norm(layer.positions, axis=1, keepdims=True)
    u, v, inside = project_to_photo(camera, directions)
    deep_inside = inside & (u > 40) & (u < 280) & (v > 40) & (v < 200)
    assert not deep_inside.any()


def test_generated_surround_is_cached_per_photo(small_canvas, tmp_path):
    camera = intrinsics_for(320, 240)
    rgb = np.full((240, 320, 3), 128, np.uint8)
    disparity = np.full((240, 320), 0.5, np.float32)
    estimate = lambda image: np.ones(image.shape[:2], np.float32)  # noqa: E731
    first = FlatBackend()
    built = surround.build_surround(rgb, camera, disparity, first, estimate, cache_dir=tmp_path)
    again = FlatBackend()
    cached = surround.build_surround(rgb, camera, disparity, again, estimate, cache_dir=tmp_path)
    assert first.prompts and not again.prompts  # the second build generated nothing
    assert (cached.panorama == built.panorama).all()
    other = rgb.copy()
    other[0, 0] = 0
    surround.build_surround(other, camera, disparity, again, estimate, cache_dir=tmp_path)
    assert again.prompts  # a different photo is generated afresh


def test_cli_generate_auto_degrades_and_on_insists(tmp_path, monkeypatch):
    def missing(device="auto"):
        raise GeneratorUnavailable("no GPU")

    monkeypatch.setattr(generate, "load_backend", missing)
    photo = tmp_path / "scene.jpg"
    photo.write_bytes(scene_jpeg())
    common = ["build", str(photo), "--depth-model", "heuristic", "--quality", "mobile", "--inpaint", "none"]
    assert world_cli(common) == 0
    assert world_cli(common + ["--generate", "on"]) == 1


def test_a_built_places_style_is_in_every_prompt():
    gothic = Scene("a corridor", "indoor", "lit by daylight through windows", 0.9,
                   style="with gothic stone arches and vaulted ceilings")
    for direction in ("around", "up", "down"):
        assert "gothic stone arches" in gothic.prompt(direction)
    assert gothic.as_dict()["style"] == gothic.style
    assert "  " not in GENERIC.prompt()  # no style, no gap
