import io
import re

import numpy as np
import pytest
from PIL import Image

from viewfinder.world import SCHEMA, build_world
from viewfinder.world.cli import run as world_cli
from viewfinder.world.depth import HeuristicDepthEstimator, model_cached
from viewfinder.world.export import standalone_html, write_world
from viewfinder.world.gltf import read_glb
from viewfinder.world.ingest import ImageInputError

from world_fixtures import SceneEstimator, scene_jpeg

MANIFEST_KEYS = {
    "schema", "name", "source", "profile", "profile_settings", "depth", "camera",
    "navigation", "environment", "layers", "textures", "timings_ms", "warnings", "glb_bytes",
}


@pytest.fixture(scope="module")
def mobile_world():
    stages = []
    result = build_world(
        scene_jpeg(),
        profile="mobile",
        estimator=SceneEstimator(),
        progress=lambda stage, fraction: stages.append((stage, fraction)),
    )
    return result, stages


def test_world_has_every_layer_and_a_valid_glb(mobile_world):
    result, _ = mobile_world
    glb = read_glb(result.glb)
    names = [mesh["name"] for mesh in glb.document["meshes"]]
    assert names == ["dome", "sky", "ground", "background", "foreground", "silhouettes"]
    assert glb.extras["viewfinder_world"]["schema"] == SCHEMA
    # photo, environment, silhouette mask (PNG) and background textures
    assert len(glb.document["images"]) == 4
    mimes = [image["mimeType"] for image in glb.document["images"]]
    assert mimes.count("image/png") == 1
    silhouettes = glb.mesh("silhouettes")["primitives"][0]
    assert "viewfinder_mask" in glb.document["materials"][silhouettes["material"]]["extras"]
    assert Image.open(io.BytesIO(glb.image_bytes(0))).format == "JPEG"


def test_manifest_schema_and_environment(mobile_world):
    result, _ = mobile_world
    manifest = result.manifest
    assert MANIFEST_KEYS <= set(manifest)
    assert manifest["schema"] == SCHEMA and manifest["profile"] == "mobile"
    assert manifest["depth"]["estimator"] == "synthetic"
    assert manifest["environment"]["sky_fraction"] == pytest.approx(0.3, abs=0.03)
    assert manifest["environment"]["ground"]["accepted"]
    assert manifest["camera"]["focal_source"] == "default"
    assert all(v > 0 for v in manifest["navigation"]["half_extents"])
    assert manifest["glb_bytes"] == len(result.glb)
    assert {"ingest", "depth", "layers", "environment", "mesh", "export", "total"} <= set(manifest["timings_ms"])


def test_mobile_profile_uses_16_bit_indices_everywhere(mobile_world):
    result, _ = mobile_world
    assert all(layer["index_type"] == "uint16" for layer in result.manifest["layers"].values())


def test_box_occluder_is_cut_out_of_the_foreground(mobile_world):
    result, _ = mobile_world
    stats = result.manifest["layers"]
    foreground = next(m for m in result.meshes if m.name == "foreground")
    # The near box is in front of the ground: its vertices are much nearer.
    nearest = float((-foreground.positions[:, 2]).min())
    assert nearest < 1.5
    assert stats["background"]["triangles"] > 0


def test_sky_and_dome_are_not_fogged_but_geometry_is(mobile_world):
    result, _ = mobile_world
    fog = {name: layer["fog"] for name, layer in result.manifest["layers"].items()}
    assert fog == {
        "dome": False, "sky": False, "ground": False,
        "background": True, "foreground": True, "silhouettes": True,
    }


def test_progress_is_reported_in_order(mobile_world):
    _, stages = mobile_world
    names = [stage for stage, _ in stages]
    assert names == ["start", "ingest", "depth", "layers", "environment", "mesh", "export"]
    fractions = [fraction for _, fraction in stages]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0


def test_heuristic_estimator_builds_without_any_model():
    result = build_world(scene_jpeg(), profile="mobile", estimator=HeuristicDepthEstimator())
    assert result.manifest["depth"]["estimator"] == "heuristic"
    assert result.manifest["environment"]["sky_fraction"] == 0.0  # PDD-REVIEW R11
    assert "sky" not in result.manifest["layers"]
    read_glb(result.glb)


def test_a_360_panorama_builds_a_wrapped_cylindrical_world():
    rgb = np.zeros((300, 2400, 3), np.uint8)
    rgb[:100] = (110, 160, 230)
    rgb[100:] = (80, 110, 50)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    result = build_world(buffer.getvalue(), profile="mobile", estimator=HeuristicDepthEstimator())
    camera = result.manifest["camera"]
    assert camera["projection"] == "cylindrical"
    assert camera["hfov_deg"] == pytest.approx(360.0, abs=0.5)
    # No side padding on a wrapped panorama: the edges meet instead.
    assert result.manifest["source"]["padded"][0] == result.manifest["source"]["working"][0]
    read_glb(result.glb)


def test_every_world_carries_a_textured_environment(mobile_world):
    result, _ = mobile_world
    glb = read_glb(result.glb)
    dome = glb.mesh("dome")["primitives"][0]
    assert "TEXCOORD_0" in dome["attributes"]
    texture = glb.document["materials"][dome["material"]]["pbrMetallicRoughness"]["baseColorTexture"]
    image = glb.document["textures"][texture["index"]]["source"]
    with Image.open(io.BytesIO(glb.image_bytes(image))) as environment:
        assert environment.size == (1024, 512)  # power of two: mipmaps on WebGL1 phones


def test_validator_accepts_built_worlds(mobile_world):
    from viewfinder.world.validate import validate_world

    result, _ = mobile_world
    assert validate_world(result) == []


def test_bad_input_is_an_input_error():
    with pytest.raises(ImageInputError):
        build_world(b"definitely not a photo", estimator=SceneEstimator())


def test_standalone_html_is_self_contained(mobile_world):
    result, _ = mobile_world
    html = standalone_html(result)
    assert 'src="static/' not in html and 'href="static/' not in html
    assert "window.__VIEWFINDER_WORLD__=" in html
    # The world payload must load before the viewer scripts that read it.
    assert html.index("__VIEWFINDER_WORLD__=") < html.index("WorldGlb = {")
    assert "<\\/script" not in html or "</script" in html
    assert re.search(r"<title>[^<]+· View Finder World</title>", html)


def test_write_world_writes_all_files(tmp_path, mobile_world):
    result, _ = mobile_world
    paths = write_world(result, tmp_path, "scene", html=True)
    assert {p.name for p in paths.values()} == {"scene_world.glb", "scene_world.json", "scene_world.html"}
    assert read_glb(paths["glb"].read_bytes())


def test_cli_build_and_exit_codes(tmp_path, capsys):
    photo = tmp_path / "scene.jpg"
    photo.write_bytes(scene_jpeg())
    code = world_cli(
        ["build", str(photo), "--depth-model", "heuristic", "--quality", "mobile", "--html", "--inpaint", "none"]
    )
    assert code == 0
    assert (tmp_path / "scene_world.glb").is_file() and (tmp_path / "scene_world.html").is_file()
    assert "World built" in capsys.readouterr().out
    assert world_cli(["build", str(tmp_path / "missing.jpg"), "--depth-model", "heuristic"]) == 2
    junk = tmp_path / "junk.jpg"
    junk.write_bytes(b"not a jpeg")
    assert world_cli(["build", str(junk), "--depth-model", "heuristic", "--inpaint", "none"]) == 2


def test_top_level_cli_routes_world(tmp_path):
    from viewfinder.cli import run

    photo = tmp_path / "scene.jpg"
    photo.write_bytes(scene_jpeg())
    assert run(["world", "build", str(photo), "--depth-model", "heuristic", "--quality", "mobile", "--inpaint", "none"]) == 0


@pytest.mark.skipif(not model_cached(), reason="Depth Anything V2 Small is not cached (run `view-finder world setup`)")
def test_real_depth_model_reads_near_as_larger_disparity():
    from viewfinder.world.depth import load_estimator

    estimator = load_estimator("depth-anything-v2-small", device="auto")
    rgb = np.asarray(Image.open(io.BytesIO(scene_jpeg())).convert("RGB"))
    disparity = estimator.estimate(rgb)
    assert disparity.shape == rgb.shape[:2] and disparity.dtype == np.float32
    assert disparity[-20:].mean() > disparity[:20].mean()  # ground below is nearer than sky above
    result = build_world(scene_jpeg(), profile="mobile", estimator=estimator)
    assert result.manifest["depth"]["model_id"].startswith("depth-anything/")
