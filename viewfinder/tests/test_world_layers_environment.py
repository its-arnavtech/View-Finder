import numpy as np
import pytest

from viewfinder.world.environment import (
    _extrapolate_disparity,
    dome_colors,
    dome_mesh,
    extend_frame,
    fit_ground,
    fog_density,
    ground_disc_mesh,
)
from viewfinder.world.layers import (
    background_layer,
    background_sources,
    detect_sky,
    occluder_pixels,
    propagate_background,
    sky_colors,
)

from world_fixtures import HORIZON_ROW, scene_disparity, scene_rgb


def test_sky_is_low_disparity_connected_to_the_top():
    disparity = np.full((100, 100), 0.6, np.float32)
    disparity[:30] = 0.0  # sky
    disparity[70:80, 40:60] = 0.0  # dark low-disparity blob NOT touching the top
    sky = detect_sky(disparity)
    assert sky[:30].mean() > 0.97
    assert not sky[70:80, 40:60].any()


def test_no_sky_for_indoor_scenes():
    assert not detect_sky(np.full((60, 80), 0.5, np.float32)).any()


def test_sky_colors_come_from_the_sky_or_the_top_band():
    rgb = scene_rgb()
    sky = np.zeros(rgb.shape[:2], bool)
    sky[:HORIZON_ROW] = True
    zenith, horizon = sky_colors(rgb, sky)
    np.testing.assert_allclose(zenith, np.array([110, 160, 230]) / 255, atol=0.01)
    np.testing.assert_allclose(horizon, np.array([110, 160, 230]) / 255, atol=0.01)
    zenith2, horizon2 = sky_colors(rgb, np.zeros_like(sky))
    assert np.all(zenith2 <= horizon2)


def test_occluders_are_only_the_near_side():
    disparity = np.full((40, 40), 0.1, np.float32)
    disparity[10:30, 10:30] = 0.9
    occluders = occluder_pixels(disparity, radius=3, jump=0.05)
    assert occluders[10:30, 10:13].all()  # near-side rim of the box
    assert not occluders[:, :10].any()  # background side keeps its real colour
    assert not occluders[15:25, 15:25].any()  # box interior is far from any edge


def test_background_colour_flows_from_the_far_side():
    rgb = np.zeros((20, 40, 3), np.uint8)
    rgb[:, :10] = (0, 200, 0)  # far background, left
    rgb[:, 10:] = (200, 0, 0)  # near box, right (wide: it has a deep interior)
    disparity = np.where(np.arange(40)[None, :] < 10, 0.1, 0.9).astype(np.float32) * np.ones((20, 1))
    unknown = occluder_pixels(disparity, radius=4, jump=0.05)
    sources = background_sources(disparity, radius=4, jump=0.05) & ~unknown
    assert unknown[:, 10:14].all()
    assert not sources[:, 10:18].any()  # the occluder's rim never donates colour
    filled = propagate_background(rgb, unknown, sources)
    assert (filled[:, 10:14, 1] == 200).all() and (filled[:, 10:14, 0] == 0).all()
    assert (filled[:, 20:, 0] == 200).all()  # interior untouched


def test_fill_copies_the_nearest_source_colour():
    """Pins the raster-order label convention of distanceTransformWithLabels."""
    rgb = np.zeros((9, 9, 3), np.uint8)
    sources = np.zeros((9, 9), bool)
    for (y, x), color in {(0, 0): 10, (0, 8): 20, (8, 0): 30, (8, 8): 40}.items():
        sources[y, x] = True
        rgb[y, x] = color
    unknown = ~sources
    filled = propagate_background(rgb, unknown, sources)
    assert filled[1, 1, 0] == 10 and filled[1, 7, 0] == 20
    assert filled[7, 1, 0] == 30 and filled[7, 7, 0] == 40


def test_background_layer_sits_behind_the_occluder():
    rgb = scene_rgb()
    pixel_disparity = 0.025 + scene_disparity() * 0.6
    grid_disparity = pixel_disparity[::4, ::4]
    sky = np.zeros_like(grid_disparity, bool)
    sky[: HORIZON_ROW // 4] = True
    from viewfinder.world.geometry import near_side_of_jumps

    near = near_side_of_jumps(grid_disparity, 0.05) & ~sky
    layer = background_layer(rgb, pixel_disparity, grid_disparity, sky, near, dilation_px=8, jump=0.05)
    assert layer.region.any() and not (layer.region & sky).any()
    covered = layer.region
    assert np.all(layer.disparity[covered] <= grid_disparity[covered] + 1e-6)
    assert np.all(layer.disparity[~covered] == grid_disparity[~covered])
    assert layer.rgb.shape[:2] == (rgb.shape[0] // 2, rgb.shape[1] // 2)


def test_frame_extension_continues_planes_and_fades_to_the_dome():
    height, width = 60, 80
    rows = np.linspace(0.2, 0.8, height, dtype=np.float32)[:, None]
    plane = np.broadcast_to(rows, (height, width)).copy()
    padded = _extrapolate_disparity(plane, 10, 10)
    # Bottom pad keeps rising like the plane (fading gradient), not a flat curtain.
    column = padded[:, 40]
    assert column[-1] > column[70] > plane[-1, 0] - 1e-6
    assert column[0] < plane[0, 0]

    rgb = np.full((height, width, 3), 120, np.uint8)
    fade = np.zeros((height + 20, width + 26, 3), np.float32)  # pads: 10 rows, 13 columns
    canvas = extend_frame(rgb, plane, np.zeros_like(plane, bool), (10, 13), fade_to=fade)
    assert canvas.rgb.shape[:2] == (80, 106)
    assert canvas.original.sum() == height * width
    assert canvas.rgb[0, 0].mean() < 30  # outer corner faded toward the (black) dome
    assert canvas.rgb[40, 53].mean() == pytest.approx(120, abs=1)  # photo untouched


def _plane_points(normal, offset, count=600, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    helper = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 0, 1.0])
    a = np.cross(normal, helper)
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    coords = rng.uniform(-3, 3, size=(count, 2))
    points = -normal * offset + coords[:, :1] * a + coords[:, 1:] * b
    points[:, 2] = -np.abs(points[:, 2]) - 2  # in front of the camera
    points += rng.normal(scale=noise, size=points.shape)
    return points


def test_ground_plane_is_found_under_the_camera():
    points = _plane_points([0, 1, 0], 1.5, noise=0.005)
    colors = np.full((len(points), 3), 0.4)
    plane = fit_ground(points, colors)
    assert plane.accepted, plane.reason
    assert plane.normal[1] > 0.99
    assert plane.offset == pytest.approx(1.5, abs=0.05)


def test_a_foreground_surface_with_scene_below_it_is_not_ground():
    """Dashcam photo: the dashboard is a clean up-facing plane in the lower
    frame, but the road is visible *below* it. Extending it would slice a
    slab through the road."""
    dashboard = _plane_points([0, 1, 0], 0.4, noise=0.002)
    road = _plane_points([0, 1, 0], 1.6, seed=1)
    colors = np.full((len(dashboard), 3), 0.1)
    assert fit_ground(dashboard, colors).accepted  # alone it looks like ground
    plane = fit_ground(dashboard, colors, scene_points=np.vstack([dashboard, road]))
    assert not plane.accepted and "below the fitted plane" in plane.reason
    # The true ground passes the same gate.
    assert fit_ground(road, colors, scene_points=np.vstack([dashboard, road])).accepted


def test_walls_and_too_few_points_are_not_ground():
    wall = _plane_points([1, 0, 0], 1.0)
    assert not fit_ground(wall, np.zeros((len(wall), 3))).accepted
    assert fit_ground(np.zeros((5, 3)), np.zeros((5, 3))).reason == "too few points"


def test_dome_colors_follow_elevation():
    zenith, horizon, ground = np.array([0, 0, 1.0]), np.array([1.0, 1, 1]), np.array([0, 1.0, 0])
    up = np.array([0, 1.0, 0])
    np.testing.assert_allclose(dome_colors(np.array([0, 1.0, 0]), up, zenith, horizon, ground), zenith)
    np.testing.assert_allclose(dome_colors(np.array([0, 0, -1.0]), up, zenith, horizon, ground), horizon)
    np.testing.assert_allclose(dome_colors(np.array([0, -1.0, 0]), up, zenith, horizon, ground), ground)
    positions, colors, indices = dome_mesh(100.0, 16, up, zenith, horizon, ground)
    np.testing.assert_allclose(np.linalg.norm(positions, axis=1), 100.0, rtol=1e-5)
    assert colors.dtype == np.uint8 and indices.max() < len(positions)


def test_ground_disc_is_sunk_below_the_plane():
    plane = fit_ground(_plane_points([0, 1, 0], 1.5), np.full((600, 3), 0.3))
    positions, colors, indices = ground_disc_mesh(plane, 50.0, 24, np.array([1.0, 1, 1]), sink=0.075)
    heights = positions @ plane.normal + plane.offset
    np.testing.assert_allclose(heights, -0.075, atol=1e-4)
    assert indices.max() < len(positions)


def test_environment_map_reproduces_the_photo_ahead_and_wraps_seamlessly():
    from viewfinder.world.camera import intrinsics_for
    from viewfinder.world.environment import environment_map, sample_environment

    rgb = np.zeros((300, 400, 3), np.uint8)
    rgb[:, :200] = (250, 0, 0)  # left half red
    rgb[:, 200:] = (0, 0, 250)  # right half blue
    camera = intrinsics_for(400, 300)
    env = environment_map(rgb, camera, (512, 256), np.array([0.5, 0.6, 0.9]), np.array([0.3, 0.3, 0.2]))
    ahead_left = sample_environment(env, np.array([[[-0.15, 0.0, -1.0]]]) / np.linalg.norm([-0.15, 0, -1]))
    ahead_right = sample_environment(env, np.array([[[0.15, 0.0, -1.0]]]) / np.linalg.norm([0.15, 0, -1]))
    assert ahead_left[0, 0, 0] > 150 and ahead_right[0, 0, 2] > 150
    # Behind the viewer the two continuations meet: the seam columns match.
    assert np.abs(env[:, 0].astype(int) - env[:, -1].astype(int)).max() < 12
    # Straight up and down fade to the supplied zenith / ground tones.
    np.testing.assert_allclose(env[0].mean(axis=0), [127, 153, 229], atol=30)
    np.testing.assert_allclose(env[-1].mean(axis=0), [76, 76, 51], atol=30)


def test_full_panorama_environment_is_the_panorama_all_round():
    from viewfinder.world.camera import intrinsics_for
    from viewfinder.world.environment import environment_map

    rgb = np.zeros((250, 2000, 3), np.uint8)
    rgb[:, :, 1] = np.linspace(0, 255, 2000).astype(np.uint8)[None, :]
    camera = intrinsics_for(2000, 250)
    assert camera.wraps
    env = environment_map(rgb, camera, (512, 256), np.zeros(3), np.zeros(3))
    horizon = env[128, :, 1].astype(float)
    assert np.isfinite(horizon).all()
    # Green rises with azimuth all the way round (no mirrored back arc).
    assert np.corrcoef(np.arange(512)[40:-40], horizon[40:-40])[0, 1] > 0.95


def test_ground_disc_takes_its_colour_from_the_surround():
    plane = fit_ground(_plane_points([0, 1, 0], 1.5), np.full((600, 3), 0.3))
    env = np.zeros((64, 128, 3), np.uint8)
    env[30:] = (10, 200, 10)  # everything from just above the horizon down is green
    _, colors, _ = ground_disc_mesh(plane, 50.0, 24, np.ones(3), sink=0.05, environment=env)
    assert (colors[:, 1] > 150).all()


def test_fog_density_hits_target_at_far():
    density = fog_density(40.0, 0.6)
    assert 1 - np.exp(-(density * 40.0) ** 2) == pytest.approx(0.6)
    assert 1 - np.exp(-(density * 5.0) ** 2) < 0.02  # near field stays clear
