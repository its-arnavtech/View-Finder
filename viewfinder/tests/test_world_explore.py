import numpy as np
import pytest

from viewfinder.world.explore import (
    EYE_HEIGHT,
    HEAL_DISTANCE,
    HEAL_STRENGTH,
    HOLE_STRENGTH,
    Canvas,
    bubble_disparity,
    bubble_mesh,
    floor_disparity_equirect,
    ground_height,
    heal_canvas,
    plan_stations,
    render_canvas,
)
from viewfinder.world.panorama import equirect_directions
from viewfinder.world.render import HOLE, SKY, SOLID, RendererUnavailable, WorldRenderer
from viewfinder.world.scene import GENERIC


def _level_ground_panorama(width=360, height=180, h=0.5, wall=4.0):
    """Inverse distance of a world with a level ground ``h`` below the eye
    and walls ``wall`` away all round (sky above them)."""
    directions = equirect_directions(width, height)
    y = directions[..., 1]
    horizontal = np.hypot(directions[..., 0], directions[..., 2])
    to_wall = wall / np.maximum(horizontal, 1e-6)
    to_ground = np.where(y < 0, h / np.maximum(-y, 1e-6), np.inf)
    distance = np.minimum(to_wall, to_ground)
    wall_top = wall * np.tan(np.radians(20))  # walls reach 20 degrees up
    distance = np.where((y > 0) & (distance * y > wall_top), np.inf, distance)
    return np.where(np.isfinite(distance), 1.0 / distance, 0.0).astype(np.float32)


def test_ground_height_is_read_off_the_floor_below():
    disparity = _level_ground_panorama(h=0.5)
    assert ground_height(disparity) == pytest.approx(0.5, rel=0.02)


def test_floor_prior_turns_reflections_into_water_not_holes():
    disparity = _level_ground_panorama(h=0.5)
    reflected = disparity.copy()
    reflected[120:150, 100:200] = 0.0  # sky reflected in water reads as infinitely far
    floor = floor_disparity_equirect(*disparity.shape, 0.5)
    fixed = np.maximum(reflected, floor)
    assert (fixed[120:150, 100:200] > 0).all()  # now a surface, at about the water's depth
    assert np.allclose(fixed[120:150, 100:200], disparity[120:150, 100:200] * 0.9, rtol=0.02)
    assert (floor[:88] == 0).all()  # never above the horizon


def test_stations_stay_clear_of_obstacles_and_start_ahead():
    open_world = plan_stations(_level_ground_panorama(h=0.5, wall=40.0), EYE_HEIGHT / 0.5)
    first, second = open_world[0].position, open_world[1].position
    assert first[2] == pytest.approx(-3.0) and abs(first[0]) < 1e-6  # walk into the photo first
    assert second[2] == pytest.approx(-6.5) and abs(second[0]) < 1e-6  # and on
    boxed_in = plan_stations(_level_ground_panorama(h=0.5, wall=2.0), EYE_HEIGHT / 0.5)
    # Walls 6.4 m away (2 units x 3.2 m/unit): 3 m stations fit, 6.5 m ones do not.
    assert boxed_in and all(np.linalg.norm(s.position) <= 3.0 + 1e-6 for s in boxed_in)


def _renderer():
    try:
        return WorldRenderer()
    except RendererUnavailable as exc:
        pytest.skip(f"no OpenGL: {exc}")


def _wall(z, half, colour):
    """A square facing the eye at distance ``-z`` on the -Z axis."""
    positions = np.array([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]], np.float32)
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)
    return positions, uvs, np.array([0, 1, 2, 0, 2, 3], np.uint32), np.full((8, 8, 3), colour, np.uint8)


def test_canvas_sees_the_world_all_the_way_round():
    renderer = _renderer()
    start = np.zeros(3)
    renderer.add(*_wall(-5.0, 2.0, (200, 40, 40)), station=start)  # ahead
    renderer.add(*_wall(-50.0, 30.0, (40, 40, 200)), station=np.array([0.0, 0.0, 9.0]))  # another station's
    canvas = render_canvas(renderer, np.zeros(3), (256, 128), show=[start], face=128)
    ahead = (64, 128)  # row, column of straight ahead
    assert canvas.coverage[ahead] == SOLID and canvas.distance[ahead] == pytest.approx(5.0, rel=0.02)
    assert tuple(canvas.rgb[ahead]) == (200, 40, 40)
    # Off-axis the distance is along the ray, not the face's axis.
    row, column = 64, 128 + 12  # 17 degrees right
    assert canvas.distance[row, column] == pytest.approx(5.0 / np.cos(np.radians(16.9)), rel=0.03)
    assert canvas.coverage[64, 0] == HOLE  # behind: nothing there
    assert not (canvas.rgb[..., 2] > 150).any()  # the other station's wall is not shown
    renderer.release()


class _RecordingBackend:
    name = "paint"

    def __init__(self):
        self.calls = []

    def inpaint(self, image, mask, prompt, negative, seed, strength=0.99, strength_map=None):
        self.calls.append((float(mask.mean()), strength))
        self.maps = getattr(self, "maps", []) + [strength_map]
        out = image.copy()
        stripes = np.zeros(image.shape[:2], np.uint8)
        stripes[:, ::2] = 120  # texture survives colour matching, a flat colour would not
        out[mask] = (40 + stripes[mask])[:, None]
        return out


def _canvas(width=256, height=128):
    """Near ground below the horizon, a far mountain band above it, sky on
    top, and a hole (nothing rendered) behind the station."""
    rgb = np.full((height, width, 3), 128, np.uint8)
    distance = np.zeros((height, width), np.float32)
    coverage = np.full((height, width), SKY, np.uint8)
    coverage[height // 2:] = SOLID
    distance[height // 2:] = 4.0  # near: repainted
    coverage[height // 2 - 10:height // 2] = SOLID
    distance[height // 2 - 10:height // 2] = 300.0  # far: kept
    rgb[height // 2 - 10:height // 2] = (20, 30, 40)
    coverage[height // 2 - 20:, :20] = HOLE  # behind the station (azimuth -180)
    coverage[height // 2 - 20:, -20:] = HOLE
    distance[coverage == HOLE] = 0.0
    return Canvas(rgb, distance, coverage, np.full((height, width), 2.0, np.float32))


def test_heal_fills_holes_and_repaints_only_what_is_near():
    canvas = _canvas()
    far_before = canvas.rgb[54:64].copy()
    backend = _RecordingBackend()
    filled, repainted, views = heal_canvas(canvas, GENERIC, backend, 64)
    strengths = {round(strength, 3) for _, strength in backend.calls}
    assert strengths == {HOLE_STRENGTH, HEAL_STRENGTH}  # views with holes, views with near surfaces only
    # Per pixel: holes generated outright, near surfaces repainted lightly.
    behind = next(m for m in backend.maps if np.isclose(m.max(), HOLE_STRENGTH))
    assert {round(float(v), 3) for v in np.unique(behind[behind > 0])} <= {HOLE_STRENGTH, HEAL_STRENGTH}
    assert views == len(backend.calls)
    assert filled[100, 5] and not filled[100, 128]
    assert repainted[100, 128] and not repainted[58, 128] and not repainted[10, 128]
    assert (canvas.rgb[54:64, 40:-40] == far_before[:, 40:-40]).all()  # far scenery untouched
    assert canvas.rgb[110, 100:150].astype(int).std() > 20  # near ground repainted
    assert canvas.distance[58, 128] > HEAL_DISTANCE


def test_bubble_depth_keeps_far_scenery_and_aligns_the_rest():
    canvas = _canvas()
    repainted = (canvas.coverage == SOLID) & (canvas.distance < HEAL_DISTANCE)
    filled = canvas.coverage == HOLE

    def estimate(rgb):  # a flat guess: aligned to the rendered depth around it
        return np.full(rgb.shape[:2], 2.0, np.float32)

    disparity = bubble_disparity(canvas, filled | repainted, estimate, np.zeros(3), None, (256, 128))
    assert disparity[58, 128] == pytest.approx(1.0 / 300.0, rel=1e-3)  # far: exactly as rendered
    assert disparity[10, 128] == 0.0  # sky stays sky
    assert disparity[100, 128] > 1.0 / 30.0  # near ground: near
    floor = floor_disparity_equirect(128, 256, EYE_HEIGHT)
    assert (disparity[70:, 30:-30] >= floor[70:, 30:-30] - 1e-6).all()  # never beyond the ground


def test_bubble_mesh_stands_at_its_station():
    eye = np.array([1.0, 0.0, -3.0])
    disparity = np.full((128, 256), 0.25, np.float32)  # everything 4 m away
    disparity[:30] = 0.0  # sky overhead
    texture = np.zeros((128, 256, 3), np.uint8)
    bubble = bubble_mesh("walk0", eye, texture, disparity, (64, 32), 0.3, 1e-3)
    assert bubble is not None and bubble.triangle_count > 0
    distances = np.linalg.norm(bubble.positions - eye, axis=1)
    assert np.allclose(distances, 4.0, rtol=1e-3)  # centred on the station
    assert bubble.mask.shape == (128, 256) and not bubble.mask[5].any() and bubble.mask[100].all()
    assert np.allclose(bubble.station, eye)


def test_badly_magnified_streaks_are_wiped_before_the_repaint():
    canvas = _canvas()
    canvas.rgb[64:, ::2] = 40  # hard streaks across the near ground
    canvas.rgb[64:, 1::2] = 220
    canvas.density[64:] = 0.05  # a texel smeared over 20 pixels

    class Seen(_RecordingBackend):
        inputs = []

        def inpaint(self, image, mask, prompt, negative, seed, strength=0.99, strength_map=None):
            self.inputs.append((image.copy(), strength))
            return super().inpaint(image, mask, prompt, negative, seed, strength, strength_map)

    backend = Seen()
    heal_canvas(canvas, GENERIC, backend, 64)
    image = next(image for image, strength in backend.inputs if strength >= HEAL_STRENGTH)
    # The repaint saw the ground's colour, not its streaks.
    assert np.abs(np.diff(image[56:, 20:44, 1].astype(int), axis=1)).mean() < 60


def test_the_sky_just_beyond_a_near_silhouette_is_repainted_with_it():
    canvas = _canvas()
    canvas.coverage[20:40, 100:140] = SOLID  # a near rock against the sky
    canvas.distance[20:40, 100:140] = 6.0
    _, repainted, _ = heal_canvas(canvas, GENERIC, _RecordingBackend(), 64)
    assert repainted[30, 120] and repainted[30, 98] and repainted[18, 120]  # the rock and a rim of sky
    assert not repainted[30, 60] and not repainted[5, 120]  # not the open sky


def test_stations_start_from_the_least_repainted_bubble_nearby():
    from viewfinder.world.explore import choose_source

    built = [np.zeros(3), np.array([0.0, 0.0, -3.0])]
    hops = [0, 1]
    # Equally near the photo's bubble and a repainted one: the photo's.
    assert choose_source(np.array([-2.6, 0.0, -1.5]), built, hops) == 0
    # Much nearer a repainted one (6.5 m from the photo, 3.5 m from it): that one.
    assert choose_source(np.array([0.0, 0.0, -6.5]), built, hops) == 1


def test_each_repaint_behind_a_station_strengthens_its_repaint():
    from viewfinder.world.explore import HOP_STRENGTH, MAX_HEAL_STRENGTH

    seen = {}
    for hops in (0, 1, 5):
        backend = _RecordingBackend()
        canvas = _canvas()
        canvas.coverage[canvas.coverage == HOLE] = SOLID  # near surfaces only
        canvas.distance[canvas.coverage == SOLID] = np.maximum(canvas.distance[canvas.coverage == SOLID], 4.0)
        heal_canvas(canvas, GENERIC, backend, 64, hops=hops)
        seen[hops] = max(strength for _, strength in backend.calls)
    assert seen[0] == pytest.approx(HEAL_STRENGTH)
    assert seen[1] == pytest.approx(HEAL_STRENGTH + HOP_STRENGTH)
    assert seen[5] == pytest.approx(MAX_HEAL_STRENGTH)


def test_no_standing_point_hangs_over_a_drop():
    # Standing on a hilltop: level ground for 2 m, then a valley far below.
    h = 0.5  # scene units; EYE_HEIGHT / 0.5 = 3.2 m per unit
    directions = equirect_directions(360, 180)
    y = directions[..., 1]
    horizontal = np.hypot(directions[..., 0], directions[..., 2])
    to_ground = np.where(y < 0, h / np.maximum(-y, 1e-6), np.inf)
    edge = 2.0 / 3.2  # the ground ends 2 m out
    beyond = to_ground * horizontal > edge
    distance = np.where(beyond & (y < 0), 300.0, to_ground)  # the valley, far below
    overlook = np.where(np.isfinite(distance), 1.0 / distance, 0.0).astype(np.float32)
    assert plan_stations(overlook, EYE_HEIGHT / h) == []
    flat = _level_ground_panorama(h=h, wall=40.0)
    assert plan_stations(flat, EYE_HEIGHT / h)  # the same spots on level ground are fine


def test_outdoor_stations_never_paint_treetops_with_the_open_sky_prompt():
    class Prompts(_RecordingBackend):
        seen = []

        def inpaint(self, image, mask, prompt, negative, seed, strength=0.99, strength_map=None):
            self.seen.append(negative)
            return super().inpaint(image, mask, prompt, negative, seed, strength, strength_map)

    canvas = _canvas()
    canvas.coverage[:30] = SOLID  # near treetops overhead
    canvas.distance[:30] = 8.0
    backend = Prompts()
    heal_canvas(canvas, GENERIC, backend, 64)
    assert backend.seen and not any("trees" in negative for negative in backend.seen)


def test_the_more_a_surface_is_stretched_the_less_of_it_is_kept():
    from viewfinder.world.explore import WIPED_STRENGTH

    canvas = _canvas()
    canvas.coverage[canvas.coverage == HOLE] = SOLID
    canvas.distance[:] = np.where(canvas.coverage == SOLID, 4.0, 0.0)
    canvas.density[64:, :128] = 0.65  # stretched 1.5x
    canvas.density[64:, 128:] = 2.0  # sharp
    backend = _RecordingBackend()
    heal_canvas(canvas, GENERIC, backend, 64)
    values = np.concatenate([m[m > 0] for m in backend.maps])
    assert values.min() == pytest.approx(HEAL_STRENGTH, abs=1e-3)  # the sharp half
    middle = values[(values > HEAL_STRENGTH + 0.01) & (values < WIPED_STRENGTH - 0.01)]
    assert middle.size  # the stretched half, in between
