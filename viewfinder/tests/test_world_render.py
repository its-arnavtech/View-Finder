import numpy as np
import pytest

from viewfinder.world.render import DOME, HOLE, SKY, SOLID, Camera, RendererUnavailable, WorldRenderer


def _quad(z, half=1.0):
    positions = np.array([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]], np.float32)
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], np.uint32)
    return positions, uvs, indices


def test_depth_colour_and_holes():
    try:
        r = WorldRenderer()
    except RendererUnavailable as exc:
        pytest.skip(f"no OpenGL: {exc}")
    red = np.zeros((8, 8, 3), np.uint8)
    red[...] = (200, 30, 30)
    r.add(*_quad(-4.0), red)
    frame = r.render(Camera.looking(np.zeros(3), 0.0, 0.0, 60.0, 64, 64))
    centre = frame.coverage[32, 32]
    assert centre == SOLID and frame.depth[32, 32] == pytest.approx(4.0, rel=1e-3)
    assert tuple(frame.rgb[32, 32]) == (200, 30, 30)
    assert frame.coverage[0, 0] == HOLE and frame.depth[0, 0] == 0.0  # nothing drawn there
    r.release()


def test_masks_hide_texels():
    try:
        r = WorldRenderer()
    except RendererUnavailable as exc:
        pytest.skip(f"no OpenGL: {exc}")
    blue = np.full((8, 8, 3), (20, 20, 200), np.uint8)
    green = np.full((8, 8, 3), (20, 200, 20), np.uint8)
    keep_left = np.zeros((8, 8), np.uint8)
    keep_left[:, :4] = 255
    r.add(*_quad(-5.0), blue)
    r.add(*_quad(-4.0), green, mask=keep_left)
    frame = r.render(Camera.looking(np.zeros(3), 0.0, 0.0, 60.0, 64, 64))
    assert frame.rgb[32, 26][1] > 150  # left half: the nearer quad
    assert frame.rgb[32, 38][2] > 150  # right half: masked out, the farther one shows
    r.release()


def test_only_the_nearest_stations_meshes_show():
    try:
        r = WorldRenderer()
    except RendererUnavailable as exc:
        pytest.skip(f"no OpenGL: {exc}")
    blue = np.full((8, 8, 3), (20, 20, 200), np.uint8)
    green = np.full((8, 8, 3), (20, 200, 20), np.uint8)
    red = np.full((8, 8, 3), (200, 20, 20), np.uint8)
    start, near, far = np.zeros(3), np.array([0.0, 0.0, -3.0]), np.array([0.0, 0.0, -6.5])
    r.add(*_quad(-10.0, 4.0), blue, station=start)
    r.add(*_quad(-9.0, 4.0), green, station=near)
    r.add(*_quad(-8.0, 4.0), red, station=far)  # nearer the camera, but not its station's turn
    at = lambda eye: r.render(Camera.looking(eye, 0.0, 0.0, 40.0, 32, 32)).rgb[16, 16]  # noqa: E731
    assert at(np.zeros(3))[2] > 150  # the start point shows its own world only
    assert at(near + np.array([0.3, 0.0, 0.0]))[1] > 150
    assert at(far + np.array([0.0, 0.0, 0.5]))[0] > 150
    shown = r.render(Camera.looking(np.zeros(3), 0.0, 0.0, 40.0, 32, 32), show=[near]).rgb[16, 16]
    assert shown[1] > 150  # explicitly chosen stations show from anywhere
    r.release()


def test_dome_is_sky_or_hole_by_the_land_mask():
    try:
        r = WorldRenderer()
    except RendererUnavailable as exc:
        pytest.skip(f"no OpenGL: {exc}")
    land = np.zeros((8, 8), np.uint8)
    land[:, :4] = 255  # left half of the dome texture holds land
    r.add(*_quad(-50.0, 100.0), np.full((8, 8, 3), 128, np.uint8), kind=DOME, land=land)
    frame = r.render(Camera.looking(np.zeros(3), 0.0, 0.0, 60.0, 64, 64))
    assert frame.coverage[32, 10] == HOLE and frame.coverage[32, 54] == SKY
    r.release()


def test_magnified_textures_report_low_density():
    try:
        r = WorldRenderer()
    except RendererUnavailable as exc:
        pytest.skip(f"no OpenGL: {exc}")
    tiny = np.full((4, 4, 3), 90, np.uint8)  # 4 texels across a quad that fills the view
    r.add(*_quad(-1.0, 2.0), tiny)
    frame = r.render(Camera.looking(np.zeros(3), 0.0, 0.0, 60.0, 128, 128))
    assert frame.density[64, 64] < 0.1
    r.release()
