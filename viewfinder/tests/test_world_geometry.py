import numpy as np
import pytest

from viewfinder.world.camera import intrinsics_for
from viewfinder.world.geometry import (
    make_grid,
    navigation_bounds,
    near_side_of_jumps,
    normalize_disparity,
    to_scene_disparity,
    unproject,
)
from viewfinder.world.geometry import guided_filter, sharpen_depth_edges
from viewfinder.world.mesh import foreground_mesh, grid_triangles, silhouette_mask, surface_mesh
from viewfinder.world.profiles import PROFILES, get_profile


def test_normalization_ignores_excluded_sky():
    disparity = np.concatenate([np.zeros((50, 10)), np.linspace(2, 4, 500).reshape(50, 10)]).astype(np.float32)
    sky = np.zeros_like(disparity, bool)
    sky[:50] = True
    with_sky = normalize_disparity(disparity)
    without_sky = normalize_disparity(disparity, exclude=sky)
    # Excluding the sky stretches the real scene across the full [0, 1] range.
    assert without_sky[50:].min() == pytest.approx(0.0, abs=0.03)
    assert without_sky[50:].max() == pytest.approx(1.0, abs=0.03)
    assert with_sky[50:].min() > 0.4


def test_flat_disparity_does_not_divide_by_zero():
    assert np.all(normalize_disparity(np.full((8, 8), 3.0, np.float32)) == 0.5)


def test_scene_disparity_maps_to_near_and_far():
    np.testing.assert_allclose(to_scene_disparity(np.array([0.0, 1.0]), far=40.0), [1 / 40, 1.0])


def test_grid_matches_profile_long_side_and_aspect():
    grid = make_grid(1270, 952, 160)
    assert (grid.width, grid.height) == (160, 120)
    portrait = make_grid(952, 1270, 160)
    assert (portrait.width, portrait.height) == (120, 160)
    uv = grid.uv()
    assert 0 < uv.min() and uv.max() < 1


@pytest.mark.parametrize("size", [(4000, 4000), (2048, 256), (300, 1800), (1270, 952), (64, 48)])
def test_mobile_profile_stays_within_16_bit_indices(size):
    mobile = get_profile("mobile")
    grid = make_grid(*size, mobile.grid_long)
    assert grid.width * grid.height <= 0xFFFF


def test_panorama_grid_keeps_rows():
    grid = make_grid(2048, 256, 160)
    assert grid.height >= 40  # not a 20-row sliver
    assert grid.width <= 400


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="mobile"):
        get_profile("ultra")
    assert set(PROFILES) == {"mobile", "desktop", "high"}


def test_step_edge_is_cut_exactly_at_the_step():
    width, height = 20, 10
    disparity = np.full((height, width), 0.1, np.float32)
    disparity[:, 10:] = 0.8  # right half is a near wall
    grid = make_grid(width, height, width)
    positions = unproject(grid, intrinsics_for(width, height), 1.0 / disparity)
    mesh = surface_mesh("fg", positions, grid.uv(), disparity, np.ones_like(disparity, bool), 0.05, "photo")
    all_cells = (width - 1) * (height - 1) * 2
    # Only the column of cells straddling x=9|10 is dropped.
    assert mesh.triangle_count == all_cells - (height - 1) * 2
    near = near_side_of_jumps(disparity, 0.05)
    assert near[:, 10].all() and not near[:, 9].any() and near.sum() == height


def test_grazing_ground_plane_is_never_cut():
    """A plane's disparity is linear in the row, so it has no jumps even where
    its log-depth changes fastest (near the horizon). PDD-REVIEW build note."""
    height, width = 120, 160
    rows = np.linspace(0.02, 1.0, height, dtype=np.float32)[:, None]
    disparity = np.broadcast_to(rows, (height, width)).copy()
    grid = make_grid(width, height, width)
    positions = unproject(grid, intrinsics_for(width, height), 1.0 / disparity)
    mesh = surface_mesh("fg", positions, grid.uv(), disparity, np.ones_like(disparity, bool), 0.05, "photo")
    assert mesh.triangle_count == (width - 1) * (height - 1) * 2
    log_depth = np.log(1.0 / disparity)
    assert np.abs(np.diff(log_depth, axis=0)).max() > 0.09  # a log-depth test would have cut it


def test_soft_occlusion_ramps_become_steps_but_slopes_survive():
    width = 60
    x = np.arange(width, dtype=np.float32)
    ramp = np.clip((x - 25) / 10, 0, 1) * 0.6 + 0.2  # soft edge from 0.2 to 0.8 over 10 px
    slope = 0.3 + 0.002 * x  # a gentle floor
    disparity = np.vstack([np.tile(ramp, (20, 1)), np.tile(slope, (20, 1))])
    rgb = np.zeros((40, width, 3), np.uint8)
    rgb[:20, 30:] = 255  # the colour edge sits at x = 30, inside the depth ramp
    sharp = sharpen_depth_edges(disparity, rgb, jump=0.03, stride=2.0, ramp_px=12)
    row = sharp[10]
    assert {round(float(v), 3) for v in np.unique(row[20:40])} <= {0.2, 0.8}  # two levels: a step
    step = int(np.argmax(row > 0.5))
    assert abs(step - 30) <= 1  # placed on the colour edge, not the depth midpoint
    np.testing.assert_allclose(sharp[30], slope, atol=1e-6)  # the floor is untouched


def test_guided_filter_moves_depth_edges_onto_colour_edges():
    guide = np.zeros((20, 40), np.float32)
    guide[:, 20:] = 1.0
    blurred_step = np.clip((np.arange(40, dtype=np.float32) - 15) / 10, 0, 1)[None, :].repeat(20, 0)
    out = guided_filter(guide, blurred_step, radius=4, eps=1e-4)
    assert out[10, 17] < blurred_step[10, 17] and out[10, 22] > blurred_step[10, 22]


def test_edge_cells_are_extended_at_the_near_depth_and_kept_separate():
    width, height = 12, 8
    disparity = np.full((height, width), 0.1, np.float32)
    disparity[:, 6:] = 0.8  # near wall on the right
    grid = make_grid(width, height, width)
    rays = intrinsics_for(width, height).rays(grid.u, grid.v)
    surface, edges, cell_edge, cell_mid = foreground_mesh(
        "fg", rays, grid.uv(), disparity, np.ones_like(disparity, bool), 0.05, "photo"
    )
    assert cell_edge[:, 5].all() and cell_edge.sum() == height - 1
    assert surface.triangle_count == (width - 1) * (height - 1) * 2 - (height - 1) * 2
    assert edges.triangle_count == (height - 1) * 2
    # Every edge-cell vertex sits at the near wall's depth: no rubber sheet.
    np.testing.assert_allclose(-edges.positions[:, 2], 1 / 0.8, rtol=1e-5)
    np.testing.assert_allclose(cell_mid[:, 5], 0.45, atol=1e-6)


def test_silhouette_mask_discards_only_far_texels_inside_edge_cells():
    pixel = np.full((10, 20), 0.1, np.float32)
    pixel[:, 10:] = 0.8
    cell_edge = np.zeros((4, 9), bool)
    cell_edge[:, 4] = True  # cells are 19/9 px wide: cell 4 holds pixels 9 and 10
    cell_mid = np.full((4, 9), 0.45, np.float32)
    keep = silhouette_mask(pixel, cell_edge, cell_mid)
    assert keep[:, 10:].all()  # near texels kept
    assert not keep[:, 9].any()  # far texel inside the edge cell discarded
    assert keep[:, :9].all()  # far texels outside edge cells kept


def test_grid_triangles_cover_every_cell_once():
    triangles = grid_triangles(3, 4)
    assert triangles.shape == (12, 3)
    assert set(np.unique(triangles)) == set(range(12))


def test_navigation_bounds_scale_with_near_field():
    depth = np.full((10, 10), 4.0, np.float32)
    bounds = navigation_bounds(depth, np.ones_like(depth, bool))
    assert bounds["near_field_depth"] == pytest.approx(4.0)
    assert bounds["half_extents"] == pytest.approx([1.4, 0.8, 2.4])
