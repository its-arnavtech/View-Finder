import numpy as np

from viewfinder.world import pipeline
from viewfinder.world.panorama import equirect_directions
from viewfinder.world.planes import Plane, snap_depth


def _corridor(width=512, height=256, noise=0.02):
    """Inverse distance of a corridor: floor 1.6 m down, ceiling 1.4 m up,
    walls 1.2 m either side, open far ends; plus depth noise."""
    d = equirect_directions(width, height)
    x, y = d[..., 0], d[..., 1]
    truth = np.full(x.shape, 30.0)
    with np.errstate(divide="ignore"):
        for candidate in (np.where(y < 0, 1.6 / -y, np.inf), np.where(y > 0, 1.4 / y, np.inf),
                          np.where(x > 0, 1.2 / x, np.inf), np.where(x < 0, 1.2 / -x, np.inf)):
            truth = np.minimum(truth, candidate)
    noisy = truth * (1 + np.random.default_rng(0).normal(0, noise, truth.shape))
    return truth, (1 / noisy).astype(np.float32)


def test_corridor_planes_are_found_and_snapped():
    truth, disparity = _corridor()
    planes = pipeline._panorama_planes(disparity, 1.6)
    kinds = sorted(p.kind for p in planes)
    assert kinds == ["ceiling", "ground", "wall", "wall"]
    ceiling = next(p for p in planes if p.kind == "ceiling")
    assert abs(ceiling.offset - 1.4) < 0.05
    assert all(abs(p.offset - 1.2) < 0.05 for p in planes if p.kind == "wall")
    snapped = 1 / pipeline._snap_panorama(disparity, planes)
    before = np.median(np.abs(1 / disparity - truth) / truth)
    after = np.median(np.abs(snapped - truth) / truth)
    assert after < before / 20


def test_snapping_moves_points_only_along_their_rays():
    # A 40 x 40 patch of rays looking down at ground 1.6 m below, with 2% depth noise.
    u, v = np.meshgrid(np.linspace(-0.4, 0.4, 40), np.linspace(-0.9, -0.3, 40))
    rays = np.stack([u, v, -np.ones_like(u)], axis=-1).astype(np.float32)
    truth = 1.6 / -rays[..., 1]
    depth = (truth * (1 + np.random.default_rng(1).normal(0, 0.02, truth.shape))).astype(np.float32)
    ground = Plane(np.array([0.0, 1.0, 0.0]), 1.6, "ground", 1.0)
    out = snap_depth(np.zeros(3), rays, depth, [ground])
    inner = (slice(4, -4), slice(4, -4))  # normals are undefined at the very border
    assert np.allclose((rays * out[..., None])[inner][..., 1], -1.6, atol=1e-4)  # exactly on the ground
    assert np.allclose(out[inner], truth[inner], rtol=1e-4)  # moved along the ray, to the truth


def test_snapping_never_moves_a_point_beyond_the_tolerance():
    # A sphere of radius 5 around the eye: the planes a fit finds on it
    # (tangent bands) may flatten it, but never by more than the tolerance.
    from viewfinder.world.planes import SNAP_TOLERANCE

    d = equirect_directions(128, 64)
    disparity = np.full(d.shape[:2], 0.2, np.float32)
    planes = pipeline._panorama_planes(disparity, 5.0)
    snapped = pipeline._snap_panorama(disparity, [p for p in planes if p.kind != "ground"])
    assert np.max(np.abs(1 / snapped - 5.0) / 5.0) <= SNAP_TOLERANCE * 1.01
