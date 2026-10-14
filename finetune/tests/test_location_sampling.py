"""Tests for location thinning.

The banded implementation must return the *identical* retained set as the
straightforward all-pairs version. Thinning decides what a batch actually
trains on, so an approximation here would silently change every downstream
result while looking like a speedup.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from location_sampling import (  # noqa: E402
    _banded_proximity_mask,
    haversine_km,
    thin_by_spacing,
)


def brute_force_thin(frame: pd.DataFrame, min_spacing_m: float, seed: int = 1337):
    """The original all-pairs implementation, kept as the reference."""
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    coords = shuffled[["lat", "lon"]].to_numpy(dtype=np.float64)
    keep_idx: list[int] = []
    kept = np.empty((0, 2))
    threshold_km = min_spacing_m / 1000.0
    for i, coord in enumerate(coords):
        if len(kept) == 0:
            keep_idx.append(i)
            kept = coord[None, :]
            continue
        d = haversine_km(coord[0], coord[1], kept[:, 0], kept[:, 1])
        if d.min() >= threshold_km:
            keep_idx.append(i)
            kept = np.vstack([kept, coord])
    return shuffled.iloc[keep_idx].reset_index(drop=True)


def _random_frame(n: int, seed: int, spread: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "lat": rng.uniform(-60 * spread, 60 * spread, n),
        "lon": rng.uniform(-180, 180, n),
        "image_path": [f"/tmp/{i}.jpg" for i in range(n)],
    })


def _clustered_frame(n: int, seed: int) -> pd.DataFrame:
    """Dense clusters, like street-level pulls where thinning actually bites."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-50, 50, (max(n // 20, 1), 2))
    rows = []
    for i in range(n):
        centre = centres[i % len(centres)]
        rows.append((centre[0] + rng.normal(0, 0.002), centre[1] + rng.normal(0, 0.002)))
    return pd.DataFrame({
        "lat": [r[0] for r in rows],
        "lon": [r[1] for r in rows],
        "image_path": [f"/tmp/{i}.jpg" for i in range(n)],
    })


@pytest.mark.parametrize("seed", range(4))
def test_banded_matches_brute_force_on_spread_points(seed):
    frame = _random_frame(600, seed)
    fast = thin_by_spacing(frame, min_spacing_m=100.0, seed=7)
    slow = brute_force_thin(frame, min_spacing_m=100.0, seed=7)
    assert fast["image_path"].tolist() == slow["image_path"].tolist()


@pytest.mark.parametrize("seed", range(4))
def test_banded_matches_brute_force_on_clustered_points(seed):
    """The case that matters: dense clusters where most points get rejected."""
    frame = _clustered_frame(600, seed)
    fast = thin_by_spacing(frame, min_spacing_m=100.0, seed=11)
    slow = brute_force_thin(frame, min_spacing_m=100.0, seed=11)
    assert fast["image_path"].tolist() == slow["image_path"].tolist()
    assert len(fast) < len(frame), "clustered input should actually be thinned"


@pytest.mark.parametrize("spacing", [50.0, 100.0, 500.0, 5000.0])
def test_equivalence_holds_across_thresholds(spacing):
    frame = _clustered_frame(400, seed=3)
    fast = thin_by_spacing(frame, min_spacing_m=spacing, seed=5)
    slow = brute_force_thin(frame, min_spacing_m=spacing, seed=5)
    assert fast["image_path"].tolist() == slow["image_path"].tolist()


def test_retained_points_really_are_separated():
    """Independent of the reference: check the output's own invariant."""
    frame = _clustered_frame(500, seed=9)
    kept = thin_by_spacing(frame, min_spacing_m=100.0, seed=1)
    coords = kept[["lat", "lon"]].to_numpy(dtype=np.float64)
    for i in range(len(coords)):
        others = np.delete(coords, i, axis=0)
        if len(others) == 0:
            continue
        d = haversine_km(coords[i, 0], coords[i, 1], others[:, 0], others[:, 1])
        assert d.min() >= 0.100 - 1e-9


def test_high_latitude_points_are_handled():
    """Longitude converges near the poles; latitude banding must still be exact."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "lat": rng.uniform(78.0, 82.0, 300),
        "lon": rng.uniform(-180, 180, 300),
        "image_path": [f"/tmp/{i}.jpg" for i in range(300)],
    })
    fast = thin_by_spacing(frame, min_spacing_m=100.0, seed=2)
    slow = brute_force_thin(frame, min_spacing_m=100.0, seed=2)
    assert fast["image_path"].tolist() == slow["image_path"].tolist()


def test_antimeridian_points_are_handled():
    frame = pd.DataFrame({
        "lat": [10.0, 10.0, 10.0, -10.0],
        "lon": [179.9999, -179.9999, 0.0, 179.9999],
        "image_path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
    })
    fast = thin_by_spacing(frame, min_spacing_m=100.0, seed=1)
    slow = brute_force_thin(frame, min_spacing_m=100.0, seed=1)
    assert fast["image_path"].tolist() == slow["image_path"].tolist()


def test_banded_is_substantially_faster_at_scale():
    """The whole point of the change; assert it, don't assume it."""
    frame = _random_frame(6000, seed=42)

    start = time.perf_counter()
    fast = thin_by_spacing(frame, min_spacing_m=100.0, seed=1)
    fast_seconds = time.perf_counter() - start

    start = time.perf_counter()
    slow = brute_force_thin(frame, min_spacing_m=100.0, seed=1)
    slow_seconds = time.perf_counter() - start

    assert fast["image_path"].tolist() == slow["image_path"].tolist()
    assert fast_seconds < slow_seconds, (
        f"banded {fast_seconds:.2f}s vs brute force {slow_seconds:.2f}s"
    )


def test_banded_split_proximity_matches_pairwise_exactly():
    rng = np.random.default_rng(77)
    reference = rng.uniform([-80, -180], [80, 180], size=(500, 2))
    query = rng.uniform([-80, -180], [80, 180], size=(300, 2))
    # Inject threshold violations, including across the antimeridian.
    query[:3] = reference[:3] + np.array([0.0001, 0.0001])
    reference[3] = [10.0, 179.9999]
    query[3] = [10.0, -179.9999]

    fast, _ = _banded_proximity_mask(query, reference, min_gap_m=100.0)
    distances = haversine_km(
        query[:, 0][:, None], query[:, 1][:, None],
        reference[:, 0][None, :], reference[:, 1][None, :],
    )
    slow = distances.min(axis=1) < 0.100
    assert fast.tolist() == slow.tolist()
