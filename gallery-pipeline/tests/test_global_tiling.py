"""Tests for population/land-aware global tiling."""

import itertools
import math

import pytest

from src.extract.global_tiling import (
    DEFAULT_DENSITY_BANDS,
    cell_size_for_count,
    global_tiles,
    supercell_densities,
)


def test_cell_size_is_monotonic_in_density():
    """Denser supercells never get a larger cell than sparser ones."""
    previous = None
    for count in (500, 200, 100, 50, 30, 10, 8, 3, 1):
        size = cell_size_for_count(count, DEFAULT_DENSITY_BANDS)
        if previous is not None:
            assert size >= previous, f"count {count} got smaller cell than a denser one"
        previous = size


def test_cell_size_never_exceeds_area_limit():
    """No band emits a cell above Mapillary's bbox-area cap, even if misconfigured."""
    from src.extract.global_tiling import MAX_SAFE_CELL_DEGREES

    for count in (0, 1, 5, 50, 500):
        assert cell_size_for_count(count, DEFAULT_DENSITY_BANDS) <= MAX_SAFE_CELL_DEGREES
    # A deliberately oversized band is clamped down.
    assert cell_size_for_count(1, [(1, 0.5)]) <= MAX_SAFE_CELL_DEGREES


def test_ocean_and_desert_supercells_are_absent():
    """Uninhabited cells hold no cities, so they never appear in the density map."""
    dens = supercell_densities(1.0)
    # mid-Pacific, Sahara, and central Siberia have no cities1000 points.
    for lat, lon in [(0, -140), (23, 13), (66, 100)]:
        assert (lat, lon) not in dens


def test_known_cities_get_expected_bands():
    dens = supercell_densities(1.0)

    def size_at(lat, lon):
        count = dens.get((math.floor(lat), math.floor(lon)), 0)
        return cell_size_for_count(count, DEFAULT_DENSITY_BANDS) if count else None

    # Major metros tile finest; rural inhabited land tiles coarsest.
    assert size_at(48.86, 2.35) == 0.02   # Paris
    assert size_at(51.51, -0.13) == 0.02  # London
    assert size_at(-4.0, -63.0) == 0.08   # Amazon (sparse but inhabited)
    assert size_at(23.0, 13.0) is None    # Sahara -> skipped


def test_spread_samples_many_supercells():
    """A bounded spread run touches many distinct supercells, not one metro."""
    cfg = {"tiling": {"mode": "global", "ordering": "spread"}}
    first = list(itertools.islice(global_tiles(cfg), 30))
    regions = {tile.region for tile in first}
    assert len(regions) >= 25


def test_generation_is_deterministic():
    cfg = {"tiling": {"mode": "global", "ordering": "spread"}}
    a = [t.tile_id for t in itertools.islice(global_tiles(cfg), 40)]
    b = [t.tile_id for t in itertools.islice(global_tiles(cfg), 40)]
    assert a == b


def test_bounds_scopes_output():
    box = [-123.0, 36.5, -119.5, 39.0]
    cfg = {"tiling": {"mode": "global", "ordering": "spread", "bounds": box}}
    tiles = list(itertools.islice(global_tiles(cfg), 50))
    assert tiles
    for tile in tiles:
        clat = (tile.min_lat + tile.max_lat) / 2
        clon = (tile.min_lon + tile.max_lon) / 2
        assert box[0] <= clon < box[2]
        assert box[1] <= clat < box[3]


def test_density_ordering_starts_with_densest():
    """`density` ordering emits the busiest supercell's tiles first."""
    box = [-123.0, 36.5, -119.5, 39.0]
    dens = supercell_densities(1.0, box)
    busiest = max(dens, key=lambda k: dens[k])  # (lat_idx, lon_idx)
    cfg = {"tiling": {"mode": "global", "ordering": "density", "bounds": box}}
    first = next(global_tiles(cfg))
    assert math.floor(first.min_lat) == busiest[0]
    assert math.floor(first.min_lon) == busiest[1]


def test_tile_ids_are_unique_across_bands():
    """Tiles from supercells with different cell sizes never collide by id."""
    box = [-123.0, 36.5, -119.5, 39.0]
    cfg = {"tiling": {"mode": "global", "ordering": "spread", "bounds": box}}
    ids = [t.tile_id for t in itertools.islice(global_tiles(cfg), 200)]
    assert len(ids) == len(set(ids))
