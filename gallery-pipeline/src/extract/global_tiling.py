"""Population/land-aware global tile generation.

Uniform global tiling is wasteful: ~71% of the planet is ocean and much of the
land is uninhabited, so a fixed grid spends most of its tiles querying places
with no street-level imagery, while still being too coarse over dense cities.
This module generates a tile grid whose density follows where people (and thus
Mapillary photos) actually are.

Data source
-----------
GeoNames *cities1000* -- every settlement on Earth with population >= 1000,
144,563 points -- as bundled by the ``reverse_geocoder`` package. It ships
inside the wheel (~8 MB), so this is fully offline after ``pip install`` with no
dataset download. The package is already used by the downstream viewfinder CLI,
so it is a known, proven dependency.

City points are a deliberately simple but effective proxy for two things at
once:

* **Land vs water / inhabited vs empty.** Cities exist only on inhabited land,
  so a grid cell containing no city points is ocean, desert, ice, or wilderness
  and is skipped entirely. Empirically this drops ~86% of 1-degree cells.
* **Population density.** The *count* of city points in an area tracks how built
  up it is (validated: Paris 493, London 325, New York 244 per 1-degree cell,
  versus rural Kansas 7 and the Amazon 1). Denser cells get finer tiles.

Method
------
1. Bin all city points into coarse ``supercell_degrees`` squares (default 1x1
   degree) and count points per supercell.
2. Discard supercells with zero cities.
3. Map each surviving supercell's city count to a Mapillary-safe cell size via
   ``density_bands`` (denser -> smaller cells). The band sizes reflect what the
   live API tolerated in earlier runs: ~0.02 deg in dense cities, ~0.05-0.1 deg
   in rural areas. A cell the API still rejects as too dense is handled downstream
   by the extractor's automatic subdivision, so these are safe starting points.
4. Emit tiles. Two orderings are supported (``ordering``):

   * ``"spread"`` (default) -- round-robin one tile from each populated supercell
     per pass, densest supercells first. A bounded run (``max_tiles_per_run``)
     therefore samples many different cities worldwide rather than exhausting one
     metro, which is what "systematic global coverage" wants and what makes the
     grid resumable into broad coverage over many sessions.
   * ``"density"`` -- all of the densest supercell's tiles, then the next, etc.
     Concentrates effort on the biggest cities first.

Both orderings are fully deterministic, so the checkpoint store skips
already-completed tiles correctly across incremental, multi-session runs: each
session walks the same global sequence and simply resumes past what is done.

The generator is lazy and never materializes the full global tile set (which is
billions of tiles at city cell sizes); only the per-supercell density map
(~8,700 entries) is held in memory.
"""

from __future__ import annotations

import logging
import math
import os
from collections import deque
from typing import Iterator, Sequence

from src.extract.tiling import Tile, generate_tiles

LOGGER = logging.getLogger(__name__)

DEFAULT_SUPERCELL_DEGREES = 1.0

# Mapillary refuses any bbox larger than 0.01 square degrees. A 0.1x0.1 cell hits
# that limit exactly, so floating-point rounding makes such cells intermittently
# fail; cap cell sizes strictly below it (0.09^2 = 0.0081 < 0.01). Oversized cells
# from any other source are still handled by the extractor's subdivision, but the
# global grid should not emit them in the first place.
MAX_SAFE_CELL_DEGREES = 0.09

# (minimum city count, cell size in degrees), richest band first. Thresholds come
# from the observed cities1000 distribution (p95~=81, p90~=36, p75~=12); sizes
# from what the live Mapillary API accepted in earlier regional runs, kept under
# MAX_SAFE_CELL_DEGREES.
DEFAULT_DENSITY_BANDS: tuple[tuple[int, float], ...] = (
    (100, 0.02),  # major metros (Paris, London, NYC): densest tiles
    (30, 0.03),   # large cities (Tokyo, Delhi, Sao Paulo)
    (8, 0.05),    # towns / suburbia
    (1, 0.08),    # rural but inhabited (0.08 stays under the 0.01 sq-deg area cap)
)

# Module-level cache: loading and parsing 144k rows is worth doing once.
_city_points: list[tuple[float, float]] | None = None


def load_city_points() -> list[tuple[float, float]]:
    """Return every GeoNames cities1000 ``(lat, lon)``, cached for the process.

    Raises:
        RuntimeError: If ``reverse_geocoder`` (and thus its bundled dataset) is
            not installed.
    """
    global _city_points
    if _city_points is not None:
        return _city_points

    try:
        import reverse_geocoder
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Global tiling needs the reverse_geocoder package for its bundled "
            "GeoNames city dataset (pip install reverse_geocoder)."
        ) from exc

    import csv

    dataset = os.path.join(os.path.dirname(reverse_geocoder.__file__), "rg_cities1000.csv")
    points: list[tuple[float, float]] = []
    with open(dataset, encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header: lat,lon,name,admin1,admin2,cc
        for row in reader:
            try:
                points.append((float(row[0]), float(row[1])))
            except (ValueError, IndexError):
                continue

    if not points:
        raise RuntimeError(f"No city points loaded from {dataset}")
    _city_points = points
    LOGGER.info("Loaded %d GeoNames city points for global tiling", len(points))
    return points


def _within(lat: float, lon: float, bounds: Sequence[float] | None) -> bool:
    if bounds is None:
        return True
    min_lon, min_lat, max_lon, max_lat = bounds
    return min_lon <= lon < max_lon and min_lat <= lat < max_lat


def supercell_densities(
    supercell_degrees: float = DEFAULT_SUPERCELL_DEGREES,
    bounds: Sequence[float] | None = None,
) -> dict[tuple[int, int], int]:
    """Count city points per supercell, keyed by integer ``(lat_idx, lon_idx)``.

    Only populated supercells (count >= 1) appear. ``bounds`` optionally restricts
    to ``[min_lon, min_lat, max_lon, max_lat]``.
    """
    if supercell_degrees <= 0:
        raise ValueError("supercell_degrees must be positive")
    counts: dict[tuple[int, int], int] = {}
    for lat, lon in load_city_points():
        if not _within(lat, lon, bounds):
            continue
        key = (math.floor(lat / supercell_degrees), math.floor(lon / supercell_degrees))
        counts[key] = counts.get(key, 0) + 1
    return counts


def cell_size_for_count(count: int, bands: Sequence[tuple[int, float]]) -> float:
    """Map a supercell's city count to a cell size using ``bands`` (richest first).

    The result is clamped to ``MAX_SAFE_CELL_DEGREES`` so a misconfigured band can
    never emit a cell above Mapillary's bbox-area limit.
    """
    for minimum, size in bands:
        if count >= minimum:
            return min(size, MAX_SAFE_CELL_DEGREES)
    # Below the sparsest band's threshold; fall back to that band's size.
    return min(bands[-1][1], MAX_SAFE_CELL_DEGREES)


def _parse_bands(config_bands: object) -> tuple[tuple[int, float], ...]:
    """Accept YAML band entries as ``[min_cities, cell_size]`` pairs or mappings."""
    if not config_bands:
        return DEFAULT_DENSITY_BANDS
    parsed: list[tuple[int, float]] = []
    for entry in config_bands:  # type: ignore[union-attr]
        if isinstance(entry, dict):
            parsed.append((int(entry["min_cities"]), float(entry["cell_size"])))
        else:
            minimum, size = entry
            parsed.append((int(minimum), float(size)))
    # Richest band first so the first matching threshold wins.
    parsed.sort(key=lambda pair: pair[0], reverse=True)
    return tuple(parsed)


def global_tiles(config: dict) -> Iterator[Tile]:
    """Yield population-weighted tiles across the globe from ``config['tiling']``.

    Recognised ``tiling`` keys: ``supercell_degrees``, ``density_bands``,
    ``ordering`` (``"spread"`` | ``"density"``), and an optional ``bounds``
    ``[min_lon, min_lat, max_lon, max_lat]`` to scope a run (e.g. for testing)
    while keeping the density-adaptive behaviour.
    """
    tiling = config.get("tiling", {})
    supercell_degrees = float(tiling.get("supercell_degrees", DEFAULT_SUPERCELL_DEGREES))
    bands = _parse_bands(tiling.get("density_bands"))
    ordering = str(tiling.get("ordering", "spread")).lower()
    bounds = tiling.get("bounds")

    densities = supercell_densities(supercell_degrees, bounds)
    if not densities:
        LOGGER.warning("Global tiling found no populated supercells for the given bounds")
        return

    # Densest first; lat/lon tiebreakers keep the order fully deterministic so
    # incremental multi-session runs resume consistently.
    ordered = sorted(densities.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    LOGGER.info(
        "Global tiling: %d populated supercells (of %d possible), densest=%d cities, ordering=%s",
        len(ordered),
        int(360 / supercell_degrees) * int(180 / supercell_degrees),
        ordered[0][1],
        ordering,
    )

    def supercell_tiles(key: tuple[int, int], count: int) -> Iterator[Tile]:
        lat_idx, lon_idx = key
        cell = cell_size_for_count(count, bands)
        min_lat = lat_idx * supercell_degrees
        min_lon = lon_idx * supercell_degrees
        cell_bounds = [min_lon, min_lat, min_lon + supercell_degrees, min_lat + supercell_degrees]
        # A stable region label keeps tile ids readable; the id also encodes the
        # cell size and corner, so ids stay globally unique across bands.
        region = f"global_{count:04d}"
        for tile in generate_tiles(cell_bounds, cell, region):
            # Respect an explicit run bounds even at supercell edges.
            if bounds is None or _within(
                (tile.min_lat + tile.max_lat) / 2, (tile.min_lon + tile.max_lon) / 2, bounds
            ):
                yield tile

    if ordering == "density":
        for key, count in ordered:
            yield from supercell_tiles(key, count)
        return

    # "spread": round-robin one tile per supercell per pass, densest first.
    generators = deque((supercell_tiles(key, count) for key, count in ordered))
    while generators:
        gen = generators.popleft()
        try:
            yield next(gen)
        except StopIteration:
            continue
        generators.append(gen)  # not exhausted; back of the queue for the next pass
