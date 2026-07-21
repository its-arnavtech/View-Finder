"""Deterministic geographic tiling with optional denser priority regions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Iterable, Iterator


@dataclass(frozen=True, slots=True)
class Tile:
    tile_id: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    region: str = "world"

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _coordinate_key(value: Decimal) -> str:
    sign = "p" if value >= 0 else "m"
    return sign + format(abs(value), "f").replace(".", "d")


def generate_tiles(
    bounds: Iterable[float], cell_size_degrees: float, region: str = "world"
) -> Iterator[Tile]:
    """Yield stable, non-overrunning cells for [min_lon, min_lat, max_lon, max_lat]."""
    min_lon_f, min_lat_f, max_lon_f, max_lat_f = bounds
    if not 0 < cell_size_degrees <= 180:
        raise ValueError("cell_size_degrees must be in (0, 180]")
    if min_lon_f >= max_lon_f or min_lat_f >= max_lat_f:
        raise ValueError("bounds must have increasing longitude and latitude")
    if min_lon_f < -180 or max_lon_f > 180 or min_lat_f < -90 or max_lat_f > 90:
        raise ValueError("bounds are outside valid longitude/latitude limits")

    minimum_lon = Decimal(str(min_lon_f))
    minimum_lat = Decimal(str(min_lat_f))
    maximum_lon = Decimal(str(max_lon_f))
    maximum_lat = Decimal(str(max_lat_f))
    step = Decimal(str(cell_size_degrees))
    safe_region = "".join(char if char.isalnum() else "_" for char in region.lower()).strip("_")

    latitude = minimum_lat
    while latitude < maximum_lat:
        longitude = minimum_lon
        upper_latitude = min(latitude + step, maximum_lat)
        while longitude < maximum_lon:
            upper_longitude = min(longitude + step, maximum_lon)
            tile_id = (
                f"{safe_region}_{_coordinate_key(latitude)}_"
                f"{_coordinate_key(longitude)}_{_coordinate_key(step)}"
            )
            yield Tile(
                tile_id=tile_id,
                min_lon=float(longitude),
                min_lat=float(latitude),
                max_lon=float(upper_longitude),
                max_lat=float(upper_latitude),
                region=safe_region,
            )
            longitude = upper_longitude
        latitude = upper_latitude


def tiles_from_config(config: dict) -> Iterator[Tile]:
    tiling = config["tiling"]
    if tiling.get("mode", "regions") == "world":
        yield from generate_tiles(
            tiling.get("world_bounds", [-180.0, -85.0, 180.0, 85.0]),
            tiling.get("cell_size_degrees", 1.0),
            "world",
        )
        return
    for region in tiling.get("regions", []):
        yield from generate_tiles(
            region["bounds"],
            region.get("cell_size_degrees", tiling.get("cell_size_degrees", 1.0)),
            region["name"],
        )

