"""Offline reverse geocoding for View Finder predictions.

The :mod:`reverse_geocoder` dataset is loaded lazily and cached in a
module-level singleton.  Constructing ``RGeocoder`` loads roughly 50 MB of
place data, so callers should use the public functions in this module rather
than constructing geocoder instances themselves.  The lock also ensures that
concurrent first calls initialize the dataset only once per process.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import Any

import reverse_geocoder as rg


_geocoder: Any | None = None
_geocoder_lock = threading.Lock()


def _get_geocoder() -> Any:
    """Return the process-wide, lazily initialized reverse geocoder.

    ``RGeocoder`` loads its bundled place dataset during construction.  The
    double-checked lock keeps that relatively expensive load to exactly once
    per process, including when multiple threads make the first call at the
    same time.
    """
    global _geocoder

    if _geocoder is None:
        with _geocoder_lock:
            if _geocoder is None:
                _geocoder = rg.RGeocoder(mode=1, verbose=False)
    return _geocoder


def _validate_coordinate(lat: float, lon: float, *, label: str) -> tuple[float, float]:
    """Validate and normalize one latitude/longitude pair."""
    if isinstance(lat, bool) or isinstance(lon, bool):
        raise ValueError(f"{label} latitude and longitude must be numbers, not booleans")

    try:
        normalized_lat = float(lat)
        normalized_lon = float(lon)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} latitude and longitude must be numeric") from exc

    if not math.isfinite(normalized_lat) or not math.isfinite(normalized_lon):
        raise ValueError(f"{label} latitude and longitude must be finite")
    if not -90.0 <= normalized_lat <= 90.0:
        raise ValueError(f"{label} latitude must be between -90 and 90 degrees")
    if not -180.0 <= normalized_lon <= 180.0:
        raise ValueError(f"{label} longitude must be between -180 and 180 degrees")

    return normalized_lat, normalized_lon


def _format_place(result: dict[str, Any]) -> dict[str, str]:
    """Convert a reverse_geocoder result to View Finder's public shape."""
    return {
        # reverse_geocoder exposes countries as ISO 3166-1 alpha-2 codes.
        "country": str(result.get("cc") or ""),
        "admin1": str(result.get("admin1") or ""),
        "city": str(result.get("name") or ""),
    }


def reverse_geocode(lat: float, lon: float) -> dict[str, str]:
    """Return the nearest known place to one coordinate, entirely offline.

    The returned ``country`` is the ISO 3166-1 alpha-2 country code supplied by
    :mod:`reverse_geocoder`; ``admin1`` and ``city`` contain the nearest
    first-level administrative region and locality.  Ocean and remote points
    use the library's nearest available match.

    The underlying ~50 MB dataset is initialized lazily and cached once per
    process.  This convenience function shares that cache with
    :func:`reverse_geocode_batch`.

    Raises:
        ValueError: If either coordinate is non-numeric, non-finite, or outside
            the valid latitude/longitude range.
    """
    return reverse_geocode_batch([(lat, lon)])[0]


def reverse_geocode_batch(
    coords: list[tuple[float, float]],
) -> list[dict[str, str]]:
    """Return nearest known places for a batch of coordinates, entirely offline.

    All coordinates are validated before the lazily cached dataset is loaded.
    The complete batch is then passed to ``RGeocoder.query`` in one call, which
    uses the package's efficient vectorized lookup path rather than repeatedly
    querying individual coordinates.  The dataset is initialized only once per
    process and is shared with :func:`reverse_geocode`.

    Args:
        coords: Latitude/longitude pairs in degrees.

    Returns:
        A result per input coordinate, in the same order.  An empty input
        returns an empty list without loading the geocoder dataset.

    Raises:
        ValueError: If a coordinate is not a two-item sequence or contains an
            invalid latitude or longitude.
    """
    validated: list[tuple[float, float]] = []
    for index, coordinate in enumerate(coords):
        if (
            not isinstance(coordinate, Sequence)
            or isinstance(coordinate, (str, bytes))
            or len(coordinate) != 2
        ):
            raise ValueError(f"coordinate at index {index} must be a (latitude, longitude) pair")

        validated.append(
            _validate_coordinate(
                coordinate[0],
                coordinate[1],
                label=f"coordinate at index {index}",
            )
        )

    if not validated:
        return []

    results = _get_geocoder().query(validated)
    return [_format_place(result) for result in results]
