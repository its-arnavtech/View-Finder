"""Optional regional OpenStreetMap POI enrichment for the local gazetteer."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import requests

from viewfinder.local_gazetteer import normalize_name

from geonames_builder import download_file


LOGGER = logging.getLogger(__name__)
REGION_ADMIN = {
    "nebraska": ("US", "NE"),
    "cyprus": ("CY", ""),
}
INTEREST_TAGS = (
    "tourism",
    "historic",
    "amenity",
    "shop",
    "leisure",
    "natural",
    "place",
    "aeroway",
    "railway",
    "highway",
)


def download_osm_regions(
    raw_dir: Path,
    regions: dict[str, str],
    *,
    user_agent: str,
) -> dict[str, Path]:
    session = requests.Session()
    outputs: dict[str, Path] = {}
    for name, url in regions.items():
        destination = raw_dir / f"{name}.osm.pbf"
        if not destination.is_file() or destination.stat().st_size == 0:
            download_file(url, destination, session=session, user_agent=user_agent)
        outputs[name] = destination
    return outputs


def _feature(tags: Any) -> tuple[str, str] | None:
    for key in INTEREST_TAGS:
        value = tags.get(key)
        if value:
            return "S", f"OSM_{key.upper()}_{str(value).upper()}"[:80]
    return None


def _names(tags: Any) -> list[str]:
    values = [
        tags.get("name"),
        tags.get("name:en"),
        tags.get("official_name"),
        tags.get("short_name"),
        tags.get("ref"),
        tags.get("nat_ref"),
        tags.get("int_ref"),
        tags.get("brand"),
        tags.get("addr:housename"),
    ]
    alt = tags.get("alt_name")
    if alt:
        values.extend(str(alt).split(";"))
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        normalized = normalize_name(cleaned)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(cleaned)
    return unique


def ingest_osm_region(
    pbf_path: Path,
    database_path: Path,
    *,
    region_name: str,
) -> int:
    try:
        import osmium
    except ImportError as exc:
        raise RuntimeError(
            "OSM ingestion requires the osmium Python package from "
            "viewfinder/requirements-local-evidence.txt"
        ) from exc
    if not pbf_path.is_file():
        raise FileNotFoundError(pbf_path)
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    country_code, admin1_code = REGION_ADMIN.get(region_name, ("", ""))
    connection = sqlite3.connect(database_path)
    buffer: list[tuple] = []
    inserted = 0

    def flush() -> None:
        nonlocal inserted
        if not buffer:
            return
        before = connection.total_changes
        connection.executemany(
            """
            INSERT OR REPLACE INTO osm_names(
                osm_id, normalized_name, name, lat, lon, country_code,
                admin1_code, feature_class, feature_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            buffer,
        )
        inserted += connection.total_changes - before
        buffer.clear()
        connection.commit()

    def append(kind: str, object_id: int, tags: Any, lat: float, lon: float) -> None:
        feature = _feature(tags)
        names = _names(tags)
        if feature is None or not names:
            return
        for index, name in enumerate(names):
            buffer.append(
                (
                    f"{region_name}:{kind}{object_id}:{index}",
                    normalize_name(name),
                    name,
                    lat,
                    lon,
                    country_code,
                    admin1_code,
                    feature[0],
                    feature[1],
                )
            )
        if len(buffer) >= 25_000:
            flush()

    class Handler(osmium.SimpleHandler):
        def node(self, node):
            if node.location.valid():
                append("n", node.id, node.tags, node.location.lat, node.location.lon)

        def way(self, way):
            coordinates = [
                (node.lat, node.lon)
                for node in way.nodes
                if node.location.valid()
            ]
            if coordinates:
                append(
                    "w",
                    way.id,
                    way.tags,
                    sum(lat for lat, _ in coordinates) / len(coordinates),
                    sum(lon for _, lon in coordinates) / len(coordinates),
                )

    try:
        Handler().apply_file(str(pbf_path), locations=True, idx="flex_mem")
        flush()
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (f"osm_region_{region_name}", str(pbf_path.resolve())),
        )
        connection.commit()
    finally:
        connection.close()
    LOGGER.info("Ingested %d OSM names from %s", inserted, region_name)
    return inserted
