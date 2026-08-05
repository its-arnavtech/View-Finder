"""Offline geographic name resolution backed by a local SQLite database.

The database is produced by ``local-data-pipeline/build_local_data.py`` from
GeoNames and, optionally, regional OpenStreetMap extracts. Runtime lookup never
contacts a network service.
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .aggregator import haversine_km

DEFAULT_GAZETTEER_PATH = Path("local-data/geonames/geonames.sqlite3")
MAX_QUERY_WORDS = 8

_connection_cache: dict[str, sqlite3.Connection] = {}
_connection_lock = threading.Lock()


class GazetteerError(RuntimeError):
    """Raised when the local geographic database is absent or malformed."""


@dataclass(frozen=True)
class PlaceMatch:
    name: str
    matched_name: str
    lat: float
    lon: float
    country_code: str
    country_name: str
    admin1_code: str
    admin1_name: str
    feature_class: str
    feature_code: str
    population: int
    name_score: float
    source: str
    source_id: str
    query: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "name": self.name,
            "matched_name": self.matched_name,
            "latitude": round(self.lat, 6),
            "longitude": round(self.lon, 6),
            "country_code": self.country_code,
            "country_name": self.country_name,
            "admin1_code": self.admin1_code,
            "admin1_name": self.admin1_name,
            "feature_class": self.feature_class,
            "feature_code": self.feature_code,
            "population": self.population,
            "name_score": round(self.name_score, 4),
            "source": self.source,
            "source_id": self.source_id,
        }


def normalize_name(value: str) -> str:
    """Normalize OCR and gazetteer strings into a shared comparison form."""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_like = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", ascii_like, flags=re.UNICODE))


def phrase_candidates(lines: Iterable[str]) -> list[str]:
    """Return longest-first contiguous phrases suitable for exact name lookup."""
    candidates: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = normalize_name(line)
        words = normalized.split()
        if not words:
            continue
        upper = min(len(words), MAX_QUERY_WORDS)
        for length in range(upper, 1, -1):
            for start in range(0, len(words) - length + 1):
                phrase = " ".join(words[start : start + length])
                if phrase not in seen:
                    seen.add(phrase)
                    candidates.append(phrase)
        if len(words) == 1 and len(words[0]) >= 4 and normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)
    return candidates


def _connect(path: Path) -> sqlite3.Connection:
    resolved = str(path.expanduser().resolve())
    cached = _connection_cache.get(resolved)
    if cached is not None:
        return cached
    with _connection_lock:
        cached = _connection_cache.get(resolved)
        if cached is not None:
            return cached
        if not Path(resolved).is_file():
            raise GazetteerError(
                f"Local gazetteer not found: {resolved}. Run "
                "'python local-data-pipeline/build_local_data.py geonames-all'."
            )
        connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            version = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            connection.close()
            raise GazetteerError(f"Invalid gazetteer database {resolved}: {exc}") from exc
        if version is None or version["value"] != "1":
            connection.close()
            raise GazetteerError(
                f"Unsupported gazetteer schema in {resolved}; rebuild local data."
            )
        _connection_cache[resolved] = connection
        return connection


def _feature_weight(feature_class: str, feature_code: str) -> float:
    if feature_code.startswith(("PPL", "PCL", "ADM")):
        return 1.0
    if feature_class in {"S", "L", "T", "H"}:
        return 0.9
    return 0.7


def _row_to_match(row: sqlite3.Row, query: str, name_score: float) -> PlaceMatch:
    return PlaceMatch(
        name=str(row["canonical_name"]),
        matched_name=str(row["matched_name"]),
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        country_code=str(row["country_code"] or ""),
        country_name=str(row["country_name"] or ""),
        admin1_code=str(row["admin1_code"] or ""),
        admin1_name=str(row["admin1_name"] or ""),
        feature_class=str(row["feature_class"] or ""),
        feature_code=str(row["feature_code"] or ""),
        population=int(row["population"] or 0),
        name_score=float(name_score),
        source=str(row["source"]),
        source_id=str(row["source_id"]),
        query=query,
    )


class GeoNamesResolver:
    """Resolve detected text against local GeoNames and OSM name indexes."""

    def __init__(self, database_path: str | Path = DEFAULT_GAZETTEER_PATH) -> None:
        self.database_path = Path(database_path)
        self.connection = _connect(self.database_path)

    def search_exact(self, query: str, limit: int = 20) -> list[PlaceMatch]:
        normalized = normalize_name(query)
        if not normalized:
            return []
        rows = self.connection.execute(
            """
            SELECT n.name AS matched_name,
                   p.name AS canonical_name,
                   p.lat, p.lon, p.country_code, c.name AS country_name,
                   p.admin1_code, a.name AS admin1_name,
                   p.feature_class, p.feature_code, p.population,
                   'geonames' AS source, CAST(p.geoname_id AS TEXT) AS source_id
            FROM names n
            JOIN places p ON p.geoname_id = n.geoname_id
            LEFT JOIN countries c ON c.code = p.country_code
            LEFT JOIN admin1 a
              ON a.country_code = p.country_code AND a.admin1_code = p.admin1_code
            WHERE n.normalized_name = ?
            UNION ALL
            SELECT o.name AS matched_name,
                   o.name AS canonical_name,
                   o.lat, o.lon, o.country_code, '' AS country_name,
                   o.admin1_code, '' AS admin1_name,
                   o.feature_class, o.feature_code, 0 AS population,
                   'openstreetmap' AS source, o.osm_id AS source_id
            FROM osm_names o
            WHERE o.normalized_name = ?
            LIMIT ?
            """,
            (normalized, normalized, int(limit)),
        ).fetchall()
        matches = [
            _row_to_match(row, query, min(1.0, len(normalized.split()) / 3.0 + 0.34))
            for row in rows
        ]
        return self._rank(matches)[:limit]

    def search_fuzzy(self, query: str, limit: int = 20) -> list[PlaceMatch]:
        """Use FTS candidates plus RapidFuzz when exact OCR lookup fails."""
        normalized = normalize_name(query)
        words = normalized.split()
        if not words:
            return []
        fts_query = " AND ".join(f'"{word}"*' for word in words if len(word) >= 2)
        if not fts_query:
            return []
        try:
            rows = self.connection.execute(
                """
                SELECT f.normalized_name, f.display_name AS matched_name,
                       p.name AS canonical_name,
                       p.lat, p.lon, p.country_code, c.name AS country_name,
                       p.admin1_code, a.name AS admin1_name,
                       p.feature_class, p.feature_code, p.population,
                       'geonames' AS source, CAST(p.geoname_id AS TEXT) AS source_id
                FROM names_fts f
                JOIN places p ON p.geoname_id = CAST(f.geoname_id AS INTEGER)
                LEFT JOIN countries c ON c.code = p.country_code
                LEFT JOIN admin1 a
                  ON a.country_code = p.country_code AND a.admin1_code = p.admin1_code
                WHERE names_fts MATCH ?
                ORDER BY bm25(names_fts)
                LIMIT 250
                """,
                (fts_query,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        try:
            from rapidfuzz.fuzz import ratio

            similarity = lambda candidate: ratio(normalized, candidate) / 100.0
        except ImportError:
            from difflib import SequenceMatcher

            similarity = lambda candidate: SequenceMatcher(
                None, normalized, candidate
            ).ratio()

        matches: list[PlaceMatch] = []
        for row in rows:
            score = float(similarity(str(row["normalized_name"])))
            if score >= 0.78:
                matches.append(_row_to_match(row, query, score))
        return self._rank(matches)[:limit]

    def search_exact_near(
        self,
        query: str,
        lat: float,
        lon: float,
        *,
        radius_km: float = 50.0,
        limit: int = 20,
    ) -> list[PlaceMatch]:
        """Resolve an ambiguous name/ref using an independent visual vicinity."""
        normalized = normalize_name(query)
        if not normalized:
            return []
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / max(20.0, 111.0 * math.cos(math.radians(lat)))
        rows = self.connection.execute(
            """
            SELECT n.name AS matched_name, p.name AS canonical_name,
                   p.lat, p.lon, p.country_code, c.name AS country_name,
                   p.admin1_code, a.name AS admin1_name, p.feature_class,
                   p.feature_code, p.population, 'geonames' AS source,
                   CAST(p.geoname_id AS TEXT) AS source_id
            FROM names n
            JOIN places p ON p.geoname_id = n.geoname_id
            LEFT JOIN countries c ON c.code = p.country_code
            LEFT JOIN admin1 a
              ON a.country_code = p.country_code AND a.admin1_code = p.admin1_code
            WHERE n.normalized_name = ?
              AND p.lat BETWEEN ? AND ? AND p.lon BETWEEN ? AND ?
            UNION ALL
            SELECT o.name, o.name, o.lat, o.lon, o.country_code, '',
                   o.admin1_code, '', o.feature_class, o.feature_code, 0,
                   'openstreetmap', o.osm_id
            FROM osm_names o
            WHERE o.normalized_name = ?
              AND o.lat BETWEEN ? AND ? AND o.lon BETWEEN ? AND ?
            LIMIT 5000
            """,
            (
                normalized,
                lat - lat_delta,
                lat + lat_delta,
                lon - lon_delta,
                lon + lon_delta,
                normalized,
                lat - lat_delta,
                lat + lat_delta,
                lon - lon_delta,
                lon + lon_delta,
            ),
        ).fetchall()
        matches = [
            _row_to_match(
                row,
                query,
                min(1.0, len(normalized.split()) / 3.0 + 0.34),
            )
            for row in rows
            if haversine_km(lat, lon, float(row["lat"]), float(row["lon"]))
            <= radius_km
        ]
        return sorted(
            matches,
            key=lambda item: (
                haversine_km(lat, lon, item.lat, item.lon),
                -item.population,
            ),
        )[:limit]

    def resolve_lines(self, lines: Sequence[str], limit: int = 30) -> list[PlaceMatch]:
        matches: list[PlaceMatch] = []
        seen: set[tuple[str, str]] = set()
        phrases = phrase_candidates(lines)
        for phrase in phrases[:80]:
            found = self.search_exact(phrase, limit=10)
            if not found and len(phrase.split()) >= 2:
                found = self.search_fuzzy(phrase, limit=5)
            for match in found:
                key = (match.source, match.source_id)
                if key not in seen:
                    seen.add(key)
                    matches.append(match)
        return self._rank(matches)[:limit]

    def resolve_lines_near(
        self,
        lines: Sequence[str],
        lat: float,
        lon: float,
        *,
        radius_km: float = 50.0,
        limit: int = 30,
    ) -> list[PlaceMatch]:
        matches: list[PlaceMatch] = []
        seen: set[tuple[str, str]] = set()
        for phrase in phrase_candidates(lines)[:80]:
            for match in self.search_exact_near(
                phrase,
                lat,
                lon,
                radius_km=radius_km,
                limit=10,
            ):
                key = (match.source, match.source_id)
                if key not in seen:
                    seen.add(key)
                    matches.append(match)
        return matches[:limit]

    @staticmethod
    def _rank(matches: Sequence[PlaceMatch]) -> list[PlaceMatch]:
        def score(match: PlaceMatch) -> tuple[float, int, int]:
            prominence = min(math.log10(max(match.population, 1)) / 8.0, 1.0)
            total = (
                0.65 * match.name_score
                + 0.20 * _feature_weight(match.feature_class, match.feature_code)
                + 0.15 * prominence
            )
            return total, match.population, len(match.matched_name)

        return sorted(matches, key=score, reverse=True)
