"""Download and build the global offline GeoNames SQLite gazetteer."""

from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

import requests

from viewfinder.local_gazetteer import normalize_name


LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "1"
BASE_URL = "https://download.geonames.org/export/dump"
FILES = {
    "allCountries.zip": f"{BASE_URL}/allCountries.zip",
    "alternateNamesV2.zip": f"{BASE_URL}/alternateNamesV2.zip",
    "admin1CodesASCII.txt": f"{BASE_URL}/admin1CodesASCII.txt",
    "countryInfo.txt": f"{BASE_URL}/countryInfo.txt",
}


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


def download_file(
    url: str,
    destination: Path,
    *,
    session: requests.Session,
    user_agent: str,
) -> Path:
    """Resume one HTTP download and atomically publish it when complete."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": user_agent}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with session.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        if not append:
            offset = 0
        total = int(response.headers.get("Content-Length") or 0) + offset
        mode = "ab" if append else "wb"
        downloaded = offset
        last_report = time.monotonic()
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last_report >= 10:
                    LOGGER.info(
                        "Downloading %s: %s/%s",
                        destination.name,
                        _format_bytes(downloaded),
                        _format_bytes(total) if total else "unknown",
                    )
                    last_report = time.monotonic()
    os.replace(partial, destination)
    LOGGER.info("Downloaded %s (%s)", destination, _format_bytes(destination.stat().st_size))
    return destination


def download_geonames(raw_dir: Path, *, user_agent: str) -> dict[str, Path]:
    session = requests.Session()
    downloaded: dict[str, Path] = {}
    for filename, url in FILES.items():
        destination = raw_dir / filename
        if destination.is_file() and destination.stat().st_size > 0:
            LOGGER.info("Using existing %s", destination)
        else:
            download_file(url, destination, session=session, user_agent=user_agent)
        downloaded[filename] = destination
    return downloaded


def _zip_rows(path: Path, member_suffix: str) -> Iterator[list[str]]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(member_suffix)]
        if len(members) != 1:
            raise ValueError(f"Expected one *{member_suffix} member in {path}, found {members}")
        with archive.open(members[0]) as raw:
            import io

            with io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="") as text:
                yield from csv.reader(text, delimiter="\t")


def _text_rows(path: Path, *, comments: bool = False) -> Iterator[list[str]]:
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if comments and (not line.strip() or line.startswith("#")):
                continue
            yield line.rstrip("\n").split("\t")


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-262144;
        PRAGMA locking_mode=EXCLUSIVE;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE countries (
            code TEXT PRIMARY KEY,
            iso3 TEXT,
            name TEXT NOT NULL,
            continent TEXT
        ) WITHOUT ROWID;

        CREATE TABLE admin1 (
            country_code TEXT NOT NULL,
            admin1_code TEXT NOT NULL,
            name TEXT NOT NULL,
            ascii_name TEXT,
            geoname_id INTEGER,
            PRIMARY KEY (country_code, admin1_code)
        ) WITHOUT ROWID;

        CREATE TABLE places (
            geoname_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            ascii_name TEXT,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            feature_class TEXT,
            feature_code TEXT,
            country_code TEXT,
            admin1_code TEXT,
            population INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE names (
            geoname_id INTEGER NOT NULL,
            normalized_name TEXT NOT NULL,
            name TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT '',
            is_preferred INTEGER NOT NULL DEFAULT 0,
            is_short INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL,
            PRIMARY KEY (geoname_id, normalized_name)
        ) WITHOUT ROWID;

        CREATE TABLE osm_names (
            osm_id TEXT PRIMARY KEY,
            normalized_name TEXT NOT NULL,
            name TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            country_code TEXT NOT NULL DEFAULT '',
            admin1_code TEXT NOT NULL DEFAULT '',
            feature_class TEXT NOT NULL DEFAULT 'S',
            feature_code TEXT NOT NULL DEFAULT 'OSM'
        ) WITHOUT ROWID;
        CREATE INDEX osm_names_normalized ON osm_names(normalized_name);
        """
    )


def _batched(iterable: Iterable[tuple], size: int = 50_000) -> Iterator[list[tuple]]:
    batch: list[tuple] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _country_records(path: Path) -> Iterator[tuple[str, str, str, str]]:
    for fields in _text_rows(path, comments=True):
        if len(fields) >= 9:
            yield fields[0], fields[1], fields[4], fields[8]


def _admin_records(path: Path) -> Iterator[tuple[str, str, str, str, int | None]]:
    for fields in _text_rows(path):
        if len(fields) < 4 or "." not in fields[0]:
            continue
        country, admin = fields[0].split(".", 1)
        try:
            geoname_id = int(fields[3])
        except ValueError:
            geoname_id = None
        yield country, admin, fields[1], fields[2], geoname_id


def _place_and_names(
    path: Path,
) -> Iterator[tuple[tuple, list[tuple]]]:
    for fields in _zip_rows(path, "allCountries.txt"):
        if len(fields) < 15:
            continue
        try:
            geoname_id = int(fields[0])
            lat = float(fields[4])
            lon = float(fields[5])
            population = int(fields[14] or 0)
        except ValueError:
            continue
        name = fields[1]
        ascii_name = fields[2]
        place = (
            geoname_id,
            name,
            ascii_name,
            lat,
            lon,
            fields[6],
            fields[7],
            fields[8],
            fields[10],
            population,
        )
        variants = [name, ascii_name]
        variants.extend(value for value in fields[3].split(",") if value)
        seen: set[str] = set()
        names: list[tuple] = []
        for value in variants:
            normalized = normalize_name(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            names.append(
                (
                    geoname_id,
                    normalized,
                    value,
                    "",
                    int(value == name),
                    0,
                    "allCountries",
                )
            )
        yield place, names


def _alternate_records(path: Path) -> Iterator[tuple]:
    for fields in _zip_rows(path, "alternateNamesV2.txt"):
        if len(fields) < 6:
            continue
        try:
            geoname_id = int(fields[1])
        except ValueError:
            continue
        value = fields[3]
        normalized = normalize_name(value)
        if not normalized:
            continue
        yield (
            geoname_id,
            normalized,
            value,
            fields[2],
            int(fields[4] == "1"),
            int(fields[5] == "1"),
            "alternateNamesV2",
        )


def build_geonames_database(raw_dir: Path, database_path: Path) -> Path:
    required = {name: raw_dir / name for name in FILES}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"GeoNames source files are missing: {missing}")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = database_path.with_suffix(database_path.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        _create_schema(connection)
        connection.executemany(
            "INSERT INTO countries(code, iso3, name, continent) VALUES (?, ?, ?, ?)",
            _country_records(required["countryInfo.txt"]),
        )
        connection.executemany(
            """
            INSERT INTO admin1(country_code, admin1_code, name, ascii_name, geoname_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            _admin_records(required["admin1CodesASCII.txt"]),
        )
        connection.commit()

        place_count = 0
        name_count = 0
        place_batch: list[tuple] = []
        name_batch: list[tuple] = []
        for place, names in _place_and_names(required["allCountries.zip"]):
            place_batch.append(place)
            name_batch.extend(names)
            if len(place_batch) >= 50_000:
                connection.executemany(
                    """
                    INSERT INTO places(
                        geoname_id, name, ascii_name, lat, lon, feature_class,
                        feature_code, country_code, admin1_code, population
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    place_batch,
                )
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO names(
                        geoname_id, normalized_name, name, language,
                        is_preferred, is_short, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    name_batch,
                )
                place_count += len(place_batch)
                name_count += len(name_batch)
                place_batch.clear()
                name_batch.clear()
                connection.commit()
                LOGGER.info("GeoNames places=%d base_names=%d", place_count, name_count)
        if place_batch:
            connection.executemany(
                """
                INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                place_batch,
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO names VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                name_batch,
            )
            place_count += len(place_batch)
            name_count += len(name_batch)
            connection.commit()

        alternate_count = 0
        for batch in _batched(_alternate_records(required["alternateNamesV2.zip"])):
            connection.executemany(
                """
                INSERT OR IGNORE INTO names(
                    geoname_id, normalized_name, name, language,
                    is_preferred, is_short, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            alternate_count += len(batch)
            connection.commit()
            if alternate_count % 500_000 < len(batch):
                LOGGER.info("GeoNames alternate names processed=%d", alternate_count)

        LOGGER.info("Creating exact name index")
        connection.execute("CREATE INDEX names_normalized ON names(normalized_name)")
        LOGGER.info("Creating FTS5 fuzzy-search index; this is the longest database step")
        connection.execute(
            """
            CREATE VIRTUAL TABLE names_fts USING fts5(
                normalized_name,
                display_name UNINDEXED,
                geoname_id UNINDEXED,
                tokenize='unicode61 remove_diacritics 2'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO names_fts(normalized_name, display_name, geoname_id)
            SELECT normalized_name, name, geoname_id FROM names
            """
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "source": "GeoNames allCountries + alternateNamesV2",
            "license": "CC BY 4.0",
            "place_count": str(
                connection.execute("SELECT COUNT(*) FROM places").fetchone()[0]
            ),
            "name_count": str(
                connection.execute("SELECT COUNT(*) FROM names").fetchone()[0]
            ),
            "built_at_unix": str(int(time.time())),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, database_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "database": str(database_path.resolve()),
        "database_bytes": database_path.stat().st_size,
        "sources": {
            name: {"url": FILES[name], "bytes": path.stat().st_size}
            for name, path in required.items()
        },
        "license": "GeoNames CC BY 4.0",
    }
    database_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Built %s (%s)", database_path, _format_bytes(database_path.stat().st_size))
    return database_path

