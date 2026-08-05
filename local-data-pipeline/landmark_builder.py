"""Build a bounded, attributed Wikimedia Commons landmark retrieval gallery."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image

from viewfinder.aggregator import haversine_km


LOGGER = logging.getLogger(__name__)
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
LANDMARK_CODES = (
    "AIRP",
    "BAY",
    "CNYN",
    "CSTL",
    "FLLS",
    "ISL",
    "LK",
    "MNMT",
    "MSQE",
    "MUS",
    "MT",
    "PAL",
    "PK",
    "PRK",
    "RUIN",
    "STDM",
    "STM",
    "TMPL",
    "UNIV",
    "VAL",
)
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}


def _connect_catalog(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS targets (
            target_id INTEGER PRIMARY KEY AUTOINCREMENT,
            geoname_id INTEGER NOT NULL UNIQUE,
            name TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            country_code TEXT,
            admin1_code TEXT,
            feature_class TEXT,
            feature_code TEXT,
            population INTEGER NOT NULL DEFAULT 0,
            tier TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT
        );

        CREATE TABLE IF NOT EXISTS assets (
            page_id INTEGER PRIMARY KEY,
            target_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            commons_entity_id TEXT NOT NULL,
            wikidata_id TEXT,
            image_url TEXT NOT NULL,
            page_url TEXT NOT NULL,
            mime TEXT NOT NULL,
            commons_sha1 TEXT,
            license_short TEXT,
            license_url TEXT,
            artist TEXT,
            credit TEXT,
            description TEXT,
            camera_lat REAL,
            camera_lon REAL,
            landmark_lat REAL NOT NULL,
            landmark_lon REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            local_path TEXT,
            byte_count INTEGER,
            width INTEGER,
            height INTEGER,
            sha256 TEXT,
            phash TEXT,
            last_error TEXT,
            FOREIGN KEY(target_id) REFERENCES targets(target_id)
        );
        CREATE INDEX IF NOT EXISTS assets_status ON assets(status);
        CREATE INDEX IF NOT EXISTS assets_target ON assets(target_id);
        """
    )
    existing_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(assets)")
    }
    if "commons_entity_id" not in existing_columns:
        connection.execute(
            "ALTER TABLE assets ADD COLUMN commons_entity_id TEXT NOT NULL DEFAULT ''"
        )
    if "wikidata_id" not in existing_columns:
        connection.execute("ALTER TABLE assets ADD COLUMN wikidata_id TEXT")
    target_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(targets)")
    }
    if "priority" not in target_columns:
        connection.execute(
            "ALTER TABLE targets ADD COLUMN priority INTEGER NOT NULL DEFAULT 0"
        )
    connection.commit()
    return connection


def _target_rows(
    gazetteer: sqlite3.Connection,
    *,
    country: str | None,
    notable_limit: int,
    populated_limit: int,
) -> Iterable[sqlite3.Row]:
    country_clause = "country_code = ?" if country else "country_code <> 'US'"
    params: tuple[Any, ...] = (country,) if country else ()
    code_marks = ",".join("?" for _ in LANDMARK_CODES)
    notable = gazetteer.execute(
        f"""
        SELECT p.geoname_id, p.name, p.lat, p.lon, p.country_code, p.admin1_code,
               p.feature_class, p.feature_code, p.population,
               COUNT(n.geoname_id) AS alias_count
        FROM places p
        LEFT JOIN names n ON n.geoname_id = p.geoname_id
        WHERE {"p.country_code = ?" if country else "p.country_code <> 'US'"}
          AND p.feature_code IN ({code_marks})
        GROUP BY p.geoname_id
        ORDER BY p.population DESC, alias_count DESC, p.name
        LIMIT ?
        """,
        params + LANDMARK_CODES + (notable_limit * 4,),
    ).fetchall()
    # The alias-count ordering favors well-known features; the grid pass keeps
    # those results from collapsing into a few tourist-heavy cities.
    selected_notable: list[sqlite3.Row] = []
    remaining = list(notable)
    cell_pass = 0
    while remaining and len(selected_notable) < notable_limit:
        seen_cells: set[tuple[int, int]] = set()
        next_remaining: list[sqlite3.Row] = []
        for row in remaining:
            cell = (
                math.floor(float(row["lat"]) * 2),
                math.floor(float(row["lon"]) * 2),
            )
            if cell not in seen_cells and len(selected_notable) < notable_limit:
                seen_cells.add(cell)
                selected_notable.append(row)
            else:
                next_remaining.append(row)
        remaining = next_remaining
        cell_pass += 1
        if cell_pass > 100:
            break
    populated = gazetteer.execute(
        f"""
        SELECT geoname_id, name, lat, lon, country_code, admin1_code,
               feature_class, feature_code, population
        FROM places
        WHERE {country_clause}
          AND feature_code LIKE 'PPL%'
        ORDER BY population DESC, name
        LIMIT ?
        """,
        params + (populated_limit,),
    ).fetchall()
    yield from selected_notable
    yield from populated


def plan_targets(
    gazetteer_path: Path,
    catalog_path: Path,
    *,
    us_limit: int,
    world_limit: int,
    seed_geoname_ids: Iterable[int] = (),
    seed_targets: Iterable[dict[str, Any]] = (),
) -> dict[str, int]:
    if not gazetteer_path.is_file():
        raise FileNotFoundError(f"Build GeoNames first: {gazetteer_path}")
    gazetteer = sqlite3.connect(f"file:{gazetteer_path.resolve()}?mode=ro", uri=True)
    gazetteer.row_factory = sqlite3.Row
    catalog = _connect_catalog(catalog_path)
    inserted = {"us": 0, "world": 0}
    try:
        seed_ids = tuple(dict.fromkeys(int(value) for value in seed_geoname_ids))
        if seed_ids:
            marks = ",".join("?" for _ in seed_ids)
            for row in gazetteer.execute(
                f"""
                SELECT geoname_id, name, lat, lon, country_code, admin1_code,
                       feature_class, feature_code, population
                FROM places WHERE geoname_id IN ({marks})
                """,
                seed_ids,
            ):
                tier = "us" if row["country_code"] == "US" else "world"
                cursor = catalog.execute(
                    """
                    INSERT OR IGNORE INTO targets(
                        geoname_id, name, lat, lon, country_code, admin1_code,
                        feature_class, feature_code, population, tier, priority
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 100)
                    """,
                    (*row, tier),
                )
                inserted[tier] += int(cursor.rowcount > 0)
        for seed in seed_targets:
            tier = "us" if seed.get("country_code") == "US" else "world"
            cursor = catalog.execute(
                """
                INSERT OR IGNORE INTO targets(
                    geoname_id, name, lat, lon, country_code, admin1_code,
                    feature_class, feature_code, population, tier, priority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 100)
                """,
                (
                    int(seed["geoname_id"]),
                    str(seed["name"]),
                    float(seed["lat"]),
                    float(seed["lon"]),
                    str(seed.get("country_code") or ""),
                    str(seed.get("admin1_code") or ""),
                    str(seed.get("feature_class") or "H"),
                    str(seed.get("feature_code") or "BAY"),
                    int(seed.get("population") or 0),
                    tier,
                ),
            )
            inserted[tier] += int(cursor.rowcount > 0)
        plans = (
            ("us", "US", us_limit // 2, us_limit - us_limit // 2),
            ("world", None, world_limit // 2, world_limit - world_limit // 2),
        )
        for tier, country, notable_limit, populated_limit in plans:
            for row in _target_rows(
                gazetteer,
                country=country,
                notable_limit=notable_limit,
                populated_limit=populated_limit,
            ):
                cursor = catalog.execute(
                    """
                    INSERT OR IGNORE INTO targets(
                        geoname_id, name, lat, lon, country_code, admin1_code,
                        feature_class, feature_code, population, tier
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["geoname_id"],
                        row["name"],
                        row["lat"],
                        row["lon"],
                        row["country_code"],
                        row["admin1_code"],
                        row["feature_class"],
                        row["feature_code"],
                        row["population"],
                        tier,
                    ),
                )
                inserted[tier] += int(cursor.rowcount > 0)
            catalog.commit()
    finally:
        gazetteer.close()
        catalog.close()
    LOGGER.info("Landmark targets inserted: %s", inserted)
    return inserted


def _metadata_value(metadata: dict[str, Any], name: str) -> str:
    raw = metadata.get(name) or {}
    value = raw.get("value") if isinstance(raw, dict) else raw
    cleaned = re.sub(r"<[^>]+>", " ", html.unescape(str(value or "")))
    return re.sub(r"\s+", " ", cleaned).strip()


def _asset_from_page(
    page: dict[str, Any],
    target: sqlite3.Row,
    *,
    radius_km: float,
    thumbnail_width: int,
) -> tuple[Any, ...] | None:
    info = (page.get("imageinfo") or [{}])[0]
    mime = str(info.get("mime") or "")
    image_url = str(info.get("thumburl") or "")
    if not image_url and page.get("title"):
        filename = str(page["title"]).split(":", 1)[-1]
        image_url = (
            "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
            f"{quote(filename, safe='')}?width={int(thumbnail_width)}"
        )
    if mime not in ALLOWED_MIME or not image_url:
        return None
    coordinates = page.get("coordinates") or []
    camera_lat = camera_lon = None
    if coordinates:
        try:
            camera_lat = float(coordinates[0]["lat"])
            camera_lon = float(coordinates[0]["lon"])
        except (KeyError, TypeError, ValueError):
            camera_lat = camera_lon = None
    metadata = info.get("extmetadata") or {}
    if camera_lat is None:
        try:
            camera_lat = float(_metadata_value(metadata, "GPSLatitude"))
            camera_lon = float(_metadata_value(metadata, "GPSLongitude"))
        except (TypeError, ValueError):
            camera_lat = camera_lon = None
    if camera_lat is not None and haversine_km(
        float(target["lat"]),
        float(target["lon"]),
        camera_lat,
        camera_lon,
    ) > radius_km * 2:
        return None
    page_url = str(info.get("descriptionurl") or "")
    if not page_url:
        page_url = f"https://commons.wikimedia.org/?curid={page['pageid']}"
    return (
        int(page["pageid"]),
        int(target["target_id"]),
        str(page.get("title") or ""),
        f"M{int(page['pageid'])}",
        _metadata_value(metadata, "Wikidata"),
        image_url,
        page_url,
        mime,
        str(info.get("sha1") or ""),
        _metadata_value(metadata, "LicenseShortName"),
        _metadata_value(metadata, "LicenseUrl"),
        _metadata_value(metadata, "Artist"),
        _metadata_value(metadata, "Credit"),
        _metadata_value(metadata, "ImageDescription"),
        camera_lat,
        camera_lon,
        camera_lat if camera_lat is not None else float(target["lat"]),
        camera_lon if camera_lon is not None else float(target["lon"]),
    )


def discover_commons_assets(
    catalog_path: Path,
    config: dict[str, Any],
    *,
    user_agent: str,
    limit: int | None = None,
) -> dict[str, int]:
    catalog = _connect_catalog(catalog_path)
    images_per_target = int(config["images_per_target"])
    thumbnail_width = int(config["thumbnail_width"])
    delay = float(config.get("commons_request_delay_seconds", 0.2))
    workers = int(config.get("discovery_workers", 8))
    counters = {"targets": 0, "assets": 0, "errors": 0}
    sql = (
        "SELECT * FROM targets WHERE status IN ('pending', 'error') "
        "ORDER BY priority DESC, tier, target_id"
    )
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (int(limit),)
    targets = catalog.execute(sql, params).fetchall()
    rate_lock = threading.Lock()
    next_request = 0.0

    def throttled_get(request_params: dict[str, str]) -> requests.Response:
        nonlocal next_request
        with rate_lock:
            now = time.monotonic()
            wait = max(0.0, next_request - now)
            if wait:
                time.sleep(wait)
            next_request = max(time.monotonic(), next_request) + delay
        return requests.get(
            COMMONS_API,
            params=request_params,
            headers={"User-Agent": user_agent},
            timeout=(20, 60),
        )

    def discover(target: sqlite3.Row) -> tuple[sqlite3.Row, list[tuple], str | None]:
        radius_m = 10_000 if target["feature_code"].startswith("PPL") else 5_000
        request_params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "geosearch",
            "ggsprimary": "all",
            "ggsnamespace": "6",
            "ggscoord": f"{target['lat']}|{target['lon']}",
            "ggsradius": str(radius_m),
            "ggslimit": str(min(50, images_per_target * 5)),
            "prop": "imageinfo|coordinates",
            "iiprop": "url|mime|sha1|extmetadata|size",
            "colimit": "1",
        }
        try:
            response = None
            for attempt in range(6):
                response = throttled_get(request_params)
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
                retry_after = response.headers.get("Retry-After")
                response.close()
                try:
                    retry_delay = float(retry_after) if retry_after else 0.0
                except ValueError:
                    retry_delay = 0.0
                time.sleep(
                    max(
                        retry_delay,
                        min(60.0, 2.0**attempt + random.random()),
                    )
                )
            assert response is not None
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(str(payload["error"]))
            pages = (payload.get("query") or {}).get("pages") or []
            records = []
            for page in pages:
                record = _asset_from_page(
                    page,
                    target,
                    radius_km=radius_m / 1000.0,
                    thumbnail_width=thumbnail_width,
                )
                if record is not None:
                    records.append(record)
                if len(records) >= images_per_target:
                    break
            return target, records, None
        except Exception as exc:
            return target, [], str(exc)[:1000]

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(discover, target) for target in targets]
            for future in as_completed(futures):
                target, records, error = future.result()
                if error is None:
                    before = catalog.total_changes
                    catalog.executemany(
                        """
                        INSERT OR IGNORE INTO assets(
                            page_id, target_id, title, commons_entity_id, wikidata_id,
                            image_url, page_url, mime, commons_sha1, license_short,
                            license_url, artist, credit, description, camera_lat,
                            camera_lon, landmark_lat, landmark_lon
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        records,
                    )
                    added = catalog.total_changes - before
                    catalog.execute(
                        """
                        UPDATE targets SET status='discovered', attempts=attempts+1,
                                           last_error=NULL
                        WHERE target_id=?
                        """,
                        (target["target_id"],),
                    )
                    catalog.commit()
                    counters["targets"] += 1
                    counters["assets"] += added
                    if counters["targets"] % 100 == 0:
                        LOGGER.info("Commons discovery progress: %s", counters)
                else:
                    catalog.execute(
                        """
                        UPDATE targets
                        SET status='error', attempts=attempts+1, last_error=?
                        WHERE target_id=?
                        """,
                        (error, target["target_id"]),
                    )
                    catalog.commit()
                    counters["errors"] += 1
                    LOGGER.warning("Commons target %s failed: %s", target["name"], error)
    finally:
        catalog.close()
    return counters


def _extension(mime: str) -> str:
    return {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[mime]


def _download_asset(
    row: dict[str, Any],
    images_dir: Path,
    *,
    user_agent: str,
) -> dict[str, Any]:
    path = images_dir / f"{row['page_id']}{_extension(row['mime'])}"
    partial = path.with_suffix(path.suffix + ".part")
    try:
        response = None
        for attempt in range(6):
            response = requests.get(
                row["image_url"],
                headers={"User-Agent": user_agent},
                stream=True,
                timeout=(20, 120),
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            retry_after = response.headers.get("Retry-After")
            response.close()
            try:
                delay = float(retry_after) if retry_after else 0.0
            except ValueError:
                delay = 0.0
            delay = max(delay, min(60.0, 2.0**attempt + random.random()))
            time.sleep(delay)
        assert response is not None
        response.raise_for_status()
        sha256 = hashlib.sha256()
        byte_count = 0
        with partial.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    sha256.update(chunk)
                    byte_count += len(chunk)
        os.replace(partial, path)
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if width < 160 or height < 160:
                raise ValueError(f"image too small: {width}x{height}")
            try:
                import imagehash

                phash = str(imagehash.phash(image.convert("RGB")))
            except ImportError:
                phash = ""
        return {
            "page_id": row["page_id"],
            "status": "ready",
            "local_path": str(path.resolve()),
            "byte_count": byte_count,
            "width": width,
            "height": height,
            "sha256": sha256.hexdigest(),
            "phash": phash,
            "last_error": None,
        }
    except Exception as exc:
        if partial.exists():
            partial.unlink()
        return {
            "page_id": row["page_id"],
            "status": "error",
            "local_path": None,
            "byte_count": None,
            "width": None,
            "height": None,
            "sha256": None,
            "phash": None,
            "last_error": str(exc)[:1000],
        }


def download_commons_assets(
    catalog_path: Path,
    images_dir: Path,
    config: dict[str, Any],
    *,
    user_agent: str,
    limit: int | None = None,
) -> dict[str, int]:
    images_dir.mkdir(parents=True, exist_ok=True)
    catalog = _connect_catalog(catalog_path)
    budget = int(float(config["storage_budget_gb"]) * 1024**3)
    max_images = int(config["max_images"])
    existing_bytes = int(
        catalog.execute(
            "SELECT COALESCE(SUM(byte_count), 0) FROM assets WHERE status='ready'"
        ).fetchone()[0]
    )
    existing_count = int(
        catalog.execute("SELECT COUNT(*) FROM assets WHERE status='ready'").fetchone()[0]
    )
    remaining_count = max(0, max_images - existing_count)
    if limit is not None:
        remaining_count = min(remaining_count, int(limit))
    rows = [
        dict(row)
        for row in catalog.execute(
            """
            SELECT * FROM assets
            WHERE status IN ('pending', 'error')
            ORDER BY page_id
            LIMIT ?
            """,
            (remaining_count,),
        )
    ]
    workers = int(config.get("download_workers", 8))
    counters = {"ready": 0, "errors": 0, "bytes": existing_bytes}
    write_lock = threading.Lock()

    def persist(result: dict[str, Any]) -> bool:
        nonlocal existing_bytes
        with write_lock:
            size = int(result["byte_count"] or 0)
            if result["status"] == "ready" and existing_bytes + size > budget:
                path = result.get("local_path")
                if path and Path(path).is_file():
                    Path(path).unlink()
                return False
            catalog.execute(
                """
                UPDATE assets
                SET status=?, local_path=?, byte_count=?, width=?, height=?,
                    sha256=?, phash=?, last_error=?
                WHERE page_id=?
                """,
                (
                    result["status"],
                    result["local_path"],
                    result["byte_count"],
                    result["width"],
                    result["height"],
                    result["sha256"],
                    result["phash"],
                    result["last_error"],
                    result["page_id"],
                ),
            )
            catalog.commit()
            if result["status"] == "ready":
                existing_bytes += size
                counters["ready"] += 1
                counters["bytes"] = existing_bytes
            else:
                counters["errors"] += 1
            return True

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_download_asset, row, images_dir, user_agent=user_agent): row
                for row in rows
            }
            budget_reached = False
            for completed, future in enumerate(as_completed(futures), start=1):
                if future.cancelled():
                    continue
                result = future.result()
                if budget_reached:
                    path = result.get("local_path")
                    if path and Path(path).is_file():
                        Path(path).unlink()
                    continue
                if not persist(result):
                    LOGGER.warning("Landmark storage budget reached; stopping publication")
                    for pending in futures:
                        pending.cancel()
                    budget_reached = True
                if completed % 100 == 0:
                    LOGGER.info("Commons image download progress: %s", counters)
    finally:
        catalog.close()
    return counters


def _next_version(root: Path) -> tuple[int, Path]:
    root.mkdir(parents=True, exist_ok=True)
    versions = [
        int(path.name[1:])
        for path in root.iterdir()
        if path.is_dir() and re.fullmatch(r"v\d+", path.name)
    ]
    version = max(versions, default=0) + 1
    directory = root / f"v{version}"
    directory.mkdir()
    return version, directory


def build_landmark_index(
    catalog_path: Path,
    gallery_root: Path,
    config: dict[str, Any],
) -> Path:
    try:
        import faiss
        import open_clip
        import pyarrow as pa
        import pyarrow.parquet as pq
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Landmark indexing requires faiss, OpenCLIP, PyArrow, and Torch"
        ) from exc
    catalog = _connect_catalog(catalog_path)
    rows = catalog.execute(
        """
        SELECT a.*, t.geoname_id, t.name AS landmark_name, t.country_code,
               t.admin1_code, t.feature_class, t.feature_code
        FROM assets a JOIN targets t ON t.target_id = a.target_id
        WHERE a.status='ready'
        ORDER BY a.page_id
        """
    ).fetchall()
    if not rows:
        catalog.close()
        raise ValueError("No downloaded Commons assets are ready for indexing")

    model_name = str(config["model_name"])
    pretrained = str(config["pretrained"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = int(
        config["batch_size_gpu"] if device == "cuda" else config["batch_size_cpu"]
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    index = None
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        tensors = []
        for row in batch_rows:
            with Image.open(row["local_path"]) as image:
                tensors.append(preprocess(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            features = model.encode_image(batch).float()
            features = features / features.norm(dim=-1, keepdim=True)
        vectors = features.cpu().numpy().astype(np.float32)
        if index is None:
            index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        if start % (batch_size * 20) == 0:
            LOGGER.info("Landmark embeddings: %d/%d", start + len(batch_rows), len(rows))
    assert index is not None

    version, output = _next_version(gallery_root)
    index_tmp = output / "gallery.faiss.tmp"
    faiss.write_index(index, str(index_tmp))
    os.replace(index_tmp, output / "gallery.faiss")
    metadata = pa.table(
        {
            "index_position": pa.array(range(len(rows)), pa.int64()),
            "image_id": pa.array([str(row["page_id"]) for row in rows], pa.string()),
            "page_id": pa.array([row["page_id"] for row in rows], pa.int64()),
            "commons_entity_id": pa.array(
                [row["commons_entity_id"] or f"M{row['page_id']}" for row in rows],
                pa.string(),
            ),
            "wikidata_id": pa.array(
                [row["wikidata_id"] or "" for row in rows],
                pa.string(),
            ),
            "geoname_id": pa.array([row["geoname_id"] for row in rows], pa.int64()),
            "landmark_name": pa.array([row["landmark_name"] for row in rows], pa.string()),
            "lat": pa.array([row["landmark_lat"] for row in rows], pa.float64()),
            "lon": pa.array([row["landmark_lon"] for row in rows], pa.float64()),
            "camera_lat": pa.array([row["camera_lat"] for row in rows], pa.float64()),
            "camera_lon": pa.array([row["camera_lon"] for row in rows], pa.float64()),
            "country_code": pa.array([row["country_code"] or "" for row in rows], pa.string()),
            "admin1_code": pa.array([row["admin1_code"] or "" for row in rows], pa.string()),
            "feature_class": pa.array([row["feature_class"] or "" for row in rows], pa.string()),
            "feature_code": pa.array([row["feature_code"] or "" for row in rows], pa.string()),
            "title": pa.array([row["title"] for row in rows], pa.string()),
            "page_url": pa.array([row["page_url"] for row in rows], pa.string()),
            "license_short": pa.array([row["license_short"] or "" for row in rows], pa.string()),
            "license_url": pa.array([row["license_url"] or "" for row in rows], pa.string()),
            "artist": pa.array([row["artist"] or "" for row in rows], pa.string()),
            "credit": pa.array([row["credit"] or "" for row in rows], pa.string()),
            "phash": pa.array([row["phash"] or "" for row in rows], pa.string()),
            "sha256": pa.array([row["sha256"] or "" for row in rows], pa.string()),
            "local_path": pa.array([row["local_path"] for row in rows], pa.string()),
        }
    )
    pq.write_table(metadata, output / "metadata.parquet", compression="zstd")
    countries: dict[str, int] = {}
    for row in rows:
        code = str(row["country_code"] or "")
        countries[code] = countries.get(code, 0) + 1
    manifest = {
        "version": version,
        "status": "ready",
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_image_count": len(rows),
        "embedding_dimension": int(index.d),
        "embedding_model": f"open_clip:{model_name}:{pretrained}",
        "source": "Wikimedia Commons geosearch",
        "source_license": "Per-file; see metadata.parquet",
        "geographic_coverage": countries,
        "files": {"index": "gallery.faiss", "metadata": "metadata.parquet"},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    catalog.close()
    LOGGER.info("Built landmark gallery %s with %d images", output, len(rows))
    return output


def catalog_stats(catalog_path: Path) -> dict[str, Any]:
    connection = _connect_catalog(catalog_path)
    try:
        return {
            "targets": {
                row["status"]: row["count"]
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM targets GROUP BY status"
                )
            },
            "assets": {
                row["status"]: row["count"]
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM assets GROUP BY status"
                )
            },
            "downloaded_bytes": int(
                connection.execute(
                    "SELECT COALESCE(SUM(byte_count), 0) FROM assets WHERE status='ready'"
                ).fetchone()[0]
            ),
        }
    finally:
        connection.close()
