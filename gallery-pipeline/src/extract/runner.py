"""Extraction stage orchestration independent of Airflow."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.extract.checkpoint_store import CheckpointStore, utc_now
from src.extract.mapillary_client import MapillaryClient
from src.extract.tiling import Tile, tiles_from_config


LOGGER = logging.getLogger(__name__)


def select_pending_tiles(config: dict[str, Any], store: CheckpointStore) -> list[Tile]:
    completed = store.completed_tile_ids()
    maximum = int(config["tiling"].get("max_tiles_per_run", 6))
    pending: list[Tile] = []
    for tile in tiles_from_config(config):
        if tile.tile_id not in completed:
            pending.append(tile)
        if len(pending) >= maximum:
            break
    return pending


def run_extract(config: dict[str, Any], client: MapillaryClient | None = None) -> dict[str, Any]:
    store = CheckpointStore(config["paths"]["catalog_path"])
    extract = config["extract"]
    if client is None and os.getenv("GALLERY_EXTRACT_MODE", "live").lower() == "fixture":
        from src.extract.fixture_client import FixtureMapillaryClient

        client = FixtureMapillaryClient()
        LOGGER.warning("Fixture extraction mode is active; no Mapillary API requests will be made")
    client = client or MapillaryClient(
        page_size=int(extract.get("page_size", 100)),
        request_timeout=float(extract.get("request_timeout_seconds", 30)),
        min_request_interval=float(extract.get("min_request_interval_seconds", 0.25)),
        max_attempts=int(extract.get("max_http_attempts", 6)),
        backoff_base=float(extract.get("backoff_base_seconds", 1.0)),
    )
    selected = select_pending_tiles(config, store)
    completed: list[str] = []
    total_new = 0
    for tile in selected:
        found = 0
        new_count = 0
        store.mark_tile(tile.tile_id, "in_progress")
        LOGGER.info("Extracting tile=%s bbox=%s", tile.tile_id, tile.bbox)
        try:
            for item in client.iter_images(tile, int(extract.get("max_images_per_tile", 500))):
                found += 1
                raw_path = Path(config["paths"]["raw_dir"]) / tile.tile_id / f"{item['image_id']}.jpg"
                if store.image_exists(item["image_id"]):
                    continue
                if not raw_path.exists():
                    client.download_image(item["download_url"], raw_path)
                inserted = store.add_image(
                    {
                        **item,
                        "source": getattr(client, "source", "mapillary"),
                        "tile_id": tile.tile_id,
                        "raw_file_path": str(raw_path.resolve()),
                        "download_timestamp": utc_now(),
                    }
                )
                new_count += int(inserted)
            store.mark_tile(tile.tile_id, "completed", found)
            completed.append(tile.tile_id)
            total_new += new_count
            LOGGER.info("Completed tile=%s images_found=%d new_images=%d", tile.tile_id, found, new_count)
        except Exception as exc:
            store.mark_tile(tile.tile_id, "failed", found, str(exc)[:1000])
            LOGGER.exception("Extraction failed for tile=%s after %d images", tile.tile_id, found)
            raise
    return {"tile_ids": completed, "tiles_completed": len(completed), "new_images": total_new}
