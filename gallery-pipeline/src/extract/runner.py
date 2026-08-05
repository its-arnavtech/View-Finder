"""Extraction stage orchestration independent of Airflow."""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from itertools import zip_longest
from pathlib import Path
from typing import Any

import requests

from src.extract.checkpoint_store import CheckpointStore, utc_now
from src.extract.mapillary_client import MapillaryClient
from src.extract.tiling import Tile, subdivide_tile, tiles_from_config


LOGGER = logging.getLogger(__name__)

# Two distinct HTTP 500 rejections are both fixed by splitting the tile into
# smaller cells and retrying:
#   * too many images in the bbox ("...reduce the amount of data..."), and
#   * a bbox exceeding Mapillary's maximum queryable area of 0.01 sq degrees
#     ("Bounding box area is too large..."). A 0.1x0.1 cell sits exactly on this
#     limit, so floating-point rounding pushes some just over it.
# The status alone is ambiguous (real outages are also 500), so the body decides
# whether to subdivide or fail.
SUBDIVIDABLE_REJECTION_MARKERS = (
    "reduce the amount of data",
    "bounding box area is too large",
)


def _is_subdividable_rejection(exc: Exception) -> bool:
    """True when the API refused a bbox that a smaller bbox would satisfy."""
    if not isinstance(exc, requests.HTTPError):
        return False
    response = getattr(exc, "response", None)
    if response is None or response.status_code != 500:
        return False
    message = str(exc).lower()
    return any(marker in message for marker in SUBDIVIDABLE_REJECTION_MARKERS)


def select_pending_tiles(config: dict[str, Any], store: CheckpointStore) -> list[Tile]:
    completed = store.completed_tile_ids()
    maximum = int(config["tiling"].get("max_tiles_per_run", 6))
    candidates = [tile for tile in tiles_from_config(config) if tile.tile_id not in completed]

    if config["tiling"].get("interleave_regions", False):
        # Take tiles round-robin across regions so a run bounded by max_tiles_per_run
        # samples every configured region. In config order, a region listed first can
        # otherwise consume the entire budget and later regions are never reached.
        by_region: dict[str, list[Tile]] = {}
        for tile in candidates:
            by_region.setdefault(tile.region, []).append(tile)
        interleaved: list[Tile] = []
        for group in zip_longest(*by_region.values()):
            interleaved.extend(tile for tile in group if tile is not None)
        candidates = interleaved

    return candidates[:maximum]


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
    failed: list[str] = []
    total_new = 0
    # An isolated tile failure is survivable; a streak of them is not. Above this
    # many back-to-back failures the run stops, on the assumption that the fault
    # is systemic (revoked token, network down, API outage) rather than one bbox.
    max_consecutive_failures = int(extract.get("max_consecutive_tile_failures", 8))
    consecutive_failures = 0
    aborted_early = False
    # Breadth pulls run a shallow subdivision depth on purpose, so a metro too
    # dense to divide further is an expected skip rather than a misconfiguration.
    # Default False keeps gallery builds fail-fast.
    skip_unsubdividable = bool(extract.get("skip_unsubdividable_tiles", False))
    cap = int(extract.get("max_images_per_tile", 500))
    max_depth = int(config["tiling"].get("max_subdivision_depth", 3))
    subdivision_factor = int(config["tiling"].get("subdivision_factor", 2))
    subdivisions: list[dict[str, Any]] = []

    # Optional wall-clock budget for a single invocation. A global scrape is far too
    # large for one run, so this lets each session stop cleanly after a bounded time;
    # unstarted tiles simply stay pending and the next session resumes past what the
    # checkpoint store has already completed. 0 / unset means no time limit.
    max_runtime = float(extract.get("max_runtime_seconds", 0) or 0)
    started_at = time.monotonic()
    stopped_early = False

    # A queue rather than a plain loop: a tile the API rejects as too dense is replaced
    # by its children, which are processed in this same run.
    queue: deque[tuple[Tile, int]] = deque((tile, 0) for tile in selected)
    while queue:
        if max_runtime > 0 and time.monotonic() - started_at >= max_runtime:
            stopped_early = True
            LOGGER.info(
                "Time budget of %.0fs reached; stopping with %d tile(s) still queued",
                max_runtime, len(queue),
            )
            break
        tile, depth = queue.popleft()
        found = 0
        new_count = 0
        # The cap is cumulative across runs, not per-run. Mapillary's paging is not
        # stably ordered, so a tile that was interrupted and resumed re-queries from
        # page 1 and can return a different sample; counting only this run's images
        # would let the tile grow past the cap on every resume.
        stored = store.image_count_for_tile(tile.tile_id)
        if stored >= cap:
            # Already satisfied by earlier runs - complete it without any API call.
            store.mark_tile(tile.tile_id, "completed", stored)
            completed.append(tile.tile_id)
            LOGGER.info("Tile=%s already at cap (%d/%d); skipping extraction", tile.tile_id, stored, cap)
            continue

        store.mark_tile(tile.tile_id, "in_progress")
        LOGGER.info("Extracting tile=%s bbox=%s stored=%d cap=%d", tile.tile_id, tile.bbox, stored, cap)
        try:
            # Worst case the API replays every already-stored image before yielding a
            # new one, so `cap` yields is exactly the budget needed to reach the cap,
            # and it also bounds paging on a tile whose pages are mostly duplicates.
            for item in client.iter_images(tile, cap):
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
                if stored + new_count >= cap:
                    # Stop consuming the generator: no further pages are requested.
                    LOGGER.info("Tile=%s reached cap %d; stopping extraction early", tile.tile_id, cap)
                    break
            total_stored = stored + new_count
            store.mark_tile(tile.tile_id, "completed", total_stored)
            completed.append(tile.tile_id)
            total_new += new_count
            LOGGER.info(
                "Completed tile=%s images_seen=%d new_images=%d total_stored=%d cap=%d",
                tile.tile_id, found, new_count, total_stored, cap,
            )
        except Exception as exc:
            total_new += new_count  # partial work is already committed and durable
            if _is_subdividable_rejection(exc) and depth < max_depth:
                children = subdivide_tile(tile, subdivision_factor)
                if children:
                    # Not a failure: the bbox is too large or too dense for the API.
                    # Retry it as smaller cells in this same run.
                    store.mark_tile(
                        tile.tile_id, "failed", found,
                        f"subdivided into {len(children)} cells (rejected at depth {depth})",
                    )
                    queue.extendleft((child, depth + 1) for child in reversed(children))
                    subdivisions.append(
                        {
                            "tile_id": tile.tile_id,
                            "bbox": list(tile.bbox),
                            "depth": depth,
                            "children": len(children),
                            "child_cell_size": round(
                                (children[0].max_lon - children[0].min_lon), 6
                            ),
                        }
                    )
                    LOGGER.warning(
                        "Tile=%s rejected at depth %d; subdivided into %d cells of ~%.4f deg",
                        tile.tile_id, depth, len(children),
                        children[0].max_lon - children[0].min_lon,
                    )
                    continue
            store.mark_tile(tile.tile_id, "failed", found, str(exc)[:1000])
            LOGGER.exception("Extraction failed for tile=%s after %d images", tile.tile_id, found)

            if _is_subdividable_rejection(exc) and not skip_unsubdividable:
                # Reaching here means the bbox is too dense AND subdivision depth
                # is exhausted. For a gallery build that is a configuration
                # problem -- the density bands or max_subdivision_depth cannot
                # represent this area -- and continuing would silently drop
                # coverage, so this stays fail-fast by default, as it always has.
                #
                # Breadth pulls invert that trade-off: they want many distinct
                # places, not exhaustive coverage of one, and deliberately run a
                # shallow max_subdivision_depth. There an undividable metro is a
                # tile to skip, not a misconfiguration, so they set
                # extract.skip_unsubdividable_tiles to keep the run going.
                raise

            # Anything else is a transient fault (connection reset, timeout, an
            # unexpected status) and must not discard the rest of the run. Every
            # completed tile is already durable in the checkpoint store, so
            # aborting threw away hours of good work over one bad bbox -- exactly
            # what a connection reset did to a 500-tile pull. Failed tiles stay
            # marked failed and are retried by a later run, so nothing is lost.
            consecutive_failures += 1
            failed.append(tile.tile_id)
            if consecutive_failures >= max_consecutive_failures:
                # A run of back-to-back failures is a systemic problem (token
                # revoked, network down, API outage) rather than a bad tile, and
                # continuing would just burn the remaining budget failing.
                LOGGER.error(
                    "Aborting: %d consecutive tile failures suggests a systemic fault, not a bad tile",
                    consecutive_failures,
                )
                aborted_early = True
                break
            continue
        else:
            consecutive_failures = 0
    return {
        "tile_ids": completed,
        "tiles_completed": len(completed),
        "failed_tile_ids": failed,
        "tiles_failed": len(failed),
        "new_images": total_new,
        "subdivisions": subdivisions,
        "stopped_early": stopped_early,
        "aborted_early": aborted_early,
    }
