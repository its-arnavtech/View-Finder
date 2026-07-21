"""Validation stage orchestration independent of Airflow."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from src.extract.checkpoint_store import CheckpointStore
from src.validate.dedup import find_near_duplicate, perceptual_hash
from src.validate.image_validation import validate_image


LOGGER = logging.getLogger(__name__)


def run_validate(config: dict[str, Any], tile_ids: list[str] | None = None) -> dict[str, Any]:
    store = CheckpointStore(config["paths"]["catalog_path"])
    targets = tile_ids or store.tile_ids("completed")
    maximum_distance = int(config["validate"].get("perceptual_hash_distance", 6))
    total_passed = 0
    total_failed = 0
    for tile_id in targets:
        accepted = []
        reasons: Counter[str] = Counter()
        passed = 0
        failed = 0
        for record in store.images_for_tile(tile_id):
            image_id = record["image_id"]
            valid, reason = validate_image(record["raw_file_path"])
            if not valid:
                store.update_validation(image_id, "failed", reason, None)
                reasons[str(reason)] += 1
                failed += 1
                continue
            try:
                candidate_hash = perceptual_hash(record["raw_file_path"])
            except (OSError, ValueError):
                store.update_validation(image_id, "failed", "hash_failed", None)
                reasons["hash_failed"] += 1
                failed += 1
                continue
            duplicate_of = find_near_duplicate(candidate_hash, accepted, maximum_distance)
            if duplicate_of:
                reason = f"near_duplicate_of:{duplicate_of}"
                store.update_validation(image_id, "failed", reason, str(candidate_hash))
                reasons["near_duplicate"] += 1
                failed += 1
                continue
            accepted.append((image_id, candidate_hash))
            store.update_validation(image_id, "passed", None, str(candidate_hash))
            passed += 1
        store.record_stage_result(tile_id, "validate", passed, failed, dict(reasons))
        total_passed += passed
        total_failed += failed
        LOGGER.info("Validated tile=%s passed=%d failed=%d reasons=%s", tile_id, passed, failed, dict(reasons))
    return {"tile_ids": targets, "passed": total_passed, "failed": total_failed}

