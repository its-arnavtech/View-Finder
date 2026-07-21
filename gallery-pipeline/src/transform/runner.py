"""Transformation stage orchestration independent of Airflow."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.extract.checkpoint_store import CheckpointStore
from src.transform.balancing import cap_tile
from src.transform.embedding_generator import EmbeddingGenerator
from src.transform.quality_filters import passes_blur_filter


LOGGER = logging.getLogger(__name__)


def _write_tile_parquet(path: Path, records: list[dict[str, Any]]) -> int:
    dimension = len(records[0]["embedding_vector"])
    table = pa.table(
        {
            "image_id": pa.array([item["image_id"] for item in records], pa.string()),
            "source": pa.array([item["source"] for item in records], pa.string()),
            "lat": pa.array([item["lat"] for item in records], pa.float64()),
            "lon": pa.array([item["lon"] for item in records], pa.float64()),
            "tile_id": pa.array([item["tile_id"] for item in records], pa.string()),
            "raw_file_path": pa.array([item["raw_file_path"] for item in records], pa.string()),
            "blur_score": pa.array([item["blur_score"] for item in records], pa.float32()),
            "indoor_probability": pa.array([item["indoor_probability"] for item in records], pa.float32()),
            "embedding_vector": pa.array(
                [item["embedding_vector"] for item in records], pa.list_(pa.float32(), dimension)
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)
    return dimension


def run_transform(
    config: dict[str, Any],
    tile_ids: list[str] | None = None,
    generator: EmbeddingGenerator | None = None,
) -> dict[str, Any]:
    store = CheckpointStore(config["paths"]["catalog_path"])
    targets = tile_ids or store.tile_ids("completed")
    settings = config["transform"]
    model_name = f"open_clip:{settings['embedding_model']}:{settings['embedding_pretrained']}"
    total_retained = 0
    total_filtered = 0

    for tile_id in targets:
        store.reset_transform_status_for_tile(tile_id)
        reasons: Counter[str] = Counter()
        candidates: list[dict[str, Any]] = []
        for record in store.images_for_tile(tile_id, "passed"):
            passed, blur_score = passes_blur_filter(
                record["raw_file_path"], float(settings.get("blur_variance_threshold", 75.0))
            )
            record["blur_score"] = blur_score
            if not passed:
                store.update_transform(record["image_id"], "failed_blur", blur_score, None)
                reasons["blur"] += 1
                continue
            candidates.append(record)

        if not candidates:
            store.upsert_transformed_asset(tile_id, "", 0, 0, model_name)
            store.record_stage_result(tile_id, "transform", 0, sum(reasons.values()), dict(reasons))
            total_filtered += sum(reasons.values())
            LOGGER.info("Transformed tile=%s retained=0 reasons=%s", tile_id, dict(reasons))
            continue

        if generator is None:
            generator = EmbeddingGenerator(
                model_name=settings["embedding_model"],
                pretrained=settings["embedding_pretrained"],
                batch_size_gpu=int(settings.get("embedding_batch_size_gpu", 64)),
                batch_size_cpu=int(settings.get("embedding_batch_size_cpu", 16)),
            )
            model_name = generator.full_model_name
        embeddings, indoor_probabilities = generator.generate(
            [item["raw_file_path"] for item in candidates]
        )
        if embeddings.ndim != 2 or len(embeddings) != len(candidates):
            raise ValueError("Embedding generator returned an unexpected shape")

        outdoor_records: list[dict[str, Any]] = []
        exclude_indoor = bool(settings.get("exclude_likely_indoor", True))
        indoor_threshold = float(getattr(generator, "indoor_probability_threshold", 0.65))
        for record, vector, indoor_probability in zip(candidates, embeddings, indoor_probabilities):
            likely_indoor = bool(indoor_probability >= indoor_threshold)
            record["indoor_probability"] = float(indoor_probability)
            record["embedding_vector"] = np.asarray(vector, dtype=np.float32).tolist()
            if exclude_indoor and likely_indoor:
                store.update_transform(record["image_id"], "failed_indoor", record["blur_score"], True)
                reasons["likely_indoor"] += 1
                continue
            record["likely_indoor"] = likely_indoor
            outdoor_records.append(record)

        retained, capped = cap_tile(
            outdoor_records, int(settings.get("max_images_per_tile", 200))
        )
        for record in capped:
            store.update_transform(record["image_id"], "excluded_tile_cap", record["blur_score"], record["likely_indoor"])
        reasons["geographic_cap"] += len(capped)
        for record in retained:
            store.update_transform(record["image_id"], "retained", record["blur_score"], record["likely_indoor"])

        if retained:
            parquet_path = Path(config["paths"]["transformed_dir"]) / f"{tile_id}.parquet"
            dimension = _write_tile_parquet(parquet_path, retained)
            store.upsert_transformed_asset(
                tile_id, str(parquet_path.resolve()), len(retained), dimension, model_name
            )
        else:
            store.upsert_transformed_asset(tile_id, "", 0, 0, model_name)
        filtered = sum(reasons.values())
        store.record_stage_result(tile_id, "transform", len(retained), filtered, dict(reasons))
        total_retained += len(retained)
        total_filtered += filtered
        LOGGER.info("Transformed tile=%s retained=%d filtered=%d reasons=%s", tile_id, len(retained), filtered, dict(reasons))
    return {"tile_ids": targets, "retained": total_retained, "filtered": total_filtered, "model_name": model_name}

