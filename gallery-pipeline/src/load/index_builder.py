"""Build an immutable, versioned FAISS gallery from transformed Parquet assets."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.extract.checkpoint_store import CheckpointStore
from src.load.versioning import allocate_version


LOGGER = logging.getLogger(__name__)


def classify_region(lat: float, lon: float) -> str:
    """Return a coarse continent-scale region from coordinates.

    This intentionally uses transparent bounding heuristics rather than network reverse
    geocoding. It is suitable for coverage warnings, not geopolitical classification.
    """
    if -60 <= lat <= 15 and -90 <= lon <= -30:
        return "South America"
    if 7 <= lat <= 85 and -170 <= lon <= -50:
        return "North America"
    if -35 <= lat <= 37 and -20 <= lon <= 55:
        return "Africa"
    if 35 <= lat <= 72 and -25 <= lon <= 45:
        return "Europe"
    if -50 <= lat <= 0 and 110 <= lon <= 180:
        return "Oceania"
    if 0 <= lat <= 80 and 25 <= lon <= 180:
        return "Asia"
    return "Other"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def run_load(config: dict[str, Any]) -> dict[str, Any]:
    store = CheckpointStore(config["paths"]["catalog_path"])
    assets = store.transformed_assets()
    if not assets:
        raise ValueError("No transformed tile assets are available; refusing to create an empty version")
    tables = [pq.read_table(asset["parquet_path"]) for asset in assets]
    table = pa.concat_tables(tables, promote_options="default")
    vectors = np.asarray(table["embedding_vector"].to_pylist(), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Transformed assets contain no valid embedding matrix")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or infinite values")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    version, version_dir = allocate_version(config["paths"]["gallery_dir"])
    index_tmp = version_dir / "gallery.faiss.tmp"
    index_path = version_dir / "gallery.faiss"
    faiss.write_index(index, str(index_tmp))
    index_tmp.replace(index_path)

    image_ids = table["image_id"].to_pylist()
    latitudes = table["lat"].to_pylist()
    longitudes = table["lon"].to_pylist()
    tile_ids = table["tile_id"].to_pylist()
    regions = [classify_region(float(lat), float(lon)) for lat, lon in zip(latitudes, longitudes)]
    metadata = pa.table(
        {
            "index_position": pa.array(range(len(image_ids)), pa.int64()),
            "image_id": pa.array(image_ids, pa.string()),
            "lat": pa.array(latitudes, pa.float64()),
            "lon": pa.array(longitudes, pa.float64()),
            "tile_id": pa.array(tile_ids, pa.string()),
            "region": pa.array(regions, pa.string()),
            "raw_file_path": table["raw_file_path"],
        }
    )
    metadata_tmp = version_dir / "metadata.parquet.tmp"
    metadata_path = version_dir / "metadata.parquet"
    pq.write_table(metadata, metadata_tmp, compression="zstd")
    metadata_tmp.replace(metadata_path)

    model_names = sorted({asset["model_name"] for asset in assets})
    manifest = {
        "version": version,
        "status": "built_pending_quality_check",
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_image_count": len(image_ids),
        "embedding_dimension": int(vectors.shape[1]),
        "embedding_model": model_names[0] if len(model_names) == 1 else model_names,
        "tiles_covered": len(set(tile_ids)),
        "tile_ids": sorted(set(tile_ids)),
        "geographic_coverage": dict(sorted(Counter(regions).items())),
        "files": {"index": "gallery.faiss", "metadata": "metadata.parquet"},
    }
    _atomic_json(version_dir / "manifest.json", manifest)
    LOGGER.info("Built gallery version=v%d images=%d dimension=%d tiles=%d", version, len(image_ids), vectors.shape[1], len(set(tile_ids)))
    return {"version": version, "version_dir": str(version_dir.resolve()), "total_images": len(image_ids)}

