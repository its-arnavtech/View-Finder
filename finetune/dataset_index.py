"""Build a consolidated (image, lat, lon) index from gallery-pipeline data.

The gallery pipeline writes one metadata.parquet per dataset version under
data/<dataset>/gallery/<version>/. Versions are cumulative supersets of each
other, and datasets overlap, so we dedupe on image_id and keep only rows whose
raw image is actually present on disk.

raw_file_path is recorded by the WSL2-side DAG as /mnt/c/... and has to be
translated back to a Windows path for the Windows-side torch venv.
"""

from __future__ import annotations

import glob
import os
import sqlite3
from dataclasses import dataclass

import pandas as pd

GALLERY_GLOB = r"C:\View-Finder\gallery-pipeline\data\*\gallery\*\metadata.parquet"

# Columns every gallery metadata.parquet is expected to carry.
REQUIRED_COLUMNS = ("image_id", "lat", "lon", "tile_id", "region", "raw_file_path")


def wsl_to_windows(path: str) -> str:
    """Translate a /mnt/<drive>/... WSL path into its Windows equivalent."""
    if not isinstance(path, str):
        return path
    if path.startswith("/mnt/") and len(path) > 6 and path[6] == "/":
        drive = path[5].upper()
        return f"{drive}:\\" + path[7:].replace("/", "\\")
    return path


@dataclass
class IndexStats:
    rows_scanned: int
    unique_ids: int
    present_on_disk: int
    distinct_tiles: int


def build_index(gallery_glob: str = GALLERY_GLOB) -> tuple[pd.DataFrame, IndexStats]:
    paths = sorted(glob.glob(gallery_glob))
    if not paths:
        raise FileNotFoundError(f"No gallery metadata found matching {gallery_glob}")

    frames = []
    for path in paths:
        parts = path.split(os.sep)
        frame = pd.read_parquet(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing expected columns: {missing}")
        frame["dataset"] = parts[-4]
        frame["gallery_version"] = parts[-2]
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    rows_scanned = len(combined)

    # Later gallery versions supersede earlier ones for the same image.
    combined = combined.sort_values(["dataset", "gallery_version"])
    combined = combined.drop_duplicates(subset=["image_id"], keep="last")
    unique_ids = len(combined)

    combined["image_path"] = combined["raw_file_path"].map(wsl_to_windows)
    combined = combined[combined["image_path"].map(os.path.exists)]

    # Guard against degenerate coordinates that would poison the contrastive loss.
    combined = combined[combined["lat"].between(-90, 90) & combined["lon"].between(-180, 180)]
    combined = combined[~(combined["lat"].eq(0) & combined["lon"].eq(0))]

    combined = combined.reset_index(drop=True)
    stats = IndexStats(
        rows_scanned=rows_scanned,
        unique_ids=unique_ids,
        present_on_disk=len(combined),
        distinct_tiles=combined["tile_id"].nunique(),
    )
    return combined[["image_id", "lat", "lon", "tile_id", "region", "dataset", "image_path"]], stats


def build_index_from_catalog(catalog_path: str, dataset_name: str | None = None) -> tuple[pd.DataFrame, IndexStats]:
    """Build an index straight from an extract-stage catalog.db.

    The breadth pull deliberately stops after `extract`, since fine-tuning needs
    raw images plus coordinates and not the FAISS gallery artifacts that
    validate/transform/load produce. Those stages never run, so there is no
    metadata.parquet to read -- the catalog is the source of truth.
    """
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"No catalog at {catalog_path}")

    with sqlite3.connect(catalog_path) as conn:
        frame = pd.read_sql_query(
            "SELECT image_id, lat, lon, tile_id, raw_file_path, validation_status FROM images",
            conn,
        )

    rows_scanned = len(frame)
    frame = frame.drop_duplicates(subset=["image_id"])
    unique_ids = len(frame)

    frame["image_path"] = frame["raw_file_path"].map(wsl_to_windows)
    frame = frame[frame["image_path"].map(os.path.exists)]
    frame = frame[frame["lat"].between(-90, 90) & frame["lon"].between(-180, 180)]
    frame = frame[~(frame["lat"].eq(0) & frame["lon"].eq(0))]

    frame["dataset"] = dataset_name or os.path.basename(os.path.dirname(catalog_path))
    # The breadth config tiles globally, so the coarse region label is derived
    # rather than configured.
    frame["region"] = "global"

    frame = frame.reset_index(drop=True)
    stats = IndexStats(
        rows_scanned=rows_scanned,
        unique_ids=unique_ids,
        present_on_disk=len(frame),
        distinct_tiles=frame["tile_id"].nunique(),
    )
    return frame[["image_id", "lat", "lon", "tile_id", "region", "dataset", "image_path"]], stats


if __name__ == "__main__":
    index, stats = build_index()
    print(f"rows scanned across all gallery versions : {stats.rows_scanned}")
    print(f"unique image_id                          : {stats.unique_ids}")
    print(f"present on disk with valid coords        : {stats.present_on_disk}")
    print(f"distinct tiles                           : {stats.distinct_tiles}")
    print()
    print("--- images per region ---")
    print(index["region"].value_counts().to_string())
    print()
    print("--- images per dataset ---")
    print(index["dataset"].value_counts().to_string())
    print()
    print("--- coordinate spread ---")
    print(index[["lat", "lon"]].describe().to_string())
    print()
    print("--- top tiles ---")
    print(index["tile_id"].value_counts().head(10).to_string())
