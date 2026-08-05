"""Build in-coverage probe sets for calibrating the retrieval trust threshold.

There is no clean "unseen but covered" data available. Every image inside
benchmark-gallery/v1's footprint is either a gallery entry itself or was dropped
by the pipeline's own filters; the nearest non-gallery image in the whole pool
sits 36.9 km from any gallery point. So the in-coverage regime has to be probed
with what the source catalogs kept, split by *why* it was excluded:

  Probe A -- "quality-excluded" (n~24): dropped for blur / indoor / tile cap.
      Genuinely different photographs of covered places. Small, but unbiased with
      respect to visual similarity, so this is the honest probe.

  Probe B -- "near-duplicate" (n~282): dropped by perceptual-hash dedup as
      near-duplicates of a gallery image. These are effectively the same frame,
      so retrieval will score them near-perfectly. That is an UPPER BOUND on
      retrieval's in-coverage behaviour, not a realistic estimate, and it is kept
      separate for exactly that reason.

Gallery members themselves are never included: retrieval would return the query
image and score 1.0 at 0 km, which measures memorisation and nothing else.
"""

from __future__ import annotations

import argparse
import os
import sqlite3

import pandas as pd

GALLERY_METADATA = r"C:\View-Finder\gallery-pipeline\data\benchmark-gallery\gallery\v1\metadata.parquet"
SOURCES = ("target-regions", "global-test", "live-smoke")


def wsl_to_windows(path: str) -> str:
    if isinstance(path, str) and path.startswith("/mnt/") and len(path) > 6 and path[6] == "/":
        return f"{path[5].upper()}:\\" + path[7:].replace("/", "\\")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-metadata", default=GALLERY_METADATA)
    parser.add_argument("--out-dir", default=r"C:\View-Finder\finetune\work")
    args = parser.parse_args()

    gallery = pd.read_parquet(args.gallery_metadata)
    gallery_ids = set(gallery["image_id"].astype(str))

    frames = []
    for name in SOURCES:
        conn = sqlite3.connect(rf"C:\View-Finder\gallery-pipeline\data\{name}\catalog.db")
        frame = pd.read_sql_query(
            "SELECT image_id, lat, lon, tile_id, raw_file_path, validation_reason, transform_status FROM images",
            conn,
        )
        frame["image_id"] = frame["image_id"].astype(str)
        frame["source"] = name
        frames.append(frame[~frame["image_id"].isin(gallery_ids)])

    extras = pd.concat(frames, ignore_index=True)
    extras["image_path"] = extras["raw_file_path"].map(wsl_to_windows)
    extras = extras[extras["image_path"].map(os.path.exists)]
    extras = extras[extras["lat"].between(-90, 90) & extras["lon"].between(-180, 180)]

    reason = extras["validation_reason"].fillna("")
    is_dup = reason.str.startswith("near_duplicate_of")

    probe_a = extras[~is_dup].reset_index(drop=True)
    probe_b = extras[is_dup].reset_index(drop=True)

    print(f"extras not in gallery : {len(extras)}")
    print(f"  Probe A quality-excluded (honest)   : {len(probe_a)}")
    if len(probe_a):
        print(f"     reasons: {dict(probe_a['transform_status'].value_counts())}")
    print(f"  Probe B near-duplicate (upper bound): {len(probe_b)}")

    columns = ["image_path", "lat", "lon", "image_id", "tile_id", "source", "transform_status"]
    os.makedirs(args.out_dir, exist_ok=True)
    for label, frame in (("probe_a_quality_excluded", probe_a), ("probe_b_near_duplicate", probe_b)):
        if frame.empty:
            print(f"  (skipping empty {label})")
            continue
        path = os.path.join(args.out_dir, f"incoverage_{label}.csv")
        frame[columns].to_csv(path, index=False)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
