"""Build in-coverage calibration probes for a given gallery.

The ensemble's trust threshold needs a positive class: images the gallery's
regions genuinely contain but that are not gallery entries themselves. A gallery
member is useless for this -- retrieval returns the query image at similarity
1.0 and 0.00 km error, which measures memorisation, not geolocation.

The only such images available are the ones the pipeline's own filters dropped,
split by why they were dropped:

  Probe A -- quality-excluded (blur / indoor / tile cap). Genuinely different
      photographs of covered places. Unbiased with respect to visual similarity,
      so this is what a threshold should be calibrated on.

  Probe B -- perceptual-hash near-duplicates of a gallery image. Effectively the
      same frame, so retrieval scores them near-perfectly. An upper bound, kept
      separate and never used to pick a threshold.

Generalised over --sources so probes can be rebuilt for a new gallery footprint
rather than reusing ones tied to the old coverage.
"""

from __future__ import annotations

import argparse
import os
import sqlite3

import pandas as pd

DATA_ROOT = r"C:\View-Finder\gallery-pipeline\data"


def wsl_to_windows(path: str) -> str:
    if isinstance(path, str) and path.startswith("/mnt/") and len(path) > 6 and path[6] == "/":
        return f"{path[5].upper()}:\\" + path[7:].replace("/", "\\")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-metadata", required=True,
                        help="metadata.parquet of the gallery being calibrated")
    parser.add_argument("--sources", nargs="+", required=True,
                        help="dataset names under data/ whose catalogs to mine")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--out-prefix", required=True,
                        help="output path prefix, e.g. .../probes_v2")
    args = parser.parse_args()

    gallery = pd.read_parquet(args.gallery_metadata)
    gallery_ids = set(gallery["image_id"].astype(str))
    print(f"gallery: {len(gallery)} images ({len(gallery_ids)} unique ids)")

    frames = []
    for name in args.sources:
        catalog = os.path.join(args.data_root, name, "catalog.db")
        if not os.path.exists(catalog):
            print(f"  (no catalog for {name}, skipping)")
            continue
        with sqlite3.connect(catalog) as conn:
            frame = pd.read_sql_query(
                "SELECT image_id, lat, lon, tile_id, raw_file_path, "
                "validation_reason, transform_status FROM images",
                conn,
            )
        frame["image_id"] = frame["image_id"].astype(str)
        frame["source"] = name
        extras = frame[~frame["image_id"].isin(gallery_ids)]
        print(f"  {name}: {len(frame)} catalog rows, {len(extras)} not in gallery")
        frames.append(extras)

    if not frames:
        raise SystemExit("No source catalogs found")

    extras = pd.concat(frames, ignore_index=True)
    extras["image_path"] = extras["raw_file_path"].map(wsl_to_windows)
    extras = extras[extras["image_path"].map(os.path.exists)]
    extras = extras[extras["lat"].between(-90, 90) & extras["lon"].between(-180, 180)]
    extras = extras[~(extras["lat"].eq(0) & extras["lon"].eq(0))]

    reason = extras["validation_reason"].fillna("")
    is_duplicate = reason.str.startswith("near_duplicate_of")

    probe_a = extras[~is_duplicate].reset_index(drop=True)
    probe_b = extras[is_duplicate].reset_index(drop=True)

    print(f"\nProbe A quality-excluded (honest)   : {len(probe_a)}")
    if len(probe_a):
        print(f"   reasons: {dict(probe_a['transform_status'].value_counts())}")
        print(f"   sources: {dict(probe_a['source'].value_counts())}")
    print(f"Probe B near-duplicate (upper bound): {len(probe_b)}")

    columns = ["image_path", "lat", "lon", "image_id", "tile_id", "source", "transform_status"]
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    for label, frame in (("probe_a_quality_excluded", probe_a), ("probe_b_near_duplicate", probe_b)):
        if frame.empty:
            print(f"  (skipping empty {label})")
            continue
        path = f"{args.out_prefix}_{label}.csv"
        frame[columns].to_csv(path, index=False)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
