"""Report which parts of the world the scraped pool already covers.

Location diversity, not image count, is what has been shown to matter, so a new
pull should target places that are absent rather than densify places that are
already represented. This groups every stored image by country/continent-scale
bucket so the gaps are obvious before the next pull is configured.
"""

from __future__ import annotations

import argparse
import glob
import os
import sqlite3

import pandas as pd

DATA_ROOT = r"C:\View-Finder\gallery-pipeline\data"


def load_all_catalogs(data_root: str = DATA_ROOT) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(data_root, "*", "catalog.db"))):
        dataset = os.path.basename(os.path.dirname(path))
        try:
            with sqlite3.connect(path) as conn:
                frame = pd.read_sql_query("SELECT image_id, lat, lon, tile_id FROM images", conn)
        except Exception as exc:  # noqa: BLE001 - a partial catalog should not stop the audit
            print(f"  (skipping {dataset}: {exc})")
            continue
        frame["dataset"] = dataset
        frames.append(frame)

    if not frames:
        raise SystemExit(f"No catalogs found under {data_root}")
    combined = pd.concat(frames, ignore_index=True)
    combined["image_id"] = combined["image_id"].astype(str)
    return combined.drop_duplicates(subset=["image_id"]).reset_index(drop=True)


def coarse_region(lat: float, lon: float) -> str:
    """Continent-scale bucket. Deliberately crude -- it only has to expose gaps."""
    if lat >= 35 and -25 <= lon <= 45:
        return "Europe"
    if lat >= 15 and 45 < lon <= 180:
        return "Asia (N/E)"
    if -12 <= lat < 15 and 60 < lon <= 150:
        return "SE Asia / S Asia"
    if lat < -12 and 90 < lon <= 180:
        return "Oceania"
    if lat >= 15 and -170 <= lon < -50:
        return "North America"
    if -60 <= lat < 15 and -95 <= lon < -30:
        return "South America"
    if -40 <= lat < 35 and -20 <= lon <= 55:
        return "Africa"
    if lat >= 15 and -50 <= lon < -25:
        return "N Atlantic"
    return "Other / polar"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DATA_ROOT)
    args = parser.parse_args()

    pool = load_all_catalogs(args.data_root)
    pool["region"] = [coarse_region(a, b) for a, b in zip(pool["lat"], pool["lon"])]

    print(f"total unique images across all catalogs: {len(pool)}\n")
    print("--- images per dataset ---")
    print(pool["dataset"].value_counts().to_string())

    print("\n--- coarse region coverage ---")
    summary = pool.groupby("region").agg(
        images=("image_id", "size"),
        tiles=("tile_id", "nunique"),
        lat_min=("lat", "min"),
        lat_max=("lat", "max"),
        lon_min=("lon", "min"),
        lon_max=("lon", "max"),
    ).sort_values("images", ascending=False)
    print(summary.to_string())

    print("\n--- 10-degree cells occupied (a proxy for spread) ---")
    pool["cell"] = (
        (pool["lat"] // 10 * 10).astype(int).astype(str)
        + ","
        + (pool["lon"] // 10 * 10).astype(int).astype(str)
    )
    print(f"distinct 10-deg cells: {pool['cell'].nunique()}")
    print("\ntop occupied cells (lat,lon of cell corner):")
    print(pool["cell"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
