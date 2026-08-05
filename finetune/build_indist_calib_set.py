"""Build an in-distribution calibration set that the gallery has NOT memorised.

The trap: benchmark-gallery/v1 is merged from the same scraped datasets that the
fine-tuning pool draws on, so a held-out slice taken straight from that pool
contains images which are themselves gallery entries. Retrieval then returns the
query image, scoring top_similarity ~1.0 at 0.00 km error. Calibrating on that
measures gallery membership, not geolocation, and would justify any threshold.

What we actually want is the honest local-coverage regime: a *different* photo of
a place the gallery covers. So we keep images that are

  1. NOT present in the gallery's own metadata (by image_id), and
  2. within `--coverage-km` of at least one gallery coordinate.

Condition 2 is what makes this "in-distribution" — inside the gallery's real
footprint, where retrieval is supposed to earn its keep.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from dataset_index import build_index, build_index_from_catalog
from location_sampling import haversine_km

GALLERY_METADATA = r"C:\View-Finder\gallery-pipeline\data\benchmark-gallery\gallery\v1\metadata.parquet"
BREADTH_CATALOG = r"C:\View-Finder\gallery-pipeline\data\finetune-breadth\catalog.db"


def nearest_gallery_km(coords: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    mins = []
    for start in range(0, len(coords), 256):
        block = coords[start : start + 256]
        d = haversine_km(
            block[:, 0][:, None], block[:, 1][:, None], gallery[:, 0][None, :], gallery[:, 1][None, :]
        )
        mins.append(d.min(axis=1))
    return np.concatenate(mins)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-metadata", default=GALLERY_METADATA)
    parser.add_argument("--coverage-km", type=float, default=5.0,
                        help="max distance to a gallery coordinate to count as covered")
    parser.add_argument("--out", default=r"C:\View-Finder\finetune\work\indist_calib_set.csv")
    args = parser.parse_args()

    gallery = pd.read_parquet(args.gallery_metadata)
    gallery_ids = set(gallery["image_id"].astype(str))
    gallery_coords = gallery[["lat", "lon"]].to_numpy(dtype=np.float64)
    print(f"gallery: {len(gallery)} images, {len(gallery_ids)} unique ids")

    pool, _ = build_index()
    breadth, _ = build_index_from_catalog(BREADTH_CATALOG, dataset_name="finetune-breadth")
    combined = pd.concat([pool, breadth], ignore_index=True).drop_duplicates(subset=["image_id"])
    combined["image_id"] = combined["image_id"].astype(str)
    print(f"combined pool: {len(combined)} images")

    in_gallery = combined["image_id"].isin(gallery_ids)
    print(f"  of which ARE gallery entries : {int(in_gallery.sum())}  <- excluded (self-match)")
    unseen = combined[~in_gallery].reset_index(drop=True)
    print(f"  candidates not in gallery    : {len(unseen)}")

    dist = nearest_gallery_km(unseen[["lat", "lon"]].to_numpy(dtype=np.float64), gallery_coords)
    unseen["nearest_gallery_km"] = dist
    covered = unseen[unseen["nearest_gallery_km"] <= args.coverage_km].reset_index(drop=True)
    print(f"  within {args.coverage_km:.0f} km of gallery coverage : {len(covered)}  <- in-distribution set")

    if covered.empty:
        raise SystemExit("No unseen images fall inside gallery coverage; cannot build the set.")

    print(f"\ndistance to nearest gallery point (km): median "
          f"{covered['nearest_gallery_km'].median():.3f}, max {covered['nearest_gallery_km'].max():.3f}")
    print(f"\nper dataset:\n{covered['dataset'].value_counts().to_string()}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    covered[["image_path", "lat", "lon", "image_id", "dataset", "nearest_gallery_km"]].to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(covered)} images)")


if __name__ == "__main__":
    main()
