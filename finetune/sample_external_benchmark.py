"""Sample a globally diverse validation slice from the on-disk benchmark data.

benchmark-data/manifest.csv indexes 14,002 geotagged images from two Kaggle
datasets (google-street-view, 4k-benchmark-images) spanning every longitude band.
None of it has ever been used for training here, so it serves the same role
Track B wants from GSV-Cities: an external check that fine-tuning gains
generalize beyond our own scraped held-out slices.

VALIDATION ONLY. Nothing sampled here may be fed back into training.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from location_sampling import measure_redundancy, thin_by_spacing

MANIFEST = r"C:\View-Finder\benchmark-data\manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--min-spacing-m", type=float, default=1000.0,
                        help="minimum spacing between sampled locations")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default=r"C:\View-Finder\finetune\work\external_benchmark_sample.csv")
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest).rename(columns={"latitude": "lat", "longitude": "lon"})
    frame = frame[frame["image_path"].map(os.path.exists)]
    print(f"manifest: {len(frame)} images with files present")

    # Guarantee spatial diversity rather than assuming it.
    thinned = thin_by_spacing(frame, min_spacing_m=args.min_spacing_m, seed=args.seed)
    print(f"thinned to >={args.min_spacing_m:.0f} m: {len(frame)} -> {len(thinned)}")

    sample = thinned.sample(n=min(args.total, len(thinned)), random_state=args.seed).reset_index(drop=True)

    report = measure_redundancy(sample)
    print()
    print(report.format("external benchmark sample"))
    print()
    print(f"source datasets:\n{sample['source_dataset'].value_counts().to_string()}")
    print()
    print(f"lat range: {sample['lat'].min():.2f} .. {sample['lat'].max():.2f}")
    print(f"lon range: {sample['lon'].min():.2f} .. {sample['lon'].max():.2f}")

    bins = [-180, -120, -60, 0, 60, 120, 180]
    print(f"\nlongitude spread:\n{pd.cut(sample['lon'], bins).value_counts().sort_index().to_string()}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sample[["image_path", "lat", "lon", "source_dataset"]].to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(sample)} images)")
    print("VALIDATION ONLY -- never use this sample for training.")


if __name__ == "__main__":
    main()
