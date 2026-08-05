"""Sample a diverse GSV-Cities slice for use as an external benchmark.

VALIDATION ONLY. Nothing this produces may be used for training or fine-tuning.
GSV-Cities is licensed CC BY-NC-ND 4.0; its NoDerivatives term is why it is not
a training source here, and benchmarking does not create a derivative work.

Verified against the upstream source rather than dataset-card claims
(github.com/amaralibey/gsv-cities, dataloaders/train/GSVCitiesDataset.py):

  * 23 cities in the official TRAIN_CITIES list (secondhand sources say "40+").
  * Layout: ``Images/<city_id>/<filename>`` and ``Dataframes/<City>.csv``.
  * CSV columns: place_id, year, month, northdeg, city_id, lat, lon, panoid.
  * Filename: ``{city}_{place_id:07d}_{year:04d}_{month:02d}_{northdeg:03d}_``
    ``{lat}_{lon}_{panoid}.jpg`` -- so ground-truth coordinates are recoverable
    from the filename alone, without the CSVs.

Sampling takes at most one image per place and spreads the budget evenly across
cities, so the benchmark measures breadth rather than re-measuring one metro.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

TRAIN_CITIES = [
    "Bangkok", "BuenosAires", "LosAngeles", "MexicoCity", "OSL", "Rome",
    "Barcelona", "Chicago", "Madrid", "Miami", "Phoenix", "TRT", "Boston",
    "Lisbon", "Medellin", "Minneapolis", "PRG", "WashingtonDC", "Brussels",
    "London", "Melbourne", "Osaka", "PRS",
]

# city_placeid_year_month_northdeg_lat_lon_panoid.jpg
FILENAME_RE = re.compile(
    r"^(?P<city>[^_]+)_(?P<place_id>\d+)_(?P<year>\d+)_(?P<month>\d+)_"
    r"(?P<northdeg>\d+)_(?P<lat>-?\d+\.?\d*)_(?P<lon>-?\d+\.?\d*)_(?P<panoid>.+)\.jpg$",
    re.IGNORECASE,
)


def parse_filename(name: str) -> dict | None:
    match = FILENAME_RE.match(name)
    if not match:
        return None
    parts = match.groupdict()
    try:
        parts["lat"] = float(parts["lat"])
        parts["lon"] = float(parts["lon"])
        parts["place_id"] = int(parts["place_id"])
    except ValueError:
        return None
    return parts


def sample_from_images_dir(base_path: str, total: int, seed: int = 1337) -> pd.DataFrame:
    """Walk Images/<city>/ and sample <=1 image per place, spread across cities."""
    images_root = os.path.join(base_path, "Images")
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"No Images/ directory under {base_path}")

    cities = sorted(d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d)))
    if not cities:
        raise FileNotFoundError(f"No city folders under {images_root}")

    per_city = max(1, total // len(cities))
    rows: list[dict] = []
    unparsed = 0

    for city in cities:
        city_dir = os.path.join(images_root, city)
        seen_places: set[int] = set()
        city_rows: list[dict] = []
        for name in sorted(os.listdir(city_dir)):
            parsed = parse_filename(name)
            if parsed is None:
                unparsed += 1
                continue
            if parsed["place_id"] in seen_places:
                continue  # one image per place: maximise location diversity
            seen_places.add(parsed["place_id"])
            city_rows.append({
                "image_path": os.path.join(city_dir, name),
                "lat": parsed["lat"],
                "lon": parsed["lon"],
                "place_id": parsed["place_id"],
                "city": city,
            })
        frame = pd.DataFrame(city_rows)
        if len(frame) > per_city:
            frame = frame.sample(n=per_city, random_state=seed)
        rows.extend(frame.to_dict("records"))

    result = pd.DataFrame(rows)
    if len(result) > total:
        result = result.sample(n=total, random_state=seed)
    if unparsed:
        print(f"warning: {unparsed} filenames did not match the expected pattern")
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", required=True, help="gsv_cities root containing Images/")
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default="gsv_cities_sample.csv")
    args = parser.parse_args()

    frame = sample_from_images_dir(args.base_path, args.total, args.seed)
    frame.to_csv(args.out, index=False)

    print(f"sampled {len(frame)} images across {frame['city'].nunique()} cities "
          f"and {frame['place_id'].nunique()} distinct places")
    print(f"\nper-city counts:\n{frame['city'].value_counts().to_string()}")
    print(f"\nlat range: {frame['lat'].min():.3f} .. {frame['lat'].max():.3f}")
    print(f"lon range: {frame['lon'].min():.3f} .. {frame['lon'].max():.3f}")
    print(f"\nwrote {args.out}")
    print("\nVALIDATION ONLY -- do not use this sample for training (CC BY-NC-ND 4.0).")


if __name__ == "__main__":
    main()
