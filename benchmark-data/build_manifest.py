"""Consolidate all benchmark image datasets into one unified manifest.

Phase 1 of the View Finder benchmarking system. Each source dataset stores its
ground-truth coordinates differently (CSV columns, filename-encoded, EXIF, ...),
so this script normalises them into a single table the benchmark harness can
consume without caring where any given image came from.

Output columns (CSV and Parquet):
    image_path        absolute path to the image on this machine
    latitude          float, WGS84, in [-90, 90]
    longitude         float, WGS84, in [-180, 180]
    source_dataset    provenance tag, e.g. "kaggle/4k-benchmark-images"

Currently wired sources:
  * kaggle/4k-benchmark-images  -- coords from balanced_4000_image_subset.csv
                                   (falls back to filename parsing image_{id}_{lat}_{lon}.jpg)
  * im2gps                      -- the local Im2GPS ~2k set; coords supplied via
                                   --im2gps-coords (a CSV keyed by image filename)

Adding a source later means writing one reader function that yields
ManifestRow(...) and registering it below -- nothing else changes.

Usage:
    python build_manifest.py \
        --benchmark-root .  \
        [--im2gps-images DIR --im2gps-coords COORDS.csv] \
        [--out manifest.csv]
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
from dataclasses import dataclass
from typing import Iterable, Iterator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("build_manifest")


@dataclass
class ManifestRow:
    image_path: str
    latitude: float
    longitude: float
    source_dataset: str


def _valid_coord(lat: float, lon: float) -> bool:
    """Reject placeholder/invalid coordinates (0,0 is treated as missing)."""
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return False
    if lat == 0.0 and lon == 0.0:
        return False
    return True


# --- Source: kaggle/4k-benchmark-images --------------------------------------

_4K_DIRNAME = os.path.join("kaggle", "4k-benchmark-images")
_4K_SOURCE = "kaggle/4k-benchmark-images"


def _parse_coords_from_filename(filename: str) -> tuple[float, float] | None:
    """image_{id}_{lat}_{lon}.jpg -> (lat, lon), or None if it doesn't match."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    if len(parts) < 4 or parts[0] != "image":
        return None
    try:
        return float(parts[-2]), float(parts[-1])
    except ValueError:
        return None


def read_4k_benchmark(benchmark_root: str) -> Iterator[ManifestRow]:
    """Yield rows for the 4K Global Geolocation Benchmark.

    Prefers the shipped CSV (image_name, lat, lon); resolves each image_name to
    its real path on disk and falls back to parsing coordinates from the filename
    when the CSV is absent or a row is malformed.
    """
    root = os.path.join(benchmark_root, _4K_DIRNAME)
    if not os.path.isdir(root):
        logger.warning("4k-benchmark not found at %s; skipping", root)
        return

    # Map basename -> absolute path once, so CSV rows resolve regardless of the
    # subfolder layout the archive unpacked into.
    by_name = {
        os.path.basename(p): os.path.abspath(p)
        for p in glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
    }

    csv_path = next(iter(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)), None)
    seen: set[str] = set()
    emitted = 0

    if csv_path:
        with open(csv_path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            cols = {c.lower(): c for c in (reader.fieldnames or [])}
            name_col = cols.get("image_name")
            lat_col = cols.get("lat") or cols.get("latitude")
            lon_col = cols.get("lon") or cols.get("longitude")
            if name_col and lat_col and lon_col:
                for row in reader:
                    name = os.path.basename(row[name_col])
                    path = by_name.get(name)
                    if not path:
                        continue
                    try:
                        lat, lon = float(row[lat_col]), float(row[lon_col])
                    except (TypeError, ValueError):
                        continue
                    if not _valid_coord(lat, lon):
                        continue
                    seen.add(name)
                    emitted += 1
                    yield ManifestRow(path, lat, lon, _4K_SOURCE)
            else:
                logger.warning("4k CSV %s lacks expected columns; using filenames", csv_path)

    # Any images not covered by the CSV: parse coordinates from their filename.
    for name, path in by_name.items():
        if name in seen:
            continue
        parsed = _parse_coords_from_filename(name)
        if parsed and _valid_coord(*parsed):
            emitted += 1
            yield ManifestRow(path, parsed[0], parsed[1], _4K_SOURCE)

    logger.info("4k-benchmark: %d images with valid coordinates", emitted)


# --- Source: kaggle/google-street-view ---------------------------------------

_GSV_DIRNAME = os.path.join("kaggle", "google-street-view")
_GSV_SOURCE = "kaggle/google-street-view"


def read_google_street_view(benchmark_root: str) -> Iterator[ManifestRow]:
    """Yield rows for the 10k Google Street View dataset.

    Coordinates live in ``coords.csv``. That file has no header in some releases
    and its rows correspond to images by *index* (``dataset/<n>.png``), so the
    reader handles both a proper header (lat/lon columns) and the headerless
    ``lat,lon`` form joined positionally to ``<row_number>.png``.
    """
    root = os.path.join(benchmark_root, _GSV_DIRNAME)
    if not os.path.isdir(root):
        logger.warning("google-street-view not found at %s; skipping", root)
        return

    by_name = {
        os.path.basename(p): os.path.abspath(p)
        for p in glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
    }
    csv_path = next(iter(glob.glob(os.path.join(root, "**", "coords.csv"), recursive=True)), None)
    if not csv_path:
        logger.warning("google-street-view: coords.csv not found; skipping")
        return

    emitted = 0
    with open(csv_path, encoding="utf-8", newline="") as handle:
        sample = handle.readline()
        handle.seek(0)
        has_header = any(c.isalpha() for c in sample)

        if has_header:
            reader = csv.DictReader(handle)
            cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
            lat_col = cols.get("lat") or cols.get("latitude")
            lon_col = cols.get("lon") or cols.get("lng") or cols.get("longitude")
            name_col = cols.get("image") or cols.get("filename") or cols.get("image_name") or cols.get("id")
            for index, row in enumerate(reader):
                name = os.path.basename(str(row[name_col]).strip()) if name_col else f"{index}.png"
                if not name.lower().endswith(".png"):
                    name += ".png"
                path = by_name.get(name)
                if not path:
                    continue
                try:
                    lat, lon = float(row[lat_col]), float(row[lon_col])
                except (TypeError, ValueError, KeyError):
                    continue
                if not _valid_coord(lat, lon):
                    continue
                emitted += 1
                yield ManifestRow(path, lat, lon, _GSV_SOURCE)
        else:
            # Headerless: row N -> N.png
            for index, row in enumerate(csv.reader(handle)):
                if len(row) < 2:
                    continue
                path = by_name.get(f"{index}.png")
                if not path:
                    continue
                try:
                    lat, lon = float(row[0]), float(row[1])
                except ValueError:
                    continue
                if not _valid_coord(lat, lon):
                    continue
                emitted += 1
                yield ManifestRow(path, lat, lon, _GSV_SOURCE)

    logger.info("google-street-view: %d images with valid coordinates", emitted)


# --- Source: im2gps (local ~2k set) ------------------------------------------

_IM2GPS_SOURCE = "im2gps"


def read_im2gps(images_dir: str | None, coords_file: str | None) -> Iterator[ManifestRow]:
    """Yield rows for the local Im2GPS set.

    The Im2GPS images do not carry coordinates in their filenames, so a companion
    coordinate file is required. ``coords_file`` is a CSV; the reader auto-detects
    an image/filename column plus latitude/longitude columns (common spellings),
    and joins to the images in ``images_dir`` by basename. If either argument is
    missing, the source is skipped with a note so the manifest can still be built
    from the other datasets.
    """
    if not coords_file or not images_dir:
        logger.warning(
            "im2gps: skipped (need --im2gps-images and --im2gps-coords to include it)"
        )
        return
    if not os.path.isfile(coords_file):
        logger.warning("im2gps: coords file %s not found; skipping", coords_file)
        return

    by_name = {
        os.path.basename(p): os.path.abspath(p)
        for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(images_dir, "**", ext), recursive=True)
    }

    with open(coords_file, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        name_col = (
            cols.get("image") or cols.get("filename") or cols.get("image_name")
            or cols.get("img") or cols.get("id") or cols.get("name")
        )
        lat_col = cols.get("lat") or cols.get("latitude")
        lon_col = cols.get("lon") or cols.get("lng") or cols.get("longitude")
        if not (name_col and lat_col and lon_col):
            logger.error(
                "im2gps: could not find name/lat/lon columns in %s (have: %s)",
                coords_file, reader.fieldnames,
            )
            return

        emitted = 0
        for row in reader:
            name = os.path.basename(str(row[name_col]).strip())
            path = by_name.get(name)
            if not path:  # try adding a .jpg if the coords list bare ids
                path = by_name.get(name + ".jpg")
            if not path:
                continue
            try:
                lat, lon = float(row[lat_col]), float(row[lon_col])
            except (TypeError, ValueError):
                continue
            if not _valid_coord(lat, lon):
                continue
            emitted += 1
            yield ManifestRow(path, lat, lon, _IM2GPS_SOURCE)
    logger.info("im2gps: %d images with valid coordinates", emitted)


# --- Consolidation -----------------------------------------------------------


def build(sources: Iterable[Iterator[ManifestRow]]) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for source in sources:
        rows.extend(source)
    return rows


def write_manifest(rows: list[ManifestRow], out_csv: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "latitude", "longitude", "source_dataset"])
        for r in rows:
            writer.writerow([r.image_path, f"{r.latitude:.8f}", f"{r.longitude:.8f}", r.source_dataset])
    logger.info("Wrote %d rows -> %s", len(rows), out_csv)

    # Parquet alongside the CSV when pyarrow is available (the harness prefers it).
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "image_path": [r.image_path for r in rows],
                "latitude": [r.latitude for r in rows],
                "longitude": [r.longitude for r in rows],
                "source_dataset": [r.source_dataset for r in rows],
            }
        )
        out_parquet = os.path.splitext(out_csv)[0] + ".parquet"
        pq.write_table(table, out_parquet)
        logger.info("Wrote Parquet -> %s", out_parquet)
    except ImportError:
        logger.info("pyarrow not installed; wrote CSV only")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unified benchmark manifest.")
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--benchmark-root", default=here, help="dir holding kaggle/<dataset>/")
    parser.add_argument("--im2gps-images", default=None, help="dir with the local Im2GPS images")
    parser.add_argument("--im2gps-coords", default=None, help="CSV of Im2GPS ground-truth coords")
    parser.add_argument("--out", default=os.path.join(here, "manifest.csv"), help="output CSV path")
    args = parser.parse_args()

    rows = build(
        [
            read_4k_benchmark(args.benchmark_root),
            read_google_street_view(args.benchmark_root),
            read_im2gps(args.im2gps_images, args.im2gps_coords),
        ]
    )
    if not rows:
        raise SystemExit("No manifest rows produced -- check dataset paths.")

    write_manifest(rows, args.out)

    # Summary.
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r.source_dataset] = by_source.get(r.source_dataset, 0) + 1
    logger.info("Total images: %d", len(rows))
    for source, n in sorted(by_source.items()):
        logger.info("  %-30s %d", source, n)


if __name__ == "__main__":
    main()
