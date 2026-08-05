"""Merge independently run benchmark predictors by exact image path.

Large models are intentionally benchmarked in separate processes to keep peak
GPU/RAM use bounded. This tool refuses partial or mismatched samples so the
combined file cannot silently compare different images.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEY_COLUMNS = ("image_path", "source_dataset", "true_lat", "true_lon")
PREDICTOR_PREFIXES = ("geoclip_", "retrieval_", "plonk_osv_", "plonk_yfcc_")


def load_unique(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            image_path = row.get("image_path", "")
            if not image_path:
                raise ValueError(f"{path} contains a row without image_path")
            if image_path in rows:
                raise ValueError(f"{path} contains duplicate image_path {image_path}")
            rows[image_path] = row
    return fields, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge exact-sample benchmark CSV files.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base_fields, base = load_unique(args.base)
    merged = {key: dict(row) for key, row in base.items()}
    output_fields = list(base_fields)

    for path in args.overlay:
        fields, overlay = load_unique(path)
        if set(overlay) != set(base):
            missing = len(set(base) - set(overlay))
            extra = len(set(overlay) - set(base))
            raise SystemExit(
                f"sample mismatch for {path}: {missing} base paths missing, {extra} extra"
            )
        for field in fields:
            if field not in output_fields:
                output_fields.append(field)
        for image_path, row in overlay.items():
            target = merged[image_path]
            for key in KEY_COLUMNS:
                if row.get(key, "") != target.get(key, ""):
                    raise SystemExit(f"{path}: {key} mismatch for {image_path}")
            for key, value in row.items():
                if key == "error":
                    if value:
                        target[key] = f"{target.get(key, '')}{value}"
                elif key.startswith(PREDICTOR_PREFIXES) and value != "":
                    existing = target.get(key, "")
                    if existing not in ("", value):
                        raise SystemExit(f"{path}: conflicting {key} for {image_path}")
                    target[key] = value

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        for image_path in base:
            writer.writerow(merged[image_path])

    print(f"merged {len(merged)} exact-path rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
