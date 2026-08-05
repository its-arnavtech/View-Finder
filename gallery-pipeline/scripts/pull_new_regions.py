"""Pull Mapillary imagery into continents the pool does not yet cover.

Each target continent gets its own bounded window and its own wall-clock share of
the run, because worldwide `spread` ordering ranks supercells by city density and
would otherwise keep spending the budget in the European and North American
metros that are already well represented.

Regions are visited in order of how badly they are missing, so a truncated
session still buys the most new coverage.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

GALLERY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = "config/tiling_config_new_regions.yaml"

# (name, [min_lon, min_lat, max_lon, max_lat]) ordered by how absent they are.
TARGET_REGIONS: list[tuple[str, list[float]]] = [
    ("south-america", [-82.0, -56.0, -34.0, 13.0]),
    ("africa", [-18.0, -35.0, 52.0, 37.0]),
    ("south-asia", [60.0, 5.0, 92.0, 35.0]),
    ("east-asia", [95.0, -11.0, 146.0, 46.0]),
]


def load_env(gallery_root: str) -> None:
    env_path = os.path.join(gallery_root, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--seconds-per-region", type=int, default=600)
    parser.add_argument("--max-tiles-per-region", type=int, default=400)
    parser.add_argument("--regions", nargs="*", default=None,
                        help="subset of region names to pull (default: all)")
    args = parser.parse_args()

    os.chdir(GALLERY_ROOT)
    sys.path.insert(0, GALLERY_ROOT)
    load_env(GALLERY_ROOT)

    if not os.environ.get("MAPILLARY_ACCESS_TOKEN"):
        raise SystemExit("MAPILLARY_ACCESS_TOKEN not set; cannot extract.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from src.config import load_config
    from src.extract.runner import run_extract

    targets = TARGET_REGIONS
    if args.regions:
        wanted = set(args.regions)
        targets = [entry for entry in TARGET_REGIONS if entry[0] in wanted]
        if not targets:
            raise SystemExit(f"No target regions matched {sorted(wanted)}")

    print(f"config            : {args.config}")
    print(f"regions           : {[name for name, _ in targets]}")
    print(f"budget per region : {args.seconds_per_region}s")
    print("-" * 68, flush=True)

    totals = {"new_images": 0, "tiles_completed": 0, "tiles_failed": 0}
    per_region: dict[str, dict] = {}
    started = time.perf_counter()

    for name, bounds in targets:
        config = load_config(args.config)
        config["tiling"]["bounds"] = bounds
        config["tiling"]["max_tiles_per_run"] = args.max_tiles_per_region
        config["extract"]["max_runtime_seconds"] = args.seconds_per_region

        print(f"\n=== {name}  bounds={bounds} ===", flush=True)
        region_started = time.perf_counter()
        try:
            result = run_extract(config)
        except Exception as exc:  # noqa: BLE001 - one region must not sink the rest
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            per_region[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        elapsed = time.perf_counter() - region_started
        summary = {
            "new_images": int(result.get("new_images", 0)),
            "tiles_completed": int(result.get("tiles_completed", 0)),
            "tiles_failed": int(result.get("tiles_failed", 0)),
            "aborted_early": bool(result.get("aborted_early", False)),
            "stopped_early": bool(result.get("stopped_early", False)),
            "seconds": round(elapsed, 1),
        }
        per_region[name] = summary
        for key in totals:
            totals[key] += summary[key]

        print(
            f"[{name}] +{summary['new_images']} images from {summary['tiles_completed']} tiles "
            f"({summary['tiles_failed']} failed) in {elapsed:.0f}s"
            f"{'  [aborted]' if summary['aborted_early'] else ''}",
            flush=True,
        )

    print("\n" + "=" * 68)
    print(f"total: +{totals['new_images']} images, {totals['tiles_completed']} tiles, "
          f"{totals['tiles_failed']} failed, in {time.perf_counter() - started:.0f}s")
    for name, summary in per_region.items():
        print(f"  {name:<16} {summary}")


if __name__ == "__main__":
    main()
