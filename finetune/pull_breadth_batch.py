"""Pull one worldwide breadth batch from Mapillary for fine-tuning.

Runs the gallery-pipeline extract stage directly (it is Airflow-independent) and
skips validate/transform/load: fine-tuning needs raw images plus lat/lon, not the
FAISS gallery artifacts those later stages build.

Usage (from the gallery-pipeline directory, with its .env loaded):
    python pull_breadth_batch.py --max-tiles 400
"""

from __future__ import annotations

import argparse
import os
import sys
import time

GALLERY_ROOT = r"C:\View-Finder\gallery-pipeline"
CONFIG = "config/tiling_config_finetune_breadth.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tiles", type=int, default=None, help="override max_tiles_per_run")
    parser.add_argument("--max-seconds", type=int, default=None, help="override extract.max_runtime_seconds")
    parser.add_argument("--config", type=str, default=CONFIG)
    parser.add_argument("--max-attempts", type=int, default=4,
                        help="re-enter extract this many times after a transport failure")
    parser.add_argument("--resume-delay", type=int, default=20,
                        help="seconds to wait before resuming after a crash")
    args = parser.parse_args()

    os.chdir(GALLERY_ROOT)
    sys.path.insert(0, GALLERY_ROOT)

    # Load the Mapillary token from the pipeline's .env.
    env_path = os.path.join(GALLERY_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    if not os.environ.get("MAPILLARY_ACCESS_TOKEN"):
        raise SystemExit("MAPILLARY_ACCESS_TOKEN not set; cannot extract.")

    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from src.config import load_config
    from src.extract.runner import run_extract

    config = load_config(args.config)
    if args.max_tiles is not None:
        config["tiling"]["max_tiles_per_run"] = args.max_tiles
    if args.max_seconds is not None:
        config["extract"]["max_runtime_seconds"] = args.max_seconds

    print(f"config          : {args.config}")
    print(f"data_dir        : {config['paths']['data_dir']}")
    print(f"max_tiles_run   : {config['tiling']['max_tiles_per_run']}")
    print(f"imgs_per_tile   : {config['extract']['max_images_per_tile']}")
    print(f"runtime_budget  : {config['extract']['max_runtime_seconds']}s")
    print(f"ordering        : {config['tiling']['ordering']}  bounds: {config['tiling']['bounds']}")
    print("-" * 60, flush=True)

    started = time.perf_counter()
    totals = {"new_images": 0, "tiles_completed": 0, "attempts": 0, "crashes": 0}

    # run_extract retries HTTP 500s but lets a transport-level failure
    # (ConnectionResetError -> requests.ConnectionError) propagate and kill the
    # process. The checkpoint store has already persisted every completed tile,
    # so re-entering run_extract resumes rather than repeating work -- each
    # attempt therefore makes forward progress even across resets.
    for attempt in range(1, args.max_attempts + 1):
        totals["attempts"] = attempt
        try:
            result = run_extract(config)
            totals["new_images"] += int(result.get("new_images", 0))
            totals["tiles_completed"] += int(result.get("tiles_completed", 0))
            if result.get("stopped_early"):
                print(f"[resume] extractor stopped early on attempt {attempt}; budget reached.")
            break
        except Exception as exc:  # noqa: BLE001 - resume on any transport failure
            totals["crashes"] += 1
            print(f"[resume] attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt == args.max_attempts:
                print("[resume] giving up; completed tiles remain checkpointed.")
                break
            print(f"[resume] re-entering extract in {args.resume_delay}s (completed tiles are skipped)...",
                  flush=True)
            time.sleep(args.resume_delay)

    elapsed = time.perf_counter() - started
    print("-" * 60)
    print(f"extract finished in {elapsed:.1f}s")
    for key, value in sorted(totals.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
