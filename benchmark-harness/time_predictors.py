"""Measure amortized per-image inference time for both predictors after warmup.

Loads each predictor once (as the real harness will), runs N images, discards the
first as warmup, and reports the mean/median of the rest. This is what decides
whether the full 14,002-image benchmark is feasible or needs sampling.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from viewfinder.geolocator import GeoLocator  # noqa: E402
from viewfinder.retrieval_geolocator import RetrievalGeolocator  # noqa: E402
from viewfinder.video_processor import load_image  # noqa: E402


def sample_images(manifest_csv: str, n: int, seed: int = 0) -> list[str]:
    with open(manifest_csv, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    random.Random(seed).shuffle(rows)
    return [r["image_path"] for r in rows[:n]]


def time_predictor(label: str, predict, paths: list[str]) -> dict:
    """Run predict(image) over paths; first call is warmup and excluded."""
    times: list[float] = []
    for i, p in enumerate(paths):
        try:
            image = load_image(p)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] skip unreadable {os.path.basename(p)}: {type(exc).__name__}")
            continue
        start = time.perf_counter()
        try:
            predict(image)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] predict failed on {os.path.basename(p)}: {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - start
        if i == 0:
            print(f"  [{label}] warmup call: {elapsed:.3f}s (excluded)")
            continue
        times.append(elapsed)
    if not times:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": len(times),
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.path.join(REPO_ROOT, "benchmark-data", "manifest.csv"))
    parser.add_argument("--gallery", default=os.path.join(REPO_ROOT, "gallery-pipeline", "data", "benchmark-gallery", "gallery", "v1"))
    parser.add_argument("--n", type=int, default=10, help="images to run (first is warmup)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    paths = sample_images(args.manifest, args.n)
    print(f"timing on {len(paths)} images (first excluded as warmup)\n")

    print("== loading GeoCLIP ==")
    t0 = time.perf_counter()
    geoclip = GeoLocator(offline=False)
    geoclip_load = time.perf_counter() - t0
    print(f"  load+gallery-encode: {geoclip_load:.1f}s")

    print("== loading Retrieval ==")
    t0 = time.perf_counter()
    retrieval = RetrievalGeolocator(args.gallery)
    retrieval_load = time.perf_counter() - t0
    print(f"  load: {retrieval_load:.1f}s  (gallery: {retrieval.info.image_count} images)")

    print("\n== timing GeoCLIP ==")
    g = time_predictor("geoclip", lambda im: geoclip.predict_image(im, top_k=args.top_k), paths)
    print("\n== timing Retrieval ==")
    r = time_predictor("retrieval", lambda im: retrieval.predict_image(im, top_k=args.top_k), paths)

    print("\n===================== RESULTS =====================")
    for res in (g, r):
        if res.get("n"):
            print(f"{res['label']:10s} n={res['n']:3d}  mean={res['mean']:.3f}s  median={res['median']:.3f}s  "
                  f"min={res['min']:.3f}s  max={res['max']:.3f}s")
    combined = (g.get("mean", 0) or 0) + (r.get("mean", 0) or 0)
    print(f"\ncombined per-image (both predictors): {combined:.3f}s")
    total_full = combined * 14002
    print(f"load overhead (one-time): {geoclip_load + retrieval_load:.1f}s")
    print(f"\nESTIMATED FULL RUN (14,002 images): {total_full/3600:.1f} hours ({total_full/86400:.2f} days)")
    for n in (300, 500, 1000, 2000):
        print(f"  sample n={n:5d}: {combined*n/60:.1f} min")


if __name__ == "__main__":
    main()
