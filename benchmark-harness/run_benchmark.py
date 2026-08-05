"""Run View Finder predictors over the benchmark manifest and score them.

Models and indices load ONCE and are reused across every image -- the harness
imports viewfinder's modules directly rather than shelling out to main.py, which
would reload ~136s of model/gallery state per image.

Each image is run through selected predictors independently, scored by haversine
distance against ground truth, and written to a per-image results file.

Usage:
    python run_benchmark.py --sample-size 1000
    python run_benchmark.py --sample-size 0        # full manifest
    python run_benchmark.py --predictors geoclip   # one predictor only
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("benchmark")
# The predictors log per-call detail we don't want 1000x.
logging.getLogger("viewfinder.retrieval_geolocator").setLevel(logging.WARNING)
logging.getLogger("viewfinder.geolocator").setLevel(logging.WARNING)

EARTH_RADIUS_KM = 6371.0088

RESULT_COLUMNS = [
    "image_path", "source_dataset", "true_lat", "true_lon",
    "geoclip_pred_lat", "geoclip_pred_lon", "geoclip_confidence", "geoclip_distance_km",
    "retrieval_pred_lat", "retrieval_pred_lon", "retrieval_confidence",
    "retrieval_top_similarity", "retrieval_distance_km",
    "plonk_osv_pred_lat", "plonk_osv_pred_lon", "plonk_osv_confidence",
    "plonk_osv_distance_km",
    "plonk_yfcc_pred_lat", "plonk_yfcc_pred_lon", "plonk_yfcc_confidence",
    "plonk_yfcc_distance_km",
    "error",
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


@dataclass
class Row:
    image_path: str
    source_dataset: str
    true_lat: float
    true_lon: float
    geoclip_pred_lat: float | None = None
    geoclip_pred_lon: float | None = None
    geoclip_confidence: float | None = None
    geoclip_distance_km: float | None = None
    retrieval_pred_lat: float | None = None
    retrieval_pred_lon: float | None = None
    retrieval_confidence: float | None = None
    retrieval_top_similarity: float | None = None
    retrieval_distance_km: float | None = None
    plonk_osv_pred_lat: float | None = None
    plonk_osv_pred_lon: float | None = None
    plonk_osv_confidence: float | None = None
    plonk_osv_distance_km: float | None = None
    plonk_yfcc_pred_lat: float | None = None
    plonk_yfcc_pred_lon: float | None = None
    plonk_yfcc_confidence: float | None = None
    plonk_yfcc_distance_km: float | None = None
    error: str = ""


def load_manifest(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Sample n rows proportionally across source_dataset.

    Proportional allocation keeps each source's share of the sample equal to its
    share of the manifest, so per-source metrics stay representative and the
    overall numbers are not skewed toward whichever dataset is larger.
    """
    if n <= 0 or n >= len(rows):
        return rows

    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source_dataset"], []).append(r)

    rng = random.Random(seed)
    sample: list[dict] = []
    sources = sorted(by_source)
    # Largest-remainder allocation so the totals sum exactly to n.
    exact = {s: len(by_source[s]) * n / len(rows) for s in sources}
    alloc = {s: int(exact[s]) for s in sources}
    remainder = n - sum(alloc.values())
    for s in sorted(sources, key=lambda s: exact[s] - int(exact[s]), reverse=True)[:remainder]:
        alloc[s] += 1

    for s in sources:
        pool = by_source[s][:]
        rng.shuffle(pool)
        sample.extend(pool[: alloc[s]])
    rng.shuffle(sample)
    return sample


def _primary(aggregate_result) -> tuple[float, float, float, float | None]:
    """(lat, lon, confidence, top_similarity) of the top-ranked cluster."""
    p = aggregate_result.primary
    return p.lat, p.lon, p.confidence, getattr(p, "top_similarity", None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the View Finder benchmark.")
    parser.add_argument("--manifest", default=os.path.join(REPO_ROOT, "benchmark-data", "manifest.csv"))
    parser.add_argument("--gallery", default=os.path.join(REPO_ROOT, "gallery-pipeline", "data", "benchmark-gallery", "gallery", "v1"))
    parser.add_argument("--sample-size", type=int, default=1000, help="0 = run the entire manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--predictors",
        default="both",
        choices=("both", "all", "geoclip", "retrieval", "plonk-osv", "plonk-yfcc"),
    )
    parser.add_argument("--plonk-samples", type=int, default=32)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv"))
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="skip images already in --out")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    logger.info("manifest: %d images", len(rows))
    selected = stratified_sample(rows, args.sample_size, args.seed)
    logger.info("selected: %d images", len(selected))
    by_source: dict[str, int] = {}
    for r in selected:
        by_source[r["source_dataset"]] = by_source.get(r["source_dataset"], 0) + 1
    for s, c in sorted(by_source.items()):
        logger.info("  %-32s %d", s, c)

    done: set[str] = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8", newline="") as handle:
            done = {r["image_path"] for r in csv.DictReader(handle)}
        logger.info("resume: %d already done, skipping", len(done))
        selected = [r for r in selected if r["image_path"] not in done]

    # --- load models ONCE ---
    from viewfinder.aggregator import FramePrediction, aggregate
    from viewfinder.video_processor import load_image

    geoclip = retrieval = plonk_osv = plonk_yfcc = None
    weak_threshold = None
    if args.predictors in ("both", "all", "geoclip"):
        from viewfinder.geolocator import GeoLocator
        t0 = time.perf_counter()
        geoclip = GeoLocator(offline=False)
        logger.info("GeoCLIP loaded in %.1fs", time.perf_counter() - t0)
    if args.predictors in ("both", "all", "retrieval"):
        from viewfinder.aggregator import DEFAULT_WEAK_SIMILARITY_THRESHOLD
        from viewfinder.retrieval_geolocator import RetrievalGeolocator
        weak_threshold = DEFAULT_WEAK_SIMILARITY_THRESHOLD
        t0 = time.perf_counter()
        retrieval = RetrievalGeolocator(args.gallery)
        logger.info("Retrieval loaded in %.1fs (gallery: %d images)",
                    time.perf_counter() - t0, retrieval.info.image_count)
    if args.predictors in ("all", "plonk-osv", "plonk-yfcc"):
        from viewfinder.plonk_geolocator import (
            MODEL_OSV5M,
            MODEL_YFCC,
            PlonkGeolocator,
        )
        if args.predictors in ("all", "plonk-osv"):
            t0 = time.perf_counter()
            plonk_osv = PlonkGeolocator(
                MODEL_OSV5M, samples_per_image=args.plonk_samples
            )
            logger.info("PLONK OSV-5M loaded in %.1fs", time.perf_counter() - t0)
        if args.predictors in ("all", "plonk-yfcc"):
            t0 = time.perf_counter()
            plonk_yfcc = PlonkGeolocator(
                MODEL_YFCC, samples_per_image=args.plonk_samples
            )
            logger.info("PLONK YFCC loaded in %.1fs", time.perf_counter() - t0)

    write_header = not (args.resume and os.path.exists(args.out))
    handle = open(args.out, "a" if args.resume else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
    if write_header:
        writer.writeheader()

    start = time.perf_counter()
    failures = 0
    processed = 0

    try:
        for i, m in enumerate(selected, start=1):
            row = Row(
                image_path=m["image_path"],
                source_dataset=m["source_dataset"],
                true_lat=float(m["latitude"]),
                true_lon=float(m["longitude"]),
            )
            try:
                image = load_image(m["image_path"])
            except Exception as exc:  # noqa: BLE001
                row.error = f"load:{type(exc).__name__}"
                failures += 1
                writer.writerow(asdict(row))
                continue

            if geoclip is not None:
                try:
                    cands = geoclip.predict_image(image, top_k=args.top_k)
                    result = aggregate([FramePrediction(candidates=cands)], top_k=args.top_k,
                                       max_predictions=5)
                    lat, lon, conf, _ = _primary(result)
                    row.geoclip_pred_lat, row.geoclip_pred_lon = lat, lon
                    row.geoclip_confidence = conf
                    row.geoclip_distance_km = haversine_km(row.true_lat, row.true_lon, lat, lon)
                except Exception as exc:  # noqa: BLE001
                    row.error += f"geoclip:{type(exc).__name__};"
                    failures += 1

            if retrieval is not None:
                try:
                    cands = retrieval.predict_image(image, top_k=max(args.top_k, 10))
                    result = aggregate([FramePrediction(candidates=cands)], top_k=max(args.top_k, 10),
                                       max_predictions=5,
                                       weak_similarity_threshold=weak_threshold)
                    lat, lon, conf, top_sim = _primary(result)
                    row.retrieval_pred_lat, row.retrieval_pred_lon = lat, lon
                    row.retrieval_confidence = conf
                    row.retrieval_top_similarity = top_sim
                    row.retrieval_distance_km = haversine_km(row.true_lat, row.true_lon, lat, lon)
                except Exception as exc:  # noqa: BLE001
                    row.error += f"retrieval:{type(exc).__name__};"
                    failures += 1

            for key, locator in (
                ("plonk_osv", plonk_osv),
                ("plonk_yfcc", plonk_yfcc),
            ):
                if locator is None:
                    continue
                try:
                    cands = locator.predict_image(image, top_k=args.top_k)
                    result = aggregate(
                        [FramePrediction(candidates=cands)],
                        top_k=args.top_k,
                        max_predictions=5,
                    )
                    lat, lon, conf, _ = _primary(result)
                    setattr(row, f"{key}_pred_lat", lat)
                    setattr(row, f"{key}_pred_lon", lon)
                    setattr(row, f"{key}_confidence", conf)
                    setattr(
                        row,
                        f"{key}_distance_km",
                        haversine_km(row.true_lat, row.true_lon, lat, lon),
                    )
                except Exception as exc:  # noqa: BLE001
                    row.error += f"{key}:{type(exc).__name__};"
                    failures += 1

            writer.writerow(asdict(row))
            processed += 1

            if i % args.log_every == 0:
                handle.flush()
                elapsed = time.perf_counter() - start
                rate = elapsed / i
                remaining = rate * (len(selected) - i)
                logger.info("%d/%d processed, elapsed %.1fm, est. remaining %.1fm (%.2fs/img, %d failures)",
                            i, len(selected), elapsed / 60, remaining / 60, rate, failures)
    finally:
        handle.close()

    elapsed = time.perf_counter() - start
    logger.info("DONE: %d processed, %d failures, %.1f min total (%.2fs/img)",
                processed, failures, elapsed / 60, elapsed / max(processed, 1))
    logger.info("results -> %s", args.out)


if __name__ == "__main__":
    main()
