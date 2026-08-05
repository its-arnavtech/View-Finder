"""Calibrate the ensemble trust threshold from real per-image results.

The cached benchmark files could not answer this: external_paired.json holds
per-image errors for the two GeoCLIP variants only, and
external_benchmark_results.json holds aggregate metrics with no per-image rows.
Neither ever recorded retrieval's ``top_similarity``, because the earlier harness
kept only ``candidate.lat``/``lon`` and discarded ``candidate.score``.

So this re-derives the calibration set from scratch: for every image it records
both predictors' top-1 coordinate, retrieval's top-1 cosine similarity, and each
predictor's haversine error against ground truth. Those rows are then bucketed by
``top_similarity`` to find where retrieval actually starts beating GeoCLIP.

Both predictors are used exactly as production builds them
(``viewfinder.geolocator.GeoLocator`` and
``viewfinder.retrieval_geolocator.RetrievalGeolocator``); nothing is reimplemented.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, r"C:\View-Finder")

from location_sampling import haversine_km

# Bucket edges over retrieval's top-1 cosine similarity.
DEFAULT_BUCKETS = (0.0, 0.3, 0.5, 0.65, 0.8, 1.01)


def collect(frame: pd.DataFrame, gallery_path: str, batch_size: int, top_k: int = 5) -> pd.DataFrame:
    """Run both predictors over every image and record top-1 results."""
    from viewfinder.geolocator import GeoLocator
    from viewfinder.retrieval_geolocator import RetrievalGeolocator

    print("loading predictors...", flush=True)
    geoclip = GeoLocator(offline=True)
    retrieval = RetrievalGeolocator(gallery_path)

    rows: list[dict] = []
    paths = frame["image_path"].tolist()
    truth = frame[["lat", "lon"]].to_numpy(dtype=np.float64)

    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in chunk]

        geo_batch = geoclip.predict_batch(images, top_k=top_k)
        ret_batch = retrieval.predict_batch(images, top_k=top_k)

        for offset, (geo_c, ret_c) in enumerate(zip(geo_batch, ret_batch)):
            index = start + offset
            row = {
                "image_path": chunk[offset],
                "true_lat": truth[index, 0],
                "true_lon": truth[index, 1],
            }

            if geo_c:
                row["geoclip_lat"] = geo_c[0].lat
                row["geoclip_lon"] = geo_c[0].lon
                row["geoclip_score"] = geo_c[0].score
            if ret_c:
                row["retrieval_lat"] = ret_c[0].lat
                row["retrieval_lon"] = ret_c[0].lon
                # Gallery and query vectors are L2-normalised against an
                # IndexFlatIP, so the inner product is exactly cosine similarity.
                row["top_similarity"] = ret_c[0].score
            rows.append(row)

        done = min(start + batch_size, len(paths))
        if done % (batch_size * 5) == 0 or done == len(paths):
            print(f"  {done}/{len(paths)}", flush=True)

    result = pd.DataFrame(rows)
    result["geoclip_error_km"] = haversine_km(
        result["true_lat"].to_numpy(), result["true_lon"].to_numpy(),
        result["geoclip_lat"].to_numpy(), result["geoclip_lon"].to_numpy(),
    )
    result["retrieval_error_km"] = haversine_km(
        result["true_lat"].to_numpy(), result["true_lon"].to_numpy(),
        result["retrieval_lat"].to_numpy(), result["retrieval_lon"].to_numpy(),
    )
    return result


def bucket_report(rows: pd.DataFrame, buckets=DEFAULT_BUCKETS) -> pd.DataFrame:
    """Per-similarity-bucket error comparison between the two predictors."""
    out = []
    for low, high in zip(buckets[:-1], buckets[1:]):
        sel = rows[(rows["top_similarity"] >= low) & (rows["top_similarity"] < high)]
        if sel.empty:
            out.append({"bucket": f"[{low:.2f},{high:.2f})", "n": 0})
            continue
        ret = sel["retrieval_error_km"]
        geo = sel["geoclip_error_km"]
        out.append({
            "bucket": f"[{low:.2f},{high:.2f})",
            "n": len(sel),
            "ret_median": ret.median(),
            "geo_median": geo.median(),
            "ret_mean": ret.mean(),
            "geo_mean": geo.mean(),
            "ret_wins": float((ret < geo).mean()),
            "ret_within_25km": float((ret <= 25).mean()),
            "geo_within_25km": float((geo <= 25).mean()),
        })
    return pd.DataFrame(out)


def crossover(rows: pd.DataFrame, step: float = 0.05, min_n: int = 15) -> list[dict]:
    """Sweep a threshold and measure retrieval's win rate above it.

    The threshold we want is the point above which trusting retrieval is
    genuinely better than trusting GeoCLIP, measured on the images that would
    actually be routed to retrieval.
    """
    out = []
    for threshold in np.arange(0.30, 0.96, step):
        sel = rows[rows["top_similarity"] >= threshold]
        if len(sel) < min_n:
            out.append({"threshold": round(float(threshold), 2), "n_above": len(sel), "sufficient": False})
            continue
        ret = sel["retrieval_error_km"]
        geo = sel["geoclip_error_km"]
        out.append({
            "threshold": round(float(threshold), 2),
            "n_above": len(sel),
            "sufficient": True,
            "ret_median": float(ret.median()),
            "geo_median": float(geo.median()),
            "ret_wins": float((ret < geo).mean()),
            "median_gain_km": float(geo.median() - ret.median()),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-csv", required=True)
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--label", default="calibration")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.images_csv)
    frame = frame.rename(columns={"latitude": "lat", "longitude": "lon"})
    frame = frame[frame["image_path"].map(os.path.exists)].reset_index(drop=True)
    if args.limit:
        frame = frame.head(args.limit)

    print(f"calibration: {args.label}")
    print(f"images     : {len(frame)}\n")

    rows = collect(frame, args.gallery, args.batch_size)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    rows.to_csv(args.out_csv, index=False)

    print(f"\n=== {args.label}: top_similarity distribution ===")
    print(rows["top_similarity"].describe().to_string())

    print(f"\n=== {args.label}: error by top_similarity bucket ===")
    report = bucket_report(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print(report.to_string(index=False))

    print(f"\n=== {args.label}: threshold sweep ===")
    for entry in crossover(rows):
        if not entry["sufficient"]:
            print(f"  t={entry['threshold']:.2f}  n={entry['n_above']:>4}  (too few to judge)")
        else:
            print(f"  t={entry['threshold']:.2f}  n={entry['n_above']:>4}  "
                  f"ret_median={entry['ret_median']:>9.2f}km  geo_median={entry['geo_median']:>9.2f}km  "
                  f"ret_wins={entry['ret_wins']:.3f}  gain={entry['median_gain_km']:>+9.2f}km")

    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
