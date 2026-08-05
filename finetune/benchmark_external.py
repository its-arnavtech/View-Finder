"""Benchmark predictors against an external, never-trained-on image set.

Answers a different question from the in-run held-out slices: those are carved
from our own scraped data, so they share its collection biases. This harness
scores predictors on a fully independent dataset to check whether fine-tuning
gains generalize.

Predictors compared:
  * GeoCLIP, pretrained weights
  * GeoCLIP, with a fine-tuned location-encoder checkpoint loaded
  * RetrievalGeolocator, the FAISS gallery predictor

Input is a CSV with `image_path`, `lat`, `lon` -- source-agnostic, so the same
harness runs on a GSV-Cities sample or on any held-out slice.

Errors use the haversine great-circle distance to ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from geoclip.model.misc import file_dir as geoclip_file_dir
from geoclip.model.misc import load_gps_data

from evaluate import encode_val_images, predict_from_features
from location_sampling import haversine_km
from model_setup import build_model, freeze_image_side

THRESHOLDS_KM = (1, 25, 200, 750, 2500)


def metrics_from_errors(errors_km: np.ndarray) -> dict:
    out = {
        "n": int(len(errors_km)),
        "median_error_km": float(np.median(errors_km)),
        "mean_error_km": float(np.mean(errors_km)),
        "p25_error_km": float(np.percentile(errors_km, 25)),
        "p75_error_km": float(np.percentile(errors_km, 75)),
    }
    for threshold in THRESHOLDS_KM:
        out[f"acc_{threshold}km"] = float(np.mean(errors_km <= threshold))
    return out


def errors_for(true_coords: np.ndarray, pred_coords: np.ndarray) -> np.ndarray:
    """Haversine great-circle error, elementwise."""
    return haversine_km(
        true_coords[:, 0], true_coords[:, 1], pred_coords[:, 0], pred_coords[:, 1]
    )


def run_geoclip(frame: pd.DataFrame, checkpoint: str | None, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted coords, true coords) for GeoCLIP with optional checkpoint."""
    model = build_model(device=device)
    freeze_image_side(model)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.location_encoder.load_state_dict(state["location_encoder"])
        print(f"  loaded location encoder from {os.path.basename(checkpoint)}")
    model.eval()

    gps_gallery = load_gps_data(os.path.join(geoclip_file_dir, "gps_gallery", "coordinates_100K.csv"))
    features, true_coords = encode_val_images(model, frame, batch_size, device, amp=True)
    pred_coords = predict_from_features(model, features, gps_gallery, device)

    del model
    torch.cuda.empty_cache()
    return pred_coords, true_coords


def run_retrieval(frame: pd.DataFrame, gallery_path: str, batch_size: int) -> np.ndarray:
    """Return predicted coords from the FAISS retrieval predictor (top-1)."""
    sys.path.insert(0, r"C:\View-Finder")
    from viewfinder.retrieval_geolocator import RetrievalGeolocator

    predictor = RetrievalGeolocator(gallery_path)
    preds = []
    paths = frame["image_path"].tolist()
    for start in range(0, len(paths), batch_size):
        images = [Image.open(p).convert("RGB") for p in paths[start : start + batch_size]]
        for candidates in predictor.predict_batch(images, top_k=1):
            if candidates:
                preds.append((candidates[0].lat, candidates[0].lon))
            else:
                preds.append((np.nan, np.nan))
    return np.array(preds, dtype=np.float64)


def format_table(results: dict[str, dict]) -> str:
    names = [n for n in results if not n.startswith("_")]
    width = max(len(n) for n in names) + 2
    lines = [f"{'metric':<18}" + "".join(f"{n:>{width}}" for n in names)]
    lines.append("-" * len(lines[0]))

    rows = [
        ("median err km", "median_error_km", ".2f"),
        ("mean err km", "mean_error_km", ".2f"),
        ("p25 err km", "p25_error_km", ".2f"),
        ("p75 err km", "p75_error_km", ".2f"),
    ] + [(f"acc @ {t} km", f"acc_{t}km", ".4f") for t in THRESHOLDS_KM]

    for label, key, fmt in rows:
        line = f"{label:<18}"
        for name in names:
            line += f"{results[name][key]:>{width}{fmt}}"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-csv", required=True, help="CSV with image_path, lat, lon")
    parser.add_argument("--checkpoint", default=None, help="fine-tuned location encoder .pth")
    parser.add_argument("--gallery", default=None, help="gallery version dir for the retrieval predictor")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--label", default="external benchmark")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--save-errors", action="store_true", help="include per-image errors in --json-out")
    args = parser.parse_args()

    frame = pd.read_csv(args.images_csv)
    for column in ("image_path", "lat", "lon"):
        if column not in frame.columns:
            raise SystemExit(f"{args.images_csv} is missing required column {column!r}")

    missing = [p for p in frame["image_path"] if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} image paths do not exist, e.g. {missing[0]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"benchmark: {args.label}")
    print(f"images   : {len(frame)}")
    print(f"device   : {device}\n")

    results: dict[str, dict] = {}
    per_image: dict[str, np.ndarray] = {}

    print("GeoCLIP (pretrained)...")
    pred, true_coords = run_geoclip(frame, None, device, args.batch_size)
    per_image["pretrained"] = errors_for(true_coords, pred)
    results["pretrained"] = metrics_from_errors(per_image["pretrained"])

    if args.checkpoint:
        print("GeoCLIP (fine-tuned)...")
        pred_ft, _ = run_geoclip(frame, args.checkpoint, device, args.batch_size)
        per_image["fine-tuned"] = errors_for(true_coords, pred_ft)
        results["fine-tuned"] = metrics_from_errors(per_image["fine-tuned"])

    if args.gallery:
        print("RetrievalGeolocator...")
        pred_r = run_retrieval(frame, args.gallery, args.batch_size)
        valid = ~np.isnan(pred_r[:, 0])
        if valid.sum() < len(pred_r):
            print(f"  warning: {(~valid).sum()} images returned no candidate")
        per_image["retrieval"] = errors_for(true_coords[valid], pred_r[valid])
        results["retrieval"] = metrics_from_errors(per_image["retrieval"])

    print("\n" + "=" * 70)
    print(f"{args.label}  (haversine distance to ground truth)")
    print("=" * 70)
    print(format_table(results))

    # Paired per-image comparison. Aggregate medians can hide whether a change is
    # a consistent small shift or a few large outliers moving the middle, so
    # compare the two GeoCLIP variants image by image on identical inputs.
    if "fine-tuned" in per_image and "pretrained" in per_image:
        base = per_image["pretrained"]
        tuned = per_image["fine-tuned"]
        better = int((tuned < base).sum())
        worse = int((tuned > base).sum())
        tied = int((tuned == base).sum())
        n = len(base)
        print(f"\npaired per-image (fine-tuned vs pretrained, n={n}):")
        print(f"  fine-tuned closer : {better} ({better / n:.1%})")
        print(f"  fine-tuned farther: {worse} ({worse / n:.1%})")
        print(f"  identical         : {tied} ({tied / n:.1%})")
        decisive = better + worse
        if decisive:
            # Two-sided sign test via normal approximation.
            z = (better - decisive / 2) / (0.5 * np.sqrt(decisive))
            print(f"  sign-test z       : {z:+.2f} "
                  f"({'favours fine-tuned' if z > 0 else 'favours pretrained'}, "
                  f"|z|>1.96 = significant at p<0.05)")
        results["_paired"] = {"better": better, "worse": worse, "tied": tied}

    if args.json_out:
        payload = {"label": args.label, "n": int(len(frame)), "results": results}
        if args.save_errors:
            payload["per_image_errors_km"] = {k: v.tolist() for k, v in per_image.items()}
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
