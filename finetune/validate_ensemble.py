"""Score the ensemble against GeoCLIP-alone and retrieval-alone on identical images.

The ensemble routes each image to one of two predictions that were already
recorded per-image during calibration, so its error is computable exactly from
those rows without re-running the models. `--parity-check` additionally runs the
real EnsembleGeolocator over a sample and asserts its routing matches this
offline reconstruction, so the shortcut is verified rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\View-Finder")

from viewfinder.ensemble_geolocator import DEFAULT_MENTION_FLOOR, DEFAULT_TRUST_THRESHOLD

THRESHOLDS_KM = (1, 25, 200, 750, 2500)


def metrics(errors: np.ndarray) -> dict:
    out = {
        "n": int(len(errors)),
        "median_error_km": float(np.median(errors)),
        "mean_error_km": float(np.mean(errors)),
    }
    for threshold in THRESHOLDS_KM:
        out[f"acc_{threshold}km"] = float(np.mean(errors <= threshold))
    return out


def ensemble_errors(rows: pd.DataFrame, threshold: float) -> np.ndarray:
    """Retrieval's error where it is trusted, GeoCLIP's everywhere else."""
    trust = rows["top_similarity"] >= threshold
    return np.where(trust, rows["retrieval_error_km"], rows["geoclip_error_km"])


def parity_check(rows: pd.DataFrame, gallery: str, sample: int, threshold: float) -> None:
    """Confirm the real EnsembleGeolocator routes the way this script assumes."""
    from PIL import Image

    from viewfinder.ensemble_geolocator import EnsembleGeolocator
    from viewfinder.geolocator import GeoLocator
    from viewfinder.retrieval_geolocator import RetrievalGeolocator

    subset = rows.sample(n=min(sample, len(rows)), random_state=7).reset_index(drop=True)
    print(f"\nparity check on {len(subset)} images (real EnsembleGeolocator vs offline reconstruction)...")

    ensemble = EnsembleGeolocator(
        geoclip=GeoLocator(offline=True),
        retrieval=RetrievalGeolocator(gallery),
        trust_threshold=threshold,
    )

    mismatches = 0
    for _, row in subset.iterrows():
        image = Image.open(row["image_path"]).convert("RGB")
        _, decision = ensemble.predict_with_decision([image])[0]
        expected = "retrieval" if row["top_similarity"] >= threshold else "geoclip"
        if decision.routed_to != expected:
            mismatches += 1
            print(f"  MISMATCH: sim={row['top_similarity']:.4f} "
                  f"expected={expected} got={decision.routed_to}")

    if mismatches:
        raise SystemExit(f"parity check FAILED: {mismatches}/{len(subset)} routing mismatches")
    print(f"  parity OK: {len(subset)}/{len(subset)} routing decisions match")


def report(label: str, rows: pd.DataFrame, threshold: float) -> dict:
    results = {
        "geoclip_alone": metrics(rows["geoclip_error_km"].to_numpy()),
        "retrieval_alone": metrics(rows["retrieval_error_km"].to_numpy()),
        "ensemble": metrics(ensemble_errors(rows, threshold)),
    }
    routed = int((rows["top_similarity"] >= threshold).sum())

    print(f"\n{'=' * 74}")
    print(f"{label}  (n={len(rows)}, threshold={threshold})")
    print(f"routed to retrieval: {routed}/{len(rows)} ({routed / len(rows):.1%})")
    print("=" * 74)

    names = list(results.keys())
    width = 18
    print(f"{'metric':<18}" + "".join(f"{n:>{width}}" for n in names))
    print("-" * (18 + width * len(names)))
    for label_, key, fmt in [
        ("median err km", "median_error_km", ",.2f"),
        ("mean err km", "mean_error_km", ",.2f"),
    ] + [(f"acc @ {t} km", f"acc_{t}km", ".4f") for t in THRESHOLDS_KM]:
        line = f"{label_:<18}"
        for name in names:
            line += f"{results[name][key]:>{width}{fmt}}"
        print(line)

    ens = results["ensemble"]["median_error_km"]
    geo = results["geoclip_alone"]["median_error_km"]
    ret = results["retrieval_alone"]["median_error_km"]
    print(f"\nensemble vs GeoCLIP alone  : {ens - geo:+,.2f} km median "
          f"({'better' if ens < geo else 'same' if ens == geo else 'worse'})")
    print(f"ensemble vs retrieval alone: {ens - ret:+,.2f} km median "
          f"({'better' if ens < ret else 'same' if ens == ret else 'worse'})")

    results["_routed_to_retrieval"] = routed
    results["_n"] = len(rows)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib-csv", action="append", required=True,
                        help="label=path, repeatable")
    parser.add_argument("--threshold", type=float, default=DEFAULT_TRUST_THRESHOLD)
    parser.add_argument("--gallery", default=None, help="required for --parity-check")
    parser.add_argument("--parity-check", type=int, default=0,
                        help="verify N images against the real EnsembleGeolocator")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    print(f"trust threshold : {args.threshold}")
    print(f"mention floor   : {DEFAULT_MENTION_FLOOR}")

    all_results = {}
    for spec in args.calib_csv:
        # Split on the LAST '=' so labels may contain one (e.g. "Probe A (n=24)").
        label, _, path = spec.rpartition("=")
        rows = pd.read_csv(path).dropna(
            subset=["top_similarity", "retrieval_error_km", "geoclip_error_km"]
        )
        all_results[label] = report(label, rows, args.threshold)

        if args.parity_check and args.gallery:
            parity_check(rows, args.gallery, args.parity_check, args.threshold)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"threshold": args.threshold, "sets": all_results}, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
