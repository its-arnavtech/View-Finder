"""Choose the ensemble trust threshold by joint cost analysis.

Two populations decide the threshold, and they must be weighed together:

  Negative class -- the external global set. The gallery does not cover these,
      so routing one to retrieval is a false accept and costs
      (retrieval_error - geoclip_error) km on that image.

  Positive class -- Probe A, in-coverage images the gallery has not memorised.
      Routing one to retrieval is a true accept and saves
      (geoclip_error - retrieval_error) km.

For each candidate threshold we report the accept rate on each population and
the median km gained or lost, so the choice is a stated trade-off rather than a
round number.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def load(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.dropna(subset=["top_similarity", "retrieval_error_km", "geoclip_error_km"])


def analyse(negative: pd.DataFrame, positive: pd.DataFrame, thresholds) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        neg_accept = negative[negative["top_similarity"] >= threshold]
        pos_accept = positive[positive["top_similarity"] >= threshold]

        # Ensemble error = retrieval where accepted, GeoCLIP elsewhere.
        neg_ens = np.where(
            negative["top_similarity"] >= threshold,
            negative["retrieval_error_km"], negative["geoclip_error_km"],
        )
        pos_ens = np.where(
            positive["top_similarity"] >= threshold,
            positive["retrieval_error_km"], positive["geoclip_error_km"],
        )

        rows.append({
            "threshold": round(float(threshold), 3),
            "ext_accept_n": len(neg_accept),
            "ext_accept_rate": len(neg_accept) / len(negative),
            "ext_median_geoclip": float(negative["geoclip_error_km"].median()),
            "ext_median_ensemble": float(np.median(neg_ens)),
            "ext_cost_km": float(np.median(neg_ens) - negative["geoclip_error_km"].median()),
            "cov_accept_n": len(pos_accept),
            "cov_accept_rate": len(pos_accept) / len(positive),
            "cov_median_geoclip": float(positive["geoclip_error_km"].median()),
            "cov_median_ensemble": float(np.median(pos_ens)),
            "cov_gain_km": float(positive["geoclip_error_km"].median() - np.median(pos_ens)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", default=r"C:\View-Finder\finetune\work\calib_external.csv")
    parser.add_argument("--in-coverage", default=r"C:\View-Finder\finetune\work\calib_probe_a.csv")
    parser.add_argument("--upper-bound", default=None, help="optional Probe B csv, reported separately")
    args = parser.parse_args()

    negative = load(args.external)
    positive = load(args.in_coverage)

    print(f"negative class (external, gallery does NOT cover): n={len(negative)}")
    print(f"  top_similarity: min {negative['top_similarity'].min():.3f}  "
          f"median {negative['top_similarity'].median():.3f}  "
          f"p99 {negative['top_similarity'].quantile(0.99):.3f}  "
          f"max {negative['top_similarity'].max():.3f}")
    print(f"positive class (in-coverage, not memorised): n={len(positive)}")
    print(f"  top_similarity: min {positive['top_similarity'].min():.3f}  "
          f"median {positive['top_similarity'].median():.3f}  "
          f"max {positive['top_similarity'].max():.3f}")

    if args.upper_bound:
        upper = load(args.upper_bound)
        print(f"upper bound (near-duplicates, NOT used to pick): n={len(upper)}")
        print(f"  top_similarity: min {upper['top_similarity'].min():.3f}  "
              f"median {upper['top_similarity'].median():.3f}  "
              f"max {upper['top_similarity'].max():.3f}")
        print(f"  retrieval median error: {upper['retrieval_error_km'].median():.3f} km")

    print("\n=== threshold trade-off ===")
    table = analyse(negative, positive, np.arange(0.60, 0.96, 0.025))
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(table[[
            "threshold", "ext_accept_n", "ext_median_ensemble", "ext_cost_km",
            "cov_accept_n", "cov_accept_rate", "cov_median_ensemble", "cov_gain_km",
        ]].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    # Pick: maximise in-coverage gain subject to costing nothing globally.
    safe = table[table["ext_cost_km"] <= 0.0]
    if safe.empty:
        print("\nNo threshold leaves the external median untouched; falling back to minimum cost.")
        safe = table.nsmallest(3, "ext_cost_km")
    best = safe.nlargest(1, "cov_gain_km").iloc[0]
    print(f"\nRECOMMENDED THRESHOLD: {best['threshold']:.3f}")
    print(f"  external : accepts {int(best['ext_accept_n'])}/{len(negative)} "
          f"({best['ext_accept_rate']:.3%}), median error change {best['ext_cost_km']:+.2f} km")
    print(f"  coverage : accepts {int(best['cov_accept_n'])}/{len(positive)} "
          f"({best['cov_accept_rate']:.1%}), median error "
          f"{best['cov_median_geoclip']:.2f} -> {best['cov_median_ensemble']:.2f} km "
          f"({best['cov_gain_km']:+.2f} km)")

    # Does very-low-similarity retrieval add value as an alternative?
    print("\n=== value of low-similarity retrieval as an alternative ===")
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.65)):
        sel = negative[(negative["top_similarity"] >= lo) & (negative["top_similarity"] < hi)]
        if sel.empty:
            print(f"  [{lo:.2f},{hi:.2f}): no samples")
            continue
        beats = float((sel["retrieval_error_km"] < sel["geoclip_error_km"]).mean())
        print(f"  [{lo:.2f},{hi:.2f}): n={len(sel):>4}  retrieval beats GeoCLIP {beats:.1%}  "
              f"median retrieval error {sel['retrieval_error_km'].median():,.0f} km")


if __name__ == "__main__":
    main()
