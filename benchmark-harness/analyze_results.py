"""Compute standard geolocation accuracy metrics from benchmark results.

Reports, for each predictor independently and broken down by source_dataset:
  * % of predictions within 1 / 25 / 200 / 750 / 2500 km of ground truth
    (the standard Im2GPS3k / GeoCLIP literature thresholds, so numbers are
    directionally comparable to published results)
  * mean and median distance error in km

Plus a confidence-calibration check: predictions bucketed by confidence, with
the actual mean/median error inside each bucket. If confidence is well
calibrated, error should fall monotonically as confidence rises.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict

THRESHOLDS_KM = [1, 25, 200, 750, 2500]
CONFIDENCE_BUCKETS = [(0, 25), (25, 50), (50, 75), (75, 100.01)]


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict, key: str) -> float | None:
    v = row.get(key, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def metrics(distances: list[float]) -> dict:
    if not distances:
        return {"n": 0}
    out = {"n": len(distances)}
    for t in THRESHOLDS_KM:
        out[f"within_{t}km"] = 100.0 * sum(1 for d in distances if d <= t) / len(distances)
    out["mean_km"] = statistics.mean(distances)
    out["median_km"] = statistics.median(distances)
    return out


def fmt_table(title: str, rows: list[tuple[str, dict]]) -> str:
    lines = [f"### {title}", ""]
    lines.append("| Subset | n | <=1km | <=25km | <=200km | <=750km | <=2500km | Mean km | Median km |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, m in rows:
        if not m.get("n"):
            lines.append(f"| {label} | 0 | – | – | – | – | – | – | – |")
            continue
        lines.append(
            f"| {label} | {m['n']} | {m['within_1km']:.1f}% | {m['within_25km']:.1f}% | "
            f"{m['within_200km']:.1f}% | {m['within_750km']:.1f}% | {m['within_2500km']:.1f}% | "
            f"{m['mean_km']:,.0f} | {m['median_km']:,.0f} |"
        )
    lines.append("")
    return "\n".join(lines)


def calibration(rows: list[dict], conf_key: str, dist_key: str) -> list[tuple[str, dict]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        c, d = _f(r, conf_key), _f(r, dist_key)
        if c is None or d is None:
            continue
        for lo, hi in CONFIDENCE_BUCKETS:
            if lo <= c < hi:
                buckets[f"{lo}-{int(hi)}%"].append(d)
                break
    out = []
    for lo, hi in CONFIDENCE_BUCKETS:
        label = f"{lo}-{int(hi)}%"
        ds = buckets.get(label, [])
        out.append((label, {
            "n": len(ds),
            "mean_km": statistics.mean(ds) if ds else None,
            "median_km": statistics.median(ds) if ds else None,
        }))
    return out


def fmt_calibration(title: str, buckets: list[tuple[str, dict]]) -> str:
    lines = [f"### {title}", "", "| Confidence bucket | n | Mean error km | Median error km |", "|---|---:|---:|---:|"]
    for label, m in buckets:
        if not m["n"]:
            lines.append(f"| {label} | 0 | – | – |")
        else:
            lines.append(f"| {label} | {m['n']} | {m['mean_km']:,.0f} | {m['median_km']:,.0f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.join(here, "results.csv"))
    parser.add_argument("--out", default=os.path.join(here, "metrics.md"))
    args = parser.parse_args()

    rows = load(args.results)
    print(f"loaded {len(rows)} result rows")

    sources = sorted({r["source_dataset"] for r in rows})
    sections: list[str] = []

    for predictor, dist_key, conf_key in (
        ("GeoCLIP", "geoclip_distance_km", "geoclip_confidence"),
        ("Retrieval", "retrieval_distance_km", "retrieval_confidence"),
        ("PLONK OSV-5M", "plonk_osv_distance_km", "plonk_osv_confidence"),
        ("PLONK YFCC", "plonk_yfcc_distance_km", "plonk_yfcc_confidence"),
    ):
        table_rows = []
        overall = [d for r in rows if (d := _f(r, dist_key)) is not None]
        table_rows.append(("**Overall**", metrics(overall)))
        for s in sources:
            ds = [d for r in rows if r["source_dataset"] == s and (d := _f(r, dist_key)) is not None]
            table_rows.append((s, metrics(ds)))
        sections.append(fmt_table(f"{predictor} accuracy", table_rows))
        sections.append(fmt_calibration(f"{predictor} confidence calibration", calibration(rows, conf_key, dist_key)))

    # Retrieval-specific: does top_similarity track error better than confidence?
    sim_buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        s, d = _f(r, "retrieval_top_similarity"), _f(r, "retrieval_distance_km")
        if s is None or d is None:
            continue
        for lo, hi in [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
            if lo <= s < hi:
                sim_buckets[f"{lo:.1f}-{hi:.1f}"].append(d)
                break
    lines = ["### Retrieval: error by top_similarity (raw match quality)", "",
             "| top_similarity | n | Mean error km | Median error km |", "|---|---:|---:|---:|"]
    for lo, hi in [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        label = f"{lo:.1f}-{hi:.1f}"
        ds = sim_buckets.get(label, [])
        if ds:
            lines.append(f"| {label} | {len(ds)} | {statistics.mean(ds):,.0f} | {statistics.median(ds):,.0f} |")
        else:
            lines.append(f"| {label} | 0 | – | – |")
    sections.append("\n".join(lines) + "\n")

    # Head-to-head: which predictor was closer, per image.
    both = [(g, rt) for r in rows
            if (g := _f(r, "geoclip_distance_km")) is not None
            and (rt := _f(r, "retrieval_distance_km")) is not None]
    if both:
        g_wins = sum(1 for g, rt in both if g < rt)
        sections.append(
            "### Head-to-head\n\n"
            f"- Images scored by both predictors: **{len(both)}**\n"
            f"- GeoCLIP closer: **{g_wins}** ({100*g_wins/len(both):.1f}%)\n"
            f"- Retrieval closer: **{len(both)-g_wins}** ({100*(len(both)-g_wins)/len(both):.1f}%)\n"
        )

    errors = [r for r in rows if r.get("error")]
    if errors:
        sections.append(f"### Failures\n\n{len(errors)} image(s) had errors.\n")

    text = "\n".join(sections)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(text)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
