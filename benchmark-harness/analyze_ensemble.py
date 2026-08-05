"""Analyze head-to-head and selective-consensus performance.

The agreement grid is exploratory. A rule selected from this report must be
evaluated again on a separate untouched set before it can become calibration.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

from calibrate_trust import wilson_lower

EARTH_RADIUS_KM = 6371.0088
MODELS = ("geoclip", "plonk_osv", "plonk_yfcc")
LABELS = {
    "geoclip": "GeoCLIP",
    "plonk_osv": "PLONK OSV-5M",
    "plonk_yfcc": "PLONK YFCC",
}


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    vectors = []
    for lat, lon in points:
        phi, lam = math.radians(lat), math.radians(lon)
        vectors.append((math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)))
    x = sum(v[0] for v in vectors)
    y = sum(v[1] for v in vectors)
    z = sum(v[2] for v in vectors)
    norm = math.sqrt(x * x + y * y + z * z)
    return math.degrees(math.asin(z / norm)), math.degrees(math.atan2(y, x))


def points(row: dict[str, str]) -> dict[str, tuple[float, float]]:
    return {
        model: (value(row, f"{model}_pred_lat"), value(row, f"{model}_pred_lon"))
        for model in MODELS
    }


def evaluate_group(
    rows: list[dict[str, str]],
    group: tuple[str, ...],
    radius_km: float,
    success_km: float,
) -> tuple[int, int, float, float]:
    accepted = successes = 0
    for row in rows:
        model_points = points(row)
        spread = max(
            (
                distance(model_points[left], model_points[right])
                for left, right in itertools.combinations(group, 2)
            ),
            default=0.0,
        )
        if spread > radius_km:
            continue
        accepted += 1
        prediction = centroid([model_points[model] for model in group])
        truth = (value(row, "true_lat"), value(row, "true_lon"))
        if distance(prediction, truth) <= success_km:
            successes += 1
    rate = successes / accepted if accepted else 0.0
    return accepted, successes, rate, wilson_lower(successes, accepted)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--success-km", type=float, default=200.0)
    args = parser.parse_args()

    with args.results.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    lines = [
        "# Ensemble and selective-agreement analysis",
        "",
        f"Rows: **{len(rows)}**. Success definition: **≤{args.success_km:g} km**.",
        "",
        "This grid is exploratory and cannot itself be used as held-out calibration.",
        "",
        "## Head-to-head",
        "",
        "| Pair | Left closer | Right closer |",
        "|---|---:|---:|",
    ]
    for left, right in itertools.combinations(MODELS, 2):
        left_wins = sum(
            value(row, f"{left}_distance_km") < value(row, f"{right}_distance_km")
            for row in rows
        )
        lines.append(
            f"| {LABELS[left]} vs {LABELS[right]} | "
            f"{left_wins} ({left_wins/len(rows):.1%}) | "
            f"{len(rows)-left_wins} ({(len(rows)-left_wins)/len(rows):.1%}) |"
        )

    oracle = [
        min(value(row, f"{model}_distance_km") for model in MODELS)
        for row in rows
    ]
    lines.extend(
        [
            "",
            "## Oracle upper bound",
            "",
            "This is not deployable; it shows how often at least one model was right.",
            "",
            f"- At least one model within {args.success_km:g} km: "
            f"**{sum(error <= args.success_km for error in oracle)}/{len(oracle)} "
            f"({sum(error <= args.success_km for error in oracle)/len(oracle):.1%})**",
            "",
            "## Fixed-group agreement grid",
            "",
            "| Models | Agreement radius | Accepted | Coverage | Successes | Observed | 95% Wilson lower |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    groups = list(itertools.combinations(MODELS, 2)) + [MODELS]
    for group in groups:
        label = " + ".join(LABELS[model] for model in group)
        for radius in (1, 5, 10, 25, 50, 100, 200):
            accepted, successes, rate, lower = evaluate_group(
                rows, group, radius, args.success_km
            )
            lines.append(
                f"| {label} | {radius} km | {accepted} | "
                f"{accepted/len(rows):.1%} | {successes} | {rate:.1%} | {lower:.1%} |"
            )

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
