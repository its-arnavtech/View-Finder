"""Create a conservative trust-gate calibration from held-out benchmark rows.

This script evaluates one pre-declared agreement rule. It does not search for a
best threshold on the evaluation data; doing that would leak the test set and
make the resulting reliability claim optimistic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from viewfinder.aggregator import Candidate, haversine_km, weighted_centroid

PREDICTORS = ("geoclip", "plonk_osv", "plonk_yfcc")


def _value(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _largest_agreeing_group(
    points: dict[str, Candidate],
    *,
    radius_km: float,
    minimum_models: int,
) -> list[str] | None:
    names = sorted(points)
    for size in range(len(names), minimum_models - 1, -1):
        groups: list[tuple[float, list[str]]] = []
        for group in itertools.combinations(names, size):
            distances = [
                haversine_km(
                    points[left].lat,
                    points[left].lon,
                    points[right].lat,
                    points[right].lon,
                )
                for left, right in itertools.combinations(group, 2)
            ]
            spread = max(distances, default=0.0)
            if spread <= radius_km:
                groups.append((spread, list(group)))
        if groups:
            return min(groups, key=lambda item: (item[0], item[1]))[1]
    return None


def wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    """Two-sided 95% Wilson score interval lower bound."""
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return (centre - margin) / denominator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate View Finder's fixed trusted-consensus rule."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "models" / "trust_calibration.json"
    )
    parser.add_argument("--agreement-radius-km", type=float, default=200.0)
    parser.add_argument("--minimum-models", type=int, default=2)
    parser.add_argument("--success-km", type=float, default=200.0)
    parser.add_argument("--minimum-accepted", type=int, default=100)
    parser.add_argument("--required-success-rate", type=float, default=0.95)
    args = parser.parse_args()
    if not 0.0 < args.required_success_rate <= 1.0:
        parser.error("--required-success-rate must be in (0, 1]")
    if args.minimum_models < 2:
        parser.error("--minimum-models must be at least 2")
    if args.minimum_accepted < 1:
        parser.error("--minimum-accepted must be positive")

    raw = args.results.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    accepted = successes = evaluable = 0

    for row in rows:
        true_lat = _value(row, "true_lat")
        true_lon = _value(row, "true_lon")
        if true_lat is None or true_lon is None:
            continue
        points: dict[str, Candidate] = {}
        for predictor in PREDICTORS:
            lat = _value(row, f"{predictor}_pred_lat")
            lon = _value(row, f"{predictor}_pred_lon")
            if lat is not None and lon is not None:
                points[predictor.replace("_", "-")] = Candidate(lat, lon, 1.0)
        if len(points) < args.minimum_models:
            continue
        evaluable += 1
        group = _largest_agreeing_group(
            points,
            radius_km=args.agreement_radius_km,
            minimum_models=args.minimum_models,
        )
        if not group:
            continue
        accepted += 1
        lat, lon = weighted_centroid([points[name] for name in group])
        if haversine_km(true_lat, true_lon, lat, lon) <= args.success_km:
            successes += 1

    if accepted < args.minimum_accepted:
        raise SystemExit(
            f"refusing calibration: only {accepted} accepted held-out examples; "
            f"need at least {args.minimum_accepted}"
        )

    observed = successes / accepted
    lower_bound = wilson_lower(successes, accepted)
    accepted_for_runtime = lower_bound >= args.required_success_rate
    payload = {
        "schema_version": 1,
        "agreement_radius_km": args.agreement_radius_km,
        "minimum_models": args.minimum_models,
        "evaluation_samples": evaluable,
        "accepted_samples": accepted,
        "coverage": accepted / evaluable if evaluable else 0.0,
        "success_definition_km": args.success_km,
        "successful_samples": successes,
        "observed_success_rate": observed,
        "wilson_lower_bound": lower_bound,
        "confidence_level": 0.95,
        "required_success_rate": args.required_success_rate,
        "accepted_for_runtime": accepted_for_runtime,
        "benchmark_sha256": hashlib.sha256(raw).hexdigest(),
        "predictors": ["geoclip", "plonk-osv", "plonk-yfcc"],
        "note": (
            "Valid only for this fixed rule and a benchmark not used to choose its "
            "thresholds. Recalibrate whenever models, data, or preprocessing change."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not accepted_for_runtime:
        print(
            f"REJECTED: Wilson lower bound {lower_bound:.1%} is below "
            f"{args.required_success_rate:.1%}; do not install this artifact."
        )
        return 4
    print("ACCEPTED: calibration meets the required lower bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
