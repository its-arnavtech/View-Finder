"""Conservative cross-model agreement and abstention for geolocation.

The module deliberately separates two ideas:

``consensus_reached``
    Distinct predictors landed within a configured geographic radius.

``automation_accepted``
    The same consensus rule has also been calibrated on held-out, geotagged
    images and its lower confidence bound meets the requested success rate.

Without a calibration artifact the program may report provisional consensus,
but it will not claim that the result is safe for automated use.  This is the
central guardrail against turning an attractive confidence number into a
promise the data does not support.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .aggregator import Candidate, haversine_km, weighted_centroid


DEFAULT_AGREEMENT_RADIUS_KM = 200.0
DEFAULT_MIN_MODELS = 2
DEFAULT_REQUIRED_SUCCESS_RATE = 0.95
DEFAULT_SUCCESS_DEFINITION_KM = 200.0
DEFAULT_MIN_CALIBRATION_ACCEPTED = 100
CALIBRATION_ENV = "VIEWFINDER_TRUST_CALIBRATION"
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "models" / "trust_calibration.json"


class TrustConfigurationError(ValueError):
    """Raised when a trust threshold or calibration artifact is malformed."""


def _primary_coordinates(outputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Candidate]:
    points: dict[str, Candidate] = {}
    for key, block in outputs.items():
        prediction = block.get("prediction")
        if not isinstance(prediction, Mapping):
            continue
        try:
            lat = float(prediction["latitude"])
            lon = float(prediction["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(lat) or not math.isfinite(lon):
            continue
        if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
            continue
        points[key] = Candidate(lat=lat, lon=lon, score=1.0)
    return points


def _pairwise(points: Mapping[str, Candidate]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(points), 2):
        distance = haversine_km(
            points[left].lat,
            points[left].lon,
            points[right].lat,
            points[right].lon,
        )
        pairs.append(
            {
                "models": [left, right],
                "distance_km": round(float(distance), 3),
            }
        )
    return pairs


def _best_clique(
    points: Mapping[str, Candidate],
    radius_km: float,
    min_models: int,
) -> tuple[list[str], float] | None:
    """Return the largest all-pairs-agreeing model set.

    Requiring a clique avoids transitive chaining where A is near B and B is
    near C but A and C are too far apart to represent one defensible answer.
    """
    names = sorted(points)
    candidates: list[tuple[list[str], float]] = []
    for size in range(len(names), min_models - 1, -1):
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
            maximum = max(distances, default=0.0)
            if maximum <= radius_km:
                candidates.append((list(group), float(maximum)))
        if candidates:
            break
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[1], item[0]))
    return candidates[0]


def load_calibration(path: str | Path | None = None) -> dict[str, Any] | None:
    """Load a held-out benchmark calibration artifact, if one exists."""
    configured = path or os.getenv(CALIBRATION_ENV)
    target = Path(configured).expanduser() if configured else DEFAULT_CALIBRATION_PATH
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrustConfigurationError(f"Could not read trust calibration {target}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrustConfigurationError(f"Trust calibration {target} must contain a JSON object")

    required = {
        "agreement_radius_km",
        "minimum_models",
        "evaluation_samples",
        "accepted_samples",
        "success_definition_km",
        "observed_success_rate",
        "wilson_lower_bound",
        "predictors",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise TrustConfigurationError(
            f"Trust calibration {target} is missing: {', '.join(missing)}"
        )
    if not isinstance(payload["predictors"], list) or not all(
        isinstance(item, str) for item in payload["predictors"]
    ):
        raise TrustConfigurationError(
            f"Trust calibration {target} predictors must be a list of names"
        )
    try:
        evaluation_samples = int(payload["evaluation_samples"])
        accepted_samples = int(payload["accepted_samples"])
        observed = float(payload["observed_success_rate"])
        lower = float(payload["wilson_lower_bound"])
    except (TypeError, ValueError) as exc:
        raise TrustConfigurationError(
            f"Trust calibration {target} contains invalid numeric values"
        ) from exc
    if not (
        evaluation_samples >= accepted_samples >= 0
        and 0.0 <= observed <= 1.0
        and 0.0 <= lower <= observed
    ):
        raise TrustConfigurationError(
            f"Trust calibration {target} contains inconsistent evaluation statistics"
        )
    payload["_path"] = str(target.resolve())
    return payload


def assess_trust(
    outputs: Mapping[str, Mapping[str, Any]],
    *,
    agreement_radius_km: float = DEFAULT_AGREEMENT_RADIUS_KM,
    minimum_models: int = DEFAULT_MIN_MODELS,
    required_success_rate: float = DEFAULT_REQUIRED_SUCCESS_RATE,
    success_definition_km: float = DEFAULT_SUCCESS_DEFINITION_KM,
    minimum_calibration_accepted: int = DEFAULT_MIN_CALIBRATION_ACCEPTED,
    calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assess cross-model agreement and decide whether to emit or abstain."""
    if not math.isfinite(agreement_radius_km) or agreement_radius_km <= 0:
        raise TrustConfigurationError("agreement_radius_km must be finite and positive")
    if minimum_models < 2:
        raise TrustConfigurationError("minimum_models must be at least 2")
    if not 0.0 < required_success_rate <= 1.0:
        raise TrustConfigurationError("required_success_rate must be in (0, 1]")
    if not math.isfinite(success_definition_km) or success_definition_km <= 0:
        raise TrustConfigurationError("success_definition_km must be finite and positive")
    if minimum_calibration_accepted < 1:
        raise TrustConfigurationError("minimum_calibration_accepted must be positive")

    points = _primary_coordinates(outputs)
    clique = _best_clique(points, agreement_radius_km, minimum_models)
    consensus_reached = clique is not None
    agreeing_models: list[str] = clique[0] if clique else []
    maximum_spread = clique[1] if clique else None

    consensus: dict[str, float] | None = None
    if clique:
        lat, lon = weighted_centroid([points[name] for name in agreeing_models])
        consensus = {
            "latitude": round(float(lat), 6),
            "longitude": round(float(lon), 6),
        }

    calibration = load_calibration(calibration_path)
    calibration_matches = bool(
        calibration
        and math.isclose(
            float(calibration["agreement_radius_km"]),
            agreement_radius_km,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and int(calibration["minimum_models"]) == minimum_models
        and math.isclose(
            float(calibration["success_definition_km"]),
            success_definition_km,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and int(calibration["accepted_samples"]) >= minimum_calibration_accepted
        and sorted(str(item) for item in calibration["predictors"])
        == sorted(outputs)
    )
    calibrated_success = (
        float(calibration["wilson_lower_bound"]) if calibration_matches else None
    )
    automation_accepted = bool(
        consensus_reached
        and calibration_matches
        and calibrated_success is not None
        and calibrated_success >= required_success_rate
    )

    if not consensus_reached:
        level = "abstain"
        reason = (
            f"No set of {minimum_models} predictors agreed within "
            f"{agreement_radius_km:g} km."
        )
    elif not calibration_matches:
        level = "provisional"
        reason = (
            "Models agree, but this exact agreement rule has not been calibrated "
            "on a held-out benchmark. It is not accepted for automated use."
        )
    elif not automation_accepted:
        level = "abstain"
        reason = (
            f"Models agree, but the held-out Wilson lower bound "
            f"({calibrated_success:.1%}) is below the required "
            f"{required_success_rate:.1%} success rate."
        )
    else:
        level = "high"
        reason = (
            f"{len(agreeing_models)} distinct predictors agree and held-out "
            f"calibration has a {calibrated_success:.1%} lower confidence bound."
        )

    return {
        "trust_level": level,
        "consensus_reached": consensus_reached,
        "automation_accepted": automation_accepted,
        "decision": "emit" if automation_accepted else "abstain",
        "reason": reason,
        "agreement_radius_km": float(agreement_radius_km),
        "minimum_models": int(minimum_models),
        "required_success_rate": float(required_success_rate),
        "success_definition_km": float(success_definition_km),
        "minimum_calibration_accepted": int(minimum_calibration_accepted),
        "models_evaluated": sorted(points),
        "agreeing_models": agreeing_models,
        "maximum_pairwise_spread_km": (
            None if maximum_spread is None else round(float(maximum_spread), 3)
        ),
        "consensus_prediction": consensus,
        "pairwise_distances": _pairwise(points),
        "calibration": (
            {
                key: value
                for key, value in calibration.items()
                if key != "_path"
            }
            | {"path": calibration["_path"]}
            if calibration_matches and calibration
            else None
        ),
        "warning": (
            "No image geolocation system can guarantee a correct worldwide location. "
            "This gate reduces risk by abstaining unless distinct models agree and "
            "held-out calibration supports the requested success rate."
        ),
    }
