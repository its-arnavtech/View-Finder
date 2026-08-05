"""Tests for conservative cross-model consensus and calibration gating."""

from __future__ import annotations

import json
from pathlib import Path

from viewfinder.trust import assess_trust


def _output(lat: float, lon: float) -> dict:
    return {"prediction": {"latitude": lat, "longitude": lon}}


def test_disagreement_abstains() -> None:
    result = assess_trust(
        {
            "geoclip": _output(47.6, -122.3),
            "plonk-osv": _output(35.7, 139.7),
        },
        agreement_radius_km=50,
    )
    assert not result["consensus_reached"]
    assert not result["automation_accepted"]
    assert result["decision"] == "abstain"


def test_uncalibrated_agreement_is_only_provisional(tmp_path: Path) -> None:
    result = assess_trust(
        {
            "geoclip": _output(47.60, -122.33),
            "plonk-osv": _output(47.61, -122.32),
        },
        agreement_radius_km=50,
        calibration_path=tmp_path / "missing.json",
    )
    assert result["consensus_reached"]
    assert result["trust_level"] == "provisional"
    assert not result["automation_accepted"]
    assert result["consensus_prediction"] is not None


def test_matching_strong_calibration_allows_emit(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "agreement_radius_km": 50.0,
                "minimum_models": 2,
                "evaluation_samples": 1000,
                "accepted_samples": 400,
                "success_definition_km": 200.0,
                "observed_success_rate": 0.98,
                "wilson_lower_bound": 0.955,
                "predictors": ["geoclip", "plonk-osv"],
            }
        ),
        encoding="utf-8",
    )
    result = assess_trust(
        {
            "geoclip": _output(47.60, -122.33),
            "plonk-osv": _output(47.61, -122.32),
        },
        agreement_radius_km=50,
        required_success_rate=0.95,
        calibration_path=path,
    )
    assert result["automation_accepted"]
    assert result["decision"] == "emit"
    assert result["trust_level"] == "high"


def test_mismatched_calibration_never_allows_emit(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "agreement_radius_km": 100.0,
                "minimum_models": 2,
                "evaluation_samples": 1000,
                "accepted_samples": 400,
                "success_definition_km": 200.0,
                "observed_success_rate": 0.99,
                "wilson_lower_bound": 0.97,
                "predictors": ["geoclip", "plonk-osv"],
            }
        ),
        encoding="utf-8",
    )
    result = assess_trust(
        {
            "geoclip": _output(47.60, -122.33),
            "plonk-osv": _output(47.61, -122.32),
        },
        agreement_radius_km=50,
        calibration_path=path,
    )
    assert result["consensus_reached"]
    assert not result["automation_accepted"]
