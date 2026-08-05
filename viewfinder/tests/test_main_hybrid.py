"""CLI regression tests for fail-closed hybrid output."""

from __future__ import annotations

import json

from viewfinder import main
from viewfinder.hybrid_geolocator import HybridAssessment


def test_hybrid_require_evidence_returns_three_on_abstention(
    monkeypatch, tmp_path, capsys
) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"inference is stubbed")
    output = tmp_path / "result.json"
    spec = main.PredictorSpec("hybrid", "hybrid", lambda: object(), "")

    monkeypatch.setattr(main, "_predictor_specs", lambda args: [spec])
    monkeypatch.setattr(
        main,
        "_process_image",
        lambda *args, **kwargs: {
            "predictor": "hybrid",
            "prediction": None,
            "predictions": [],
            "visual_hypotheses": [
                {
                    "rank": 1,
                    "lat": 35.0,
                    "lon": 33.0,
                    "country": "CY",
                    "admin1": "Lefkosia",
                    "city": "Morfou",
                }
            ],
            "evidence_assessment": HybridAssessment(
                decision="abstain",
                reliability_tier="visual_hypothesis_only",
                reason="No accepted evidence.",
                automation_accepted=False,
                state_country_accepted=False,
                evidence_agreement_radius_km=None,
                selected_source="visual_fallback",
            ).as_dict(),
        },
    )

    exit_code = main.run(
        [
            "--input",
            str(image),
            "--output",
            str(output),
            "--predictor",
            "hybrid",
            "--require-evidence",
            "--quiet",
        ]
    )

    assert exit_code == 3
    assert json.loads(output.read_text(encoding="utf-8"))["prediction"] is None
    assert '"prediction": null' in capsys.readouterr().out
