"""Lightweight end-to-end CLI test for trusted abstention."""

from __future__ import annotations

import json

from viewfinder import main
from viewfinder.trust import assess_trust


def test_trusted_mode_abstains_without_calibration(monkeypatch, tmp_path) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"not decoded because inference is stubbed")
    output = tmp_path / "result.json"

    specs = [
        main.PredictorSpec(key, key, lambda: object(), "")
        for key in ("geoclip", "plonk-osv", "plonk-yfcc")
    ]
    points = {
        "geoclip": (47.600, -122.330),
        "plonk-osv": (47.610, -122.320),
        "plonk-yfcc": (35.680, 139.690),
    }

    monkeypatch.setattr(main, "_predictor_specs", lambda args: specs)
    monkeypatch.setattr(
        main,
        "_process_image",
        lambda input_path, top_k, max_predictions, spec, remove_text_enabled=False: {
            "predictor": spec.key,
            "prediction": {
                "latitude": points[spec.key][0],
                "longitude": points[spec.key][1],
            },
        },
    )
    original_import = main._import_attr

    def fake_import(module: str, attribute: str):
        if (module, attribute) == ("trust", "assess_trust"):
            return assess_trust
        if (module, attribute) == ("reverse_geocode", "reverse_geocode"):
            return lambda lat, lon: {"country": "US", "admin1": "Washington", "city": "Seattle"}
        return original_import(module, attribute)

    monkeypatch.setattr(main, "_import_attr", fake_import)
    exit_code = main.run(
        [
            "--input",
            str(image),
            "--output",
            str(output),
            "--predictor",
            "trusted",
            "--require-trust",
            "--quiet",
            "--trust-calibration",
            str(tmp_path / "missing.json"),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 3
    assert payload["prediction"] is None
    assert payload["trust_assessment"]["consensus_reached"]
    assert not payload["trust_assessment"]["automation_accepted"]
    assert payload["trust_assessment"]["decision"] == "abstain"

