"""Unit tests for the optional PLONK adapter (no model download required)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from viewfinder.plonk_geolocator import (
    MODEL_OSV5M,
    PlonkError,
    PlonkGeolocator,
    samples_to_candidates,
)


def test_samples_form_ranked_geographic_modes() -> None:
    samples = np.array(
        [
            [47.60, -122.33],
            [47.61, -122.32],
            [47.59, -122.34],
            [35.68, 139.69],
        ]
    )
    candidates = samples_to_candidates(samples, top_k=2, cluster_radius_km=10)

    assert len(candidates) == 2
    assert abs(candidates[0].score - 0.75) < 1e-9
    assert abs(candidates[1].score - 0.25) < 1e-9
    assert abs(candidates[0].lat - 47.60) < 0.02


def test_invalid_samples_are_rejected() -> None:
    try:
        samples_to_candidates([[91.0, 0.0]], top_k=1)
    except PlonkError:
        pass
    else:
        raise AssertionError("out-of-range latitude should fail")


def test_fake_pipeline_obeys_predictor_protocol() -> None:
    calls: list[tuple[int, object]] = []

    def fake_pipeline(image, *, batch_size, generator):
        calls.append((batch_size, generator))
        return np.array([[1.0, 2.0], [1.01, 2.01], [-20.0, 100.0]])

    locator = PlonkGeolocator(
        MODEL_OSV5M,
        samples_per_image=3,
        cluster_radius_km=10,
        pipeline=fake_pipeline,
    )
    candidates = locator.predict_image(Image.new("RGB", (16, 16)), top_k=2)

    assert len(candidates) == 2
    assert calls[0][0] == 3
    assert locator.provenance.training_dataset == "OSV-5M"
    assert locator.provenance.as_dict()["samples_per_image"] == 3

