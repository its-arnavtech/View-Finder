"""Tests for the ensemble combiner's routing decisions.

The thresholds under test are calibrated, not arbitrary: out-of-coverage images
never exceeded 0.803 top similarity across 1000 samples, while in-coverage
matches had a median of 0.851. These tests pin the routing behaviour either side
of that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from viewfinder.ensemble_geolocator import (
    DEFAULT_MENTION_FLOOR,
    DEFAULT_TRUST_THRESHOLD,
    EnsembleGeolocator,
)


@dataclass
class FakeCandidate:
    lat: float
    lon: float
    score: float
    source_image_id: str | None = None


class FakePredictor:
    """Stands in for either real predictor; records the top_k it was asked for."""

    def __init__(self, candidates: list[FakeCandidate]) -> None:
        self.candidates = candidates
        self.calls: list[int] = []

    def predict_batch(self, images, top_k=5):
        self.calls.append(top_k)
        return [list(self.candidates) for _ in images]


def _build(retrieval_scores: list[float], geoclip_scores: list[float] | None = None):
    retrieval = FakePredictor(
        [FakeCandidate(47.6 + i, -122.3, s, f"img{i}") for i, s in enumerate(retrieval_scores)]
    )
    geoclip = FakePredictor(
        [FakeCandidate(35.7 + i, 139.7, s) for i, s in enumerate(geoclip_scores or [0.9, 0.05])]
    )
    return EnsembleGeolocator(geoclip=geoclip, retrieval=retrieval), geoclip, retrieval


def test_strong_similarity_routes_to_retrieval() -> None:
    ensemble, _, _ = _build([0.93, 0.90])
    ranked = ensemble.predict_image(object())

    assert ranked[0].source == "retrieval"
    assert ranked[0].role == "primary"
    assert ranked[0].top_similarity == pytest.approx(0.93)
    # GeoCLIP stays visible as the independent second opinion.
    assert any(c.source == "geoclip" and c.role == "alternative" for c in ranked)


def test_similarity_just_below_threshold_routes_to_geoclip() -> None:
    ensemble, _, _ = _build([DEFAULT_TRUST_THRESHOLD - 0.001])
    ranked = ensemble.predict_image(object())

    assert ranked[0].source == "geoclip"
    assert ranked[0].role == "primary"
    # Still mentioned, because it clears the mention floor.
    assert any(c.source == "retrieval" and c.role == "alternative" for c in ranked)


def test_threshold_is_inclusive() -> None:
    ensemble, _, _ = _build([DEFAULT_TRUST_THRESHOLD])
    assert ensemble.predict_image(object())[0].source == "retrieval"


def test_very_low_similarity_drops_retrieval_entirely() -> None:
    """Below the mention floor retrieval won 10.6% at ~10,000 km median: noise."""
    ensemble, _, _ = _build([DEFAULT_MENTION_FLOOR - 0.05])
    ranked = ensemble.predict_image(object())

    assert ranked[0].source == "geoclip"
    assert all(c.source == "geoclip" for c in ranked)


def test_geoclip_candidates_never_carry_a_similarity() -> None:
    """GeoCLIP's softmax score has no similarity scale and must not be compared."""
    ensemble, _, _ = _build([0.95])
    for candidate in ensemble.predict_image(object()):
        if candidate.source == "geoclip":
            assert candidate.top_similarity is None


def test_decision_explains_routing() -> None:
    ensemble, _, _ = _build([0.95])
    _, decision = ensemble.predict_with_decision([object()])[0]
    assert decision.routed_to == "retrieval"
    assert decision.top_similarity == pytest.approx(0.95)
    assert str(DEFAULT_TRUST_THRESHOLD) in decision.reason

    ensemble, _, _ = _build([0.10])
    _, decision = ensemble.predict_with_decision([object()])[0]
    assert decision.routed_to == "geoclip"
    assert "dropped as noise" in decision.reason


def test_empty_retrieval_falls_back_to_geoclip() -> None:
    ensemble, _, _ = _build([])
    ranked, decision = ensemble.predict_with_decision([object()])[0]
    assert ranked[0].source == "geoclip"
    assert decision.routed_to == "geoclip"
    assert decision.top_similarity is None


def test_empty_batch_returns_empty() -> None:
    ensemble, geoclip, retrieval = _build([0.9])
    assert ensemble.predict_batch([]) == []
    # Neither predictor should have been invoked for an empty batch.
    assert not geoclip.calls and not retrieval.calls


def test_invalid_threshold_configuration_is_rejected() -> None:
    fake = FakePredictor([])
    with pytest.raises(ValueError):
        EnsembleGeolocator(fake, fake, trust_threshold=0.5, mention_floor=0.9)


def test_both_predictors_are_called_once_per_batch() -> None:
    ensemble, geoclip, retrieval = _build([0.9])
    ensemble.predict_batch([object(), object()], top_k=3)
    assert geoclip.calls == [3]
    assert retrieval.calls == [3]
