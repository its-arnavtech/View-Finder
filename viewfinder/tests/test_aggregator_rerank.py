"""Regression tests for the retrieval predictor's weak-similarity re-ranking.

Reproduces the real Seabeck-vs-Redway failure: a small cluster of individually
stronger (but wrong-region) matches out-ranked a larger cluster of weaker (but
correct-region) matches on confidence alone. With similarity-aware re-ranking
enabled, the better-corroborated cluster must win when the confidence leader's
best match is weak.

Runnable two ways:
    pytest viewfinder/tests/test_aggregator_rerank.py
    python viewfinder/tests/test_aggregator_rerank.py   (no pytest needed)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from viewfinder.aggregator import (
    DEFAULT_WEAK_SIMILARITY_THRESHOLD,
    Candidate,
    FramePrediction,
    aggregate,
)

# Two well-separated regions (>50 km apart, so they form distinct clusters).
REDWAY = (40.327, -123.841)      # small cluster, individually strong matches
SEABECK = (47.789, -122.889)     # larger cluster, individually weaker matches


def _jitter(center, n, base_score, step):
    """n candidates tightly around center, scores base_score, base-step, ..."""
    lat, lon = center
    return [
        Candidate(lat + 0.001 * i, lon + 0.001 * i, base_score - step * i)
        for i in range(n)
    ]


def _scenario():
    """4 strong Redway matches (~0.55) vs 6 weak Seabeck matches (~0.46)."""
    candidates = _jitter(REDWAY, 4, 0.55, 0.01) + _jitter(SEABECK, 6, 0.46, 0.01)
    return [FramePrediction(candidates=candidates)]


def _city_of(prediction):
    """Which region a prediction fell in, by nearest of the two centres."""
    from viewfinder.aggregator import haversine_km

    to_redway = haversine_km(prediction.lat, prediction.lon, *REDWAY)
    to_seabeck = haversine_km(prediction.lat, prediction.lon, *SEABECK)
    return "Redway" if to_redway < to_seabeck else "Seabeck"


def test_without_rerank_confidence_leader_wins():
    """Baseline: confidence ranking alone picks the small strong cluster (the bug)."""
    result = aggregate(_scenario(), top_k=10, max_predictions=5)
    assert _city_of(result.primary) == "Redway", "baseline should reproduce the wrong pick"
    # The bug's signature: primary has fewer matched photos than the runner-up.
    assert result.primary.candidate_count == 4
    assert result.predictions[1].candidate_count == 6


def test_rerank_promotes_better_supported_cluster():
    """With re-ranking, the larger correct cluster (Seabeck) becomes primary."""
    result = aggregate(
        _scenario(),
        top_k=10,
        max_predictions=5,
        weak_similarity_threshold=DEFAULT_WEAK_SIMILARITY_THRESHOLD,
    )
    assert _city_of(result.primary) == "Seabeck", "re-rank should favour the corroborated cluster"
    assert result.primary.candidate_count == 6
    # Redway is not discarded, just demoted to the alternatives.
    assert _city_of(result.predictions[1]) == "Redway"


def test_rerank_does_not_disturb_strong_matches():
    """A strong-similarity leader (>= threshold) is never overtaken.

    Mirrors the Avenue of the Giants case: the correct cluster is both the
    confidence leader and a strong match, and must stay primary even though a
    larger weak cluster exists.
    """
    strong_small = _jitter(REDWAY, 4, 0.74, 0.01)      # correct, strong, fewer photos
    weak_large = _jitter(SEABECK, 8, 0.40, 0.01)        # wrong, weak, more photos
    frames = [FramePrediction(candidates=strong_small + weak_large)]

    result = aggregate(
        frames,
        top_k=20,
        max_predictions=5,
        weak_similarity_threshold=DEFAULT_WEAK_SIMILARITY_THRESHOLD,
    )
    assert _city_of(result.primary) == "Redway", "strong-match leader must not be re-ranked"
    assert result.primary.top_similarity >= DEFAULT_WEAK_SIMILARITY_THRESHOLD


def test_rerank_requires_dominant_support():
    """A weak leader is NOT overtaken by a cluster with only marginally more photos.

    5 vs 4 photos is 1.25x, below the 1.5x dominance ratio, so confidence ranking
    stands and the weak-but-strongest leader keeps the primary slot.
    """
    strong_small = _jitter(REDWAY, 4, 0.55, 0.01)
    weak_slightly_larger = _jitter(SEABECK, 5, 0.46, 0.01)
    frames = [FramePrediction(candidates=strong_small + weak_slightly_larger)]

    result = aggregate(
        frames,
        top_k=20,
        max_predictions=5,
        weak_similarity_threshold=DEFAULT_WEAK_SIMILARITY_THRESHOLD,
    )
    assert _city_of(result.primary) == "Redway", "1.25x support must not trigger a flip"


def test_top_similarity_is_exposed():
    """Each prediction carries the cluster's best raw match for the CLI caution."""
    result = aggregate(_scenario(), top_k=10, max_predictions=5)
    for prediction in result.predictions:
        assert 0.0 <= prediction.top_similarity <= 1.0
    # The strong cluster's best match is ~0.55.
    redway = next(p for p in result.predictions if _city_of(p) == "Redway")
    assert abs(redway.top_similarity - 0.55) < 1e-6


def _run_standalone():
    tests = [name for name in globals() if name.startswith("test_")]
    failures = 0
    for name in sorted(tests):
        try:
            globals()[name]()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
