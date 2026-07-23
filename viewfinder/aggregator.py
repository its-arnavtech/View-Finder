"""Multi-frame aggregation and confidence scoring.

This module is deliberately free of torch/geoclip/faiss imports: it operates on
plain ``Candidate`` records, so it can be reasoned about and tested without
loading a model.

It is shared by both predictors -- GeoCLIP (:mod:`geolocator`) and image-to-image
retrieval (:mod:`retrieval_geolocator`) -- so that both produce the same shape of
ranked result. Scores are normalized per frame against that frame's best
candidate, which is what makes the two predictors' different score scales
(softmax probability vs cosine similarity) both usable here without change.
Results from the two systems are never pooled into one aggregation: they are
independent predictors and merging them would be misleading.

Two jobs live here:

1. **Aggregation.** Every frame contributes a ranked list of candidate GPS
   points with similarity scores. Averaging coordinates directly is wrong -- a
   video that flips between Lisbon and Tokyo would "average" to the middle of
   Asia. Instead all retained top-k candidates are clustered by great-circle
   distance and each distinct geographic hypothesis remains a separate ranked
   prediction.

2. **Confidence scoring.** See :func:`score_confidence` for the full derivation
   of the 0-100 score.

Coordinate conventions: latitude in [-90, 90], longitude in [-180, 180],
distances in kilometres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

EARTH_RADIUS_KM = 6371.0088

# --- Tunables -----------------------------------------------------------------

# Candidates within this great-circle distance are treated as "the same place"
# when clustering. ~50 km keeps a metro area together while still separating
# neighbouring cities.
DEFAULT_CLUSTER_RADIUS_KM = 50.0

# Distance scales for the exponential decay that turns a spread in kilometres
# into a 0-1 score. At d == tau the signal is worth 1/e (~0.37).
TAU_TOPK_SPREAD_KM = 250.0
TAU_FRAME_SPREAD_KM = 150.0

# Confidence signal weights. They must sum to 1 within each mode.
# Single image: only the two per-frame signals exist.
WEIGHTS_IMAGE = {"similarity": 0.55, "geographic": 0.45}
# Video: cross-frame agreement is the strongest available evidence, because it
# is the only signal that is independent of the model's own calibration.
WEIGHTS_VIDEO = {"similarity": 0.30, "geographic": 0.25, "agreement": 0.45}

# Cross-frame agreement is meaningless with one frame and weak with two, so it
# is shrunk toward the neutral value 0.5 by n / (n + AGREEMENT_PRIOR_FRAMES).
AGREEMENT_PRIOR_FRAMES = 2.0

# Within the agreement signal: how much comes from "what share of the evidence
# backs the winning cluster" vs "how tightly packed that cluster is".
AGREEMENT_MASS_WEIGHT = 0.6
AGREEMENT_TIGHTNESS_WEIGHT = 0.4

# The score is a relative reliability indicator, not a calibrated probability.
# Never report absolute certainty.
MAX_CONFIDENCE = 0.99

# Per-cluster scoring adds an explicit evidence-volume signal. Cross-frame
# agreement answers "what fraction of frames support this place?", while
# support answers "how much independent evidence do we have at all?" This
# prevents 1/1 agreeing frames from looking as strong as 10/10 agreeing frames.
CLUSTER_WEIGHTS_IMAGE = {"similarity": 0.55, "geographic": 0.30, "support": 0.15}
CLUSTER_WEIGHTS_VIDEO = {
    "similarity": 0.30,
    "geographic": 0.20,
    "agreement": 0.35,
    "support": 0.15,
}
SUPPORT_PRIOR = 3.0


# --- Data types ---------------------------------------------------------------


@dataclass
class Candidate:
    """One scored coordinate proposed by a predictor.

    Both predictors emit this type. For GeoCLIP the score is a softmax
    probability over its GPS gallery; for the retrieval predictor it is a cosine
    similarity against a reference photo. The two scales are never mixed in one
    aggregation, and :func:`aggregate` normalizes scores per frame anyway, so
    only the *relative* ordering within a frame matters here.
    """

    lat: float
    lon: float
    score: float
    """Predictor-specific similarity score. Higher is better."""

    source_image_id: str | None = None
    """Reference photo this candidate came from, when the predictor has one.

    Populated by the retrieval predictor, which matches against real gallery
    photos. GeoCLIP scores coordinates directly and leaves this ``None``.
    """

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "score": float(self.score),
        }
        if self.source_image_id is not None:
            payload["source_image_id"] = self.source_image_id
        return payload


@dataclass
class FramePrediction:
    """The full candidate list produced for a single frame or image."""

    candidates: list[Candidate]
    frame_index: int = 0
    timestamp: float = 0.0
    blur_score: float | None = None

    @property
    def top(self) -> Candidate:
        return self.candidates[0]

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "top_prediction": self.top.as_dict(),
            "candidates": [c.as_dict() for c in self.candidates],
        }
        if self.blur_score is not None:
            out["blur_score"] = round(self.blur_score, 2)
        return out


@dataclass
class Cluster:
    """A group of candidate points that fall within the clustering radius."""

    lat: float
    lon: float
    weight: float
    members: list[Candidate] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    def dispersion_km(self) -> float:
        """Score-weighted mean distance of members from the cluster centroid."""
        if len(self.members) < 2:
            return 0.0
        distances = [haversine_km(self.lat, self.lon, m.lat, m.lon) for m in self.members]
        weights = [max(m.score, 0.0) for m in self.members]
        total = sum(weights)
        if total <= 0:
            return float(np.mean(distances))
        return float(sum(d * w for d, w in zip(distances, weights)) / total)


@dataclass
class ConfidenceBreakdown:
    """The individual 0-1 signals behind the final score, for transparency."""

    similarity: float
    geographic: float
    agreement: float | None
    weights: dict[str, float]
    confidence: float
    """Final score in the range 0-100."""
    support: float | None = None

    def as_dict(self) -> dict[str, Any]:
        signals: dict[str, float | None] = {
            "similarity_strength_and_consistency": round(self.similarity, 4),
            "geographic_agreement": round(self.geographic, 4),
            "cross_frame_agreement": (
                None if self.agreement is None else round(self.agreement, 4)
            ),
        }
        if self.support is not None:
            signals["support_reliability"] = round(self.support, 4)

        return {
            "confidence_percent": round(self.confidence, 1),
            "signals": signals,
            "weights": self.weights,
        }


@dataclass
class ClusterPrediction:
    """One independently scored geographic hypothesis."""

    lat: float
    lon: float
    confidence: float
    support_count: int
    candidate_count: int
    confidence_breakdown: ConfidenceBreakdown
    cluster: Cluster = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "confidence": round(self.confidence, 1),
            "support_count": self.support_count,
            "candidate_count": self.candidate_count,
        }


@dataclass
class AggregateResult:
    """Ranked cluster predictions, with compatibility accessors for primary."""

    predictions: list[ClusterPrediction]
    frame_count: int

    @property
    def primary(self) -> ClusterPrediction:
        return self.predictions[0]

    @property
    def lat(self) -> float:
        return self.primary.lat

    @property
    def lon(self) -> float:
        return self.primary.lon

    @property
    def confidence(self) -> ConfidenceBreakdown:
        return self.primary.confidence_breakdown

    @property
    def winning_cluster(self) -> Cluster:
        return self.primary.cluster

    @property
    def alternatives(self) -> list[Cluster]:
        return [prediction.cluster for prediction in self.predictions[1:]]

    @property
    def clusters(self) -> list[Cluster]:
        return [prediction.cluster for prediction in self.predictions]

    @property
    def frame_dispersion_km(self) -> float:
        return self.primary.cluster.dispersion_km()

    def as_dict(self) -> dict[str, Any]:
        return {"predictions": [prediction.as_dict() for prediction in self.predictions]}


# --- Geometry -----------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def weighted_centroid(points: Sequence[Candidate]) -> tuple[float, float]:
    """Score-weighted centroid of lat/lon points.

    Averaging degrees directly breaks across the antimeridian (e.g. +179 and
    -179 average to 0, the wrong side of the planet) and distorts near the
    poles. Points are converted to unit vectors in 3D, averaged there, then
    projected back onto the sphere.
    """
    if not points:
        raise ValueError("cannot compute a centroid of zero points")

    lats = np.radians([p.lat for p in points])
    lons = np.radians([p.lon for p in points])
    weights = np.array([max(p.score, 0.0) for p in points], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)

    x = np.sum(np.cos(lats) * np.cos(lons) * weights)
    y = np.sum(np.cos(lats) * np.sin(lons) * weights)
    z = np.sum(np.sin(lats) * weights)

    hyp = math.hypot(float(x), float(y))
    if hyp < 1e-12 and abs(float(z)) < 1e-12:
        # Perfectly antipodal inputs have no meaningful centroid; fall back to
        # the highest-scoring point rather than returning a degenerate value.
        best = max(points, key=lambda p: p.score)
        return best.lat, best.lon

    return math.degrees(math.atan2(float(z), hyp)), math.degrees(math.atan2(float(y), float(x)))


# --- Clustering ---------------------------------------------------------------


def cluster_candidates(
    candidates: Iterable[Candidate],
    radius_km: float = DEFAULT_CLUSTER_RADIUS_KM,
) -> list[Cluster]:
    """Group candidates with haversine connected-component clustering.

    Two points are neighbours when they are within ``radius_km``. A cluster is
    a connected component of that neighbour graph (equivalent to DBSCAN with
    ``min_samples=1``). Unlike a greedy leader pass, the result is independent
    of candidate order and does not accidentally split one hypothesis merely
    because two edge points lie on opposite sides of its centroid.

    This is O(n^2), which is appropriate for frame-count x top-k pools that are
    normally at most a few hundred points and avoids another dependency.

    Returns:
        Clusters sorted by total weight, descending.
    """
    points = list(candidates)
    if radius_km <= 0:
        raise ValueError("cluster radius must be greater than zero")

    unvisited = set(range(len(points)))
    clusters: list[Cluster] = []

    while unvisited:
        seed = max(unvisited, key=lambda index: points[index].score)
        unvisited.remove(seed)
        component = [seed]
        stack = [seed]

        while stack:
            current = stack.pop()
            neighbours = [
                index
                for index in unvisited
                if haversine_km(
                    points[current].lat,
                    points[current].lon,
                    points[index].lat,
                    points[index].lon,
                )
                <= radius_km
            ]
            for neighbour in neighbours:
                unvisited.remove(neighbour)
                component.append(neighbour)
                stack.append(neighbour)

        members = sorted(
            (points[index] for index in component),
            key=lambda candidate: candidate.score,
            reverse=True,
        )

        lat, lon = weighted_centroid(members)
        clusters.append(
            Cluster(
                lat=lat,
                lon=lon,
                weight=sum(max(member.score, 0.0) for member in members),
                members=members,
            )
        )

    clusters.sort(key=lambda c: c.weight, reverse=True)
    return clusters


# --- Confidence ---------------------------------------------------------------


def similarity_concentration(candidates: Sequence[Candidate]) -> float:
    """Signal (a): how much of the top-k score mass sits on the top-1 candidate.

    Absolute scores are not comparable between images or between predictors:
    GeoCLIP emits a softmax over a ~100k-entry GPS gallery (so the values are
    tiny), while the retrieval predictor emits cosine similarities in [-1, 1].
    What *is* comparable in both cases is the shape of the top-k tail: a
    confident prediction puts most of the retained mass on one point, an
    uncertain one spreads it evenly.

    The raw share lies in [1/k, 1], so it is rescaled to [0, 1]: an even split
    across the top-k scores 0, all mass on the top-1 scores 1.
    """
    scores = np.array([max(c.score, 0.0) for c in candidates], dtype=np.float64)
    k = len(scores)
    if k <= 1:
        # With a single candidate there is no spread to measure. 0.5 is the
        # neutral value -- claiming either extreme would be unfounded.
        return 0.5
    total = scores.sum()
    if total <= 0:
        return 0.0
    share = float(scores.max() / total)
    return float(np.clip((share - 1.0 / k) / (1.0 - 1.0 / k), 0.0, 1.0))


def geographic_agreement(candidates: Sequence[Candidate]) -> float:
    """Signal (b): how geographically tight the top-k candidates are.

    Score concentration alone can mislead. The top-5 might carry near-identical
    scores yet all sit within one city, which is a *good* outcome the
    similarity signal would score as uncertain. So the score-weighted mean
    distance from the top-1 to the remaining candidates is mapped through
    ``exp(-d / tau)``: co-located candidates approach 1, candidates scattered
    across continents approach 0.
    """
    if len(candidates) < 2:
        return 0.5

    top = candidates[0]
    rest = candidates[1:]
    weights = np.array([max(c.score, 0.0) for c in rest], dtype=np.float64)
    distances = np.array([haversine_km(top.lat, top.lon, c.lat, c.lon) for c in rest])

    if weights.sum() <= 0:
        mean_distance = float(distances.mean())
    else:
        mean_distance = float((distances * weights).sum() / weights.sum())

    return float(math.exp(-mean_distance / TAU_TOPK_SPREAD_KM))


def cross_frame_agreement(winning: Cluster, clusters: Sequence[Cluster], frame_count: int) -> float:
    """Signal (c): do independent frames of the video point at the same place?

    Two components:

    * **mass** -- the winning cluster's share of total candidate weight. If
      most frames land in one cluster, the prediction is corroborated; if the
      weight is split three ways, it is not.
    * **tightness** -- how far the cluster's own members sit from its centroid,
      mapped through ``exp(-d / tau)``. A cluster that only just holds together
      at the radius limit is weaker evidence than one where every frame agrees
      to within a kilometre.

    The result is shrunk toward the neutral 0.5 by ``n / (n + prior)``. Without
    this, a two-frame video where both frames agree would score a perfect 1.0
    on evidence that is barely better than a single image.
    """
    total_weight = sum(c.weight for c in clusters)
    mass = winning.weight / total_weight if total_weight > 0 else 0.0
    tightness = math.exp(-winning.dispersion_km() / TAU_FRAME_SPREAD_KM)

    raw = AGREEMENT_MASS_WEIGHT * mass + AGREEMENT_TIGHTNESS_WEIGHT * tightness
    reliability = frame_count / (frame_count + AGREEMENT_PRIOR_FRAMES)
    return float(reliability * raw + (1.0 - reliability) * 0.5)


def score_confidence(
    frame_predictions: Sequence[FramePrediction],
    winning: Cluster,
    clusters: Sequence[Cluster],
    top_k: int,
) -> ConfidenceBreakdown:
    """Combine the three signals into a single 0-100 confidence score.

    The score answers "how much should this prediction be trusted relative to
    other predictions from this tool", not "what is the probability the true
    location is within X km". It is a hand-weighted linear blend of:

        (a) similarity concentration  -- see :func:`similarity_concentration`
        (b) geographic agreement      -- see :func:`geographic_agreement`
        (c) cross-frame agreement     -- see :func:`cross_frame_agreement`
                                         (video with >1 frame only)

    Weights are ``WEIGHTS_IMAGE`` / ``WEIGHTS_VIDEO`` above. For video, (c)
    carries the most weight because it is the only signal that does not depend
    on the model's own (uncalibrated) score scale -- frames are near-independent
    observations, so their agreeing is real evidence.

    NOTE: these weights are chosen by hand, not fit to data. The principled
    replacement is to collect a labelled validation set (images with known GPS),
    record the three signals plus whether the prediction landed within a target
    radius (e.g. 25 km), and fit an isotonic regression from the signal vector
    to empirical accuracy. That would turn this relative score into a calibrated
    probability. The signals are exposed individually in the JSON output
    specifically so such a dataset can be assembled from real runs.
    """
    # Per-frame signals are averaged over the frames that back the winning
    # cluster. Frames that disagreed with the consensus are already penalised
    # through signal (c); letting their internal spread also drag down (a) and
    # (b) would double-count the same failure.
    winning_frames = [
        fp
        for fp in frame_predictions
        if haversine_km(fp.top.lat, fp.top.lon, winning.lat, winning.lon)
        <= DEFAULT_CLUSTER_RADIUS_KM
    ] or list(frame_predictions)

    similarity = float(
        np.mean([similarity_concentration(fp.candidates[:top_k]) for fp in winning_frames])
    )
    geographic = float(
        np.mean([geographic_agreement(fp.candidates[:top_k]) for fp in winning_frames])
    )

    frame_count = len(frame_predictions)
    if frame_count > 1:
        agreement = cross_frame_agreement(winning, clusters, frame_count)
        weights = WEIGHTS_VIDEO
        raw = (
            weights["similarity"] * similarity
            + weights["geographic"] * geographic
            + weights["agreement"] * agreement
        )
    else:
        agreement = None
        weights = WEIGHTS_IMAGE
        raw = weights["similarity"] * similarity + weights["geographic"] * geographic

    confidence = float(np.clip(raw, 0.0, MAX_CONFIDENCE)) * 100.0
    return ConfidenceBreakdown(
        similarity=similarity,
        geographic=geographic,
        agreement=agreement,
        weights=dict(weights),
        confidence=confidence,
    )


def _cluster_similarity_signal(
    cluster: Cluster,
    evidence: dict[int, tuple[int, float]],
) -> float:
    """Measure score strength and consistency for one cluster only.

    Candidate scores are normalized against the best retained candidate from
    their own frame. This makes scores comparable across frames without using
    the mass of competing geographic clusters. A coefficient-of-variation
    penalty lowers clusters whose supporting scores are highly inconsistent.
    """
    normalized = np.array([evidence[id(member)][1] for member in cluster.members])
    mean_strength = float(normalized.mean())
    if len(normalized) < 2 or mean_strength <= 0:
        consistency = 1.0
    else:
        consistency = math.exp(-float(normalized.std()) / mean_strength)
    return float(np.clip(mean_strength * consistency, 0.0, 1.0))


def _cluster_geographic_signal(cluster: Cluster) -> float:
    """Map this cluster's internal geographic dispersion to a 0-1 signal."""
    if cluster.size < 2:
        # A singleton is uncontradicted, but it is not evidence of tightness.
        return 0.5
    return float(math.exp(-cluster.dispersion_km() / TAU_FRAME_SPREAD_KM))


def _score_cluster(
    cluster: Cluster,
    evidence: dict[int, tuple[int, float]],
    frame_count: int,
) -> ClusterPrediction:
    """Compute confidence independently for one geographic hypothesis."""
    supporting_frames = {evidence[id(member)][0] for member in cluster.members}
    support_count = len(supporting_frames)
    similarity = _cluster_similarity_signal(cluster, evidence)
    geographic = _cluster_geographic_signal(cluster)
    support = 1.0 - math.exp(-support_count / SUPPORT_PRIOR)

    if frame_count > 1:
        agreement: float | None = support_count / frame_count
        weights = CLUSTER_WEIGHTS_VIDEO
        raw = (
            weights["similarity"] * similarity
            + weights["geographic"] * geographic
            + weights["agreement"] * agreement
            + weights["support"] * support
        )
    else:
        agreement = None
        weights = CLUSTER_WEIGHTS_IMAGE
        raw = (
            weights["similarity"] * similarity
            + weights["geographic"] * geographic
            + weights["support"] * support
        )

    confidence = float(np.clip(raw, 0.0, MAX_CONFIDENCE)) * 100.0
    breakdown = ConfidenceBreakdown(
        similarity=similarity,
        geographic=geographic,
        agreement=agreement,
        weights=dict(weights),
        confidence=confidence,
        support=support,
    )
    return ClusterPrediction(
        lat=cluster.lat,
        lon=cluster.lon,
        confidence=confidence,
        support_count=support_count,
        candidate_count=cluster.size,
        confidence_breakdown=breakdown,
        cluster=cluster,
    )


def aggregate(
    frame_predictions: Sequence[FramePrediction],
    top_k: int = 5,
    cluster_radius_km: float = DEFAULT_CLUSTER_RADIUS_KM,
    max_predictions: int = 5,
    max_alternatives: int | None = None,
) -> AggregateResult:
    """Cluster, independently score, and rank distinct location hypotheses.

    Args:
        frame_predictions: One entry per image or sampled video frame.
        top_k: Candidates per frame included in the clustering pool.
        cluster_radius_km: Great-circle radius for grouping candidates.
        max_predictions: Maximum ranked clusters to return, including primary.
        max_alternatives: Deprecated compatibility alias. If supplied, returns
            one primary plus this many alternatives.

    Returns:
        An :class:`AggregateResult`.

    Raises:
        ValueError: If inputs or limits are empty/invalid.
    """
    if not frame_predictions:
        raise ValueError("no frame predictions to aggregate")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if max_alternatives is not None:
        if max_alternatives < 0:
            raise ValueError("max_alternatives cannot be negative")
        max_predictions = max_alternatives + 1
    if max_predictions <= 0:
        raise ValueError("max_predictions must be greater than zero")

    pool: list[Candidate] = []
    # Candidate object identity preserves its source frame without changing the
    # public Candidate data type used by geolocator.py. Copies make this robust
    # even if a caller reuses one Candidate instance in multiple frames.
    evidence: dict[int, tuple[int, float]] = {}
    for frame_position, frame_prediction in enumerate(frame_predictions):
        retained = frame_prediction.candidates[:top_k]
        if not retained:
            raise ValueError(f"frame prediction at index {frame_position} has no candidates")

        frame_best = max(max(candidate.score, 0.0) for candidate in retained)
        denominator = frame_best if frame_best > 0 else 1.0
        for candidate in retained:
            pooled_candidate = Candidate(
                candidate.lat,
                candidate.lon,
                candidate.score,
                candidate.source_image_id,
            )
            pool.append(pooled_candidate)
            evidence[id(pooled_candidate)] = (
                frame_position,
                max(candidate.score, 0.0) / denominator,
            )

    clusters = cluster_candidates(pool, cluster_radius_km)
    predictions = [
        _score_cluster(cluster, evidence, len(frame_predictions)) for cluster in clusters
    ]
    predictions.sort(
        key=lambda prediction: (
            prediction.confidence,
            prediction.support_count,
            prediction.cluster.weight,
        ),
        reverse=True,
    )

    return AggregateResult(
        predictions=predictions[:max_predictions],
        frame_count=len(frame_predictions),
    )
