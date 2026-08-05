"""Ensemble combiner: GeoCLIP as global fallback, retrieval as local override.

This is a composing layer, not a new predictor. It runs
:class:`geolocator.GeoLocator` and
:class:`retrieval_geolocator.RetrievalGeolocator` unchanged and decides which
one's answer leads, based on retrieval's top-1 cosine similarity.

Why similarity and not confidence
---------------------------------
Retrieval's confidence measures agreement among the returned neighbours, so a
tiny gallery can produce strong agreement while every neighbour is wrong -- an
Alaska query scored 86.1% confidence at 0.523 top similarity and named the wrong
region. Confidence therefore cannot gate this decision. Raw top-1 cosine
similarity against the closest reference photo can, but only above a
surprisingly high cut.

The calibrated cut (see finetune/pick_threshold.py)
---------------------------------------------------
Measured on 1000 globally diverse images the gallery does not cover (negative
class) and 24 in-coverage images the gallery has not memorised (positive class):

===================  ======  ========  ======  ======
population           min     median    p99     max
===================  ======  ========  ======  ======
not covered (n=1000) 0.487   0.672     0.777   0.803
in coverage (n=24)   0.632   0.851     --      0.962
===================  ======  ========  ======  ======

Out of coverage, similarity carries no quality signal at all: retrieval's median
error is ~10,000 km in every bucket from 0.50 to 0.80, and it beats GeoCLIP on
only ~9-11% of images regardless. The two populations separate only at the top
of the range, because no out-of-coverage image in 1000 exceeded 0.803.

``DEFAULT_TRUST_THRESHOLD`` is therefore 0.825 -- above the observed
out-of-coverage ceiling, below the in-coverage median. At that cut the external
set contributes zero false accepts (median error unchanged, +0.00 km) while 62.5%
of in-coverage images are routed to retrieval, taking their median error from
160.19 km to 0.36 km.

A 0.6-0.65 cut, which looks reasonable a priori, would accept 903 of those 1000
external images and push their median error from 1292 km to 9787 km. Note this
is a different quantity from ``aggregator.DEFAULT_WEAK_SIMILARITY_THRESHOLD``
(also 0.6), which re-ranks clusters *within* a retrieval result and does not
choose between predictors.

Recalibrated against benchmark-gallery v2 (3314 images, ~1.9x v1, adding Africa,
South America and much more of Asia). The threshold did **not** move, but not
because nothing changed -- two opposing shifts cancelled:

===================  =============  =============
population           v1 (1719 img)  v2 (3314 img)
===================  =============  =============
out-of-coverage max  0.803          0.830
out-of-coverage mean 0.668          0.704
in-coverage median   0.851          0.863
===================  =============  =============

A larger gallery gives every query more chances at a spuriously similar photo, so
the out-of-coverage ceiling *rises* with gallery size -- pushing the threshold up.
But genuine in-coverage matches also score higher, pulling it back down. At 0.825
the external set contributes 1 accept in 1000 (median unchanged, +0.00 km) while
in-coverage acceptance improved from 62.5% to 73.1%.

The rising false-accept ceiling is the thing to watch: it is a property of gallery
*size*, not of which regions were added, so this threshold must be re-derived on
every significant gallery growth rather than treated as settled.

Caveat: the in-coverage side now rests on 93 images (was 24), drawn from the new
footprint as well as the old. They are the only images available that sit inside
gallery coverage without being gallery entries, and were excluded by the
pipeline's blur/indoor/tile-cap filters, so they are if anything harder than
typical -- which makes 0.825 conservative rather than optimistic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from PIL import Image

logger = logging.getLogger(__name__)

# Above this top-1 cosine similarity, retrieval's answer leads. Calibrated, not
# assumed -- see the module docstring.
DEFAULT_TRUST_THRESHOLD = 0.825

# Below this, retrieval's answer is dropped rather than offered as an
# alternative. Measured on the external set: below 0.65 retrieval beats GeoCLIP
# on 10.6% of images with a median error of 10,048 km, which is noise, not a
# useful second opinion.
DEFAULT_MENTION_FLOOR = 0.65

DEFAULT_TOP_K = 5


@dataclass
class EnsembleCandidate:
    """One ranked coordinate, labelled with the predictor that produced it."""

    lat: float
    lon: float
    score: float
    source: str
    """``"retrieval"`` or ``"geoclip"``."""
    role: str
    """``"primary"`` or ``"alternative"``."""
    top_similarity: float | None = None
    """Retrieval's cosine similarity. ``None`` for GeoCLIP, whose softmax score
    has no absolute scale and must not be compared against it."""
    source_image_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "score": round(self.score, 6),
            "source": self.source,
            "role": self.role,
        }
        if self.top_similarity is not None:
            payload["top_similarity"] = round(self.top_similarity, 4)
        if self.source_image_id is not None:
            payload["source_image_id"] = self.source_image_id
        return payload


@dataclass
class EnsembleDecision:
    """Why the ensemble chose the predictor it did, for transparency."""

    routed_to: str
    top_similarity: float | None
    threshold: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "routed_to": self.routed_to,
            "top_similarity": None if self.top_similarity is None else round(self.top_similarity, 4),
            "trust_threshold": self.threshold,
            "reason": self.reason,
        }


class EnsembleGeolocator:
    """Compose GeoCLIP and retrieval into one ranked answer.

    Neither underlying predictor is modified; both are called through their
    existing ``predict_batch`` interface and their outputs are merged here.
    """

    def __init__(
        self,
        geoclip: Any,
        retrieval: Any,
        trust_threshold: float = DEFAULT_TRUST_THRESHOLD,
        mention_floor: float = DEFAULT_MENTION_FLOOR,
    ) -> None:
        """
        Args:
            geoclip: A built :class:`geolocator.GeoLocator`.
            retrieval: A built :class:`retrieval_geolocator.RetrievalGeolocator`.
            trust_threshold: Top-1 cosine similarity at or above which
                retrieval's answer becomes primary.
            mention_floor: Below this similarity, retrieval's answer is dropped
                entirely rather than listed as an alternative.
        """
        if not 0.0 <= mention_floor <= trust_threshold <= 1.0:
            raise ValueError(
                f"Require 0 <= mention_floor ({mention_floor}) <= "
                f"trust_threshold ({trust_threshold}) <= 1"
            )
        self.geoclip = geoclip
        self.retrieval = retrieval
        self.trust_threshold = trust_threshold
        self.mention_floor = mention_floor

    # -- decision ---------------------------------------------------------

    def _merge(
        self,
        geo_candidates: Sequence[Any],
        ret_candidates: Sequence[Any],
        top_k: int,
    ) -> tuple[list[EnsembleCandidate], EnsembleDecision]:
        """Rank one image's candidates from both predictors."""
        top_similarity = float(ret_candidates[0].score) if ret_candidates else None

        trust_retrieval = top_similarity is not None and top_similarity >= self.trust_threshold
        mention_retrieval = top_similarity is not None and top_similarity >= self.mention_floor

        def as_ensemble(candidate: Any, source: str, role: str) -> EnsembleCandidate:
            return EnsembleCandidate(
                lat=float(candidate.lat),
                lon=float(candidate.lon),
                score=float(candidate.score),
                source=source,
                role=role,
                top_similarity=float(candidate.score) if source == "retrieval" else None,
                source_image_id=getattr(candidate, "source_image_id", None),
            )

        ranked: list[EnsembleCandidate] = []

        if trust_retrieval:
            ranked.append(as_ensemble(ret_candidates[0], "retrieval", "primary"))
            # GeoCLIP's best stays visible as the independent second opinion.
            if geo_candidates:
                ranked.append(as_ensemble(geo_candidates[0], "geoclip", "alternative"))
            # Then retrieval's remaining neighbours, then GeoCLIP's.
            ranked.extend(as_ensemble(c, "retrieval", "alternative") for c in ret_candidates[1:])
            ranked.extend(as_ensemble(c, "geoclip", "alternative") for c in geo_candidates[1:])
            decision = EnsembleDecision(
                routed_to="retrieval",
                top_similarity=top_similarity,
                threshold=self.trust_threshold,
                reason=(
                    f"top_similarity {top_similarity:.4f} >= {self.trust_threshold} "
                    "(gallery has a strong reference match)"
                ),
            )
        else:
            if geo_candidates:
                ranked.append(as_ensemble(geo_candidates[0], "geoclip", "primary"))
            if mention_retrieval:
                # Kept, but explicitly ranked below GeoCLIP: in this band
                # retrieval beat GeoCLIP on only ~9-11% of external images.
                ranked.append(as_ensemble(ret_candidates[0], "retrieval", "alternative"))
            ranked.extend(as_ensemble(c, "geoclip", "alternative") for c in geo_candidates[1:])

            if top_similarity is None:
                reason = "retrieval returned no candidate"
            elif mention_retrieval:
                reason = (
                    f"top_similarity {top_similarity:.4f} < {self.trust_threshold}; "
                    "retrieval offered as a low-confidence alternative only"
                )
            else:
                reason = (
                    f"top_similarity {top_similarity:.4f} < {self.mention_floor}; "
                    "retrieval dropped as noise"
                )
            decision = EnsembleDecision(
                routed_to="geoclip",
                top_similarity=top_similarity,
                threshold=self.trust_threshold,
                reason=reason,
            )

        return ranked[: top_k + 1], decision

    # -- prediction -------------------------------------------------------

    def predict_batch(
        self, images: Sequence[Image.Image], top_k: int = DEFAULT_TOP_K
    ) -> list[list[EnsembleCandidate]]:
        """Run both predictors over a batch and return merged ranked candidates."""
        if not images:
            return []

        geo_batch = self.geoclip.predict_batch(images, top_k=top_k)
        ret_batch = self.retrieval.predict_batch(images, top_k=top_k)

        merged: list[list[EnsembleCandidate]] = []
        for geo_candidates, ret_candidates in zip(geo_batch, ret_batch):
            ranked, decision = self._merge(geo_candidates, ret_candidates, top_k)
            logger.debug("ensemble routed to %s: %s", decision.routed_to, decision.reason)
            merged.append(ranked)
        return merged

    def predict_image(self, image: Image.Image, top_k: int = DEFAULT_TOP_K) -> list[EnsembleCandidate]:
        """Run both predictors over one image."""
        return self.predict_batch([image], top_k=top_k)[0]

    def predict_with_decision(
        self, images: Sequence[Image.Image], top_k: int = DEFAULT_TOP_K
    ) -> list[tuple[list[EnsembleCandidate], EnsembleDecision]]:
        """Like :meth:`predict_batch`, but also returns the routing rationale."""
        if not images:
            return []

        geo_batch = self.geoclip.predict_batch(images, top_k=top_k)
        ret_batch = self.retrieval.predict_batch(images, top_k=top_k)
        return [
            self._merge(geo_candidates, ret_candidates, top_k)
            for geo_candidates, ret_candidates in zip(geo_batch, ret_batch)
        ]

    def predict_frames(
        self,
        frames: Sequence[Any],
        top_k: int = DEFAULT_TOP_K,
        batch_size: int = 32,
        progress: bool = False,
    ) -> list[Any]:
        """Predict for video frames, batching to bound peak memory.

        Delegates frame handling to the same shape the other predictors use so
        the video pipeline needs no ensemble-specific branch.
        """
        from viewfinder.aggregator import FramePrediction

        predictions: list[FramePrediction] = []
        for start in range(0, len(frames), batch_size):
            chunk = list(frames[start : start + batch_size])
            batch = self.predict_batch([frame.image for frame in chunk], top_k=top_k)
            for frame, candidates in zip(chunk, batch):
                predictions.append(
                    FramePrediction(
                        index=frame.index,
                        timestamp=frame.timestamp,
                        blur_score=frame.blur_score,
                        candidates=list(candidates),
                    )
                )
        return predictions
