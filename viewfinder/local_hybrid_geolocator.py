"""Fail-closed fusion of local OCR, GeoNames, Commons retrieval, and PLONK prior."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

from .aggregator import Candidate, FramePrediction, haversine_km
from .hybrid_geolocator import extract_exif_gps
from .local_gazetteer import GeoNamesResolver, PlaceMatch, normalize_name
from .local_landmarks import LandmarkMatch, LandmarkRetriever
from .local_ocr import OcrLine, PaddleSceneText
from .progress import report as _report_progress


LOGGER = logging.getLogger(__name__)
OCR_LANDMARK_AGREEMENT_KM = 25.0
MIN_OCR_CONFIDENCE = 0.70
MIN_LANDMARK_PAIR_SIMILARITY = 0.68
MIN_REVIEW_LANDMARK_SIMILARITY = 0.72
MIN_INDEPENDENT_REFERENCES = 3
MIN_INDEPENDENT_ARTISTS = 2


@dataclass
class LocalEvidenceAssessment:
    decision: str
    reliability_tier: str
    reason: str
    automation_accepted: bool
    state_country_accepted: bool
    evidence_agreement_radius_km: float | None
    selected_source: str
    exif_gps: dict[str, float] | None = None
    ocr_lines: list[OcrLine] = field(default_factory=list)
    place_matches: list[PlaceMatch] = field(default_factory=list)
    perceptual_matches: list[LandmarkMatch] = field(default_factory=list)
    landmark_matches: list[LandmarkMatch] = field(default_factory=list)
    provider_status: dict[str, str] = field(default_factory=dict)
    prior_agreement_km: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reliability_tier": self.reliability_tier,
            "reason": self.reason,
            "automation_accepted": self.automation_accepted,
            "state_country_accepted": self.state_country_accepted,
            "evidence_agreement_radius_km": self.evidence_agreement_radius_km,
            "selected_source": self.selected_source,
            "exif_gps": self.exif_gps,
            "ocr_lines": [line.as_dict() for line in self.ocr_lines],
            "place_matches": [place.as_dict() for place in self.place_matches],
            "perceptual_matches": [
                match.as_dict() for match in self.perceptual_matches
            ],
            "landmark_matches": [
                match.as_dict() for match in self.landmark_matches
            ],
            "provider_status": self.provider_status,
            "prior_agreement_km": self.prior_agreement_km,
            "warning": (
                "Local evidence can support or reject a result, but it cannot make "
                "arbitrary image geolocation infallible. PLONK is reported only as "
                "a prior and never authorizes a prediction by itself."
            ),
        }


@dataclass(frozen=True)
class LocalHybridProvenance:
    data_root: str
    visual_prior: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": "fully-local-evidence-hybrid",
            "data_root": self.data_root,
            "visual_prior": self.visual_prior,
            "network_at_runtime": False,
            "evidence_order": [
                "embedded_exif_gps",
                "near_exact_commons_perceptual_match",
                "paddleocr_geonames_plus_commons_landmark",
                "single_channel_review",
                "plonk_prior_abstain",
            ],
        }


def _ocr_confidence(place: PlaceMatch, lines: Sequence[OcrLine]) -> float:
    query = normalize_name(place.query)
    scores = [
        line.score
        for line in lines
        if query and query in normalize_name(line.text)
    ]
    return max(scores, default=0.0)


def _best_ocr_landmark_pair(
    places: Sequence[PlaceMatch],
    groups: Sequence[Sequence[LandmarkMatch]],
    lines: Sequence[OcrLine],
) -> tuple[PlaceMatch, list[LandmarkMatch]] | None:
    candidates: list[tuple[PlaceMatch, list[LandmarkMatch]]] = []
    for place in places:
        if place.name_score < 0.90 or _ocr_confidence(place, lines) < MIN_OCR_CONFIDENCE:
            continue
        for group in groups:
            credible = _credible_landmark_group(
                group,
                min_similarity=MIN_LANDMARK_PAIR_SIMILARITY,
            )
            if credible is None:
                continue
            if haversine_km(place.lat, place.lon, credible[0].lat, credible[0].lon) <= (
                OCR_LANDMARK_AGREEMENT_KM
            ):
                candidates.append((place, credible))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda pair: (
            pair[0].name_score,
            _ocr_confidence(pair[0], lines),
            max(item.similarity for item in pair[1]),
            len(pair[1]),
        ),
    )


def _credible_landmark_group(
    group: Sequence[LandmarkMatch],
    *,
    min_similarity: float,
) -> list[LandmarkMatch] | None:
    ranked = sorted(group, key=lambda item: item.similarity, reverse=True)
    unique_pages = {item.page_id for item in ranked}
    artists = {
        normalize_name(item.artist)
        for item in ranked
        if normalize_name(item.artist)
    }
    if (
        len(unique_pages) < MIN_INDEPENDENT_REFERENCES
        or len(artists) < MIN_INDEPENDENT_ARTISTS
        or ranked[MIN_INDEPENDENT_REFERENCES - 1].similarity < min_similarity
    ):
        return None
    anchor = ranked[0]
    if any(
        haversine_km(anchor.lat, anchor.lon, item.lat, item.lon)
        > OCR_LANDMARK_AGREEMENT_KM
        for item in ranked[:MIN_INDEPENDENT_REFERENCES]
    ):
        return None
    return ranked


def _dominant_perceptual(
    matches: Sequence[LandmarkMatch],
) -> list[LandmarkMatch] | None:
    groups: dict[int, list[LandmarkMatch]] = {}
    for match in matches:
        groups.setdefault(match.geoname_id, []).append(match)
    if not groups:
        return None
    ranked = sorted(
        groups.values(),
        key=lambda group: (
            -min(item.phash_distance or 0 for item in group),
            len(group),
        ),
        reverse=True,
    )
    best = ranked[0]
    best_distance = min(item.phash_distance or 0 for item in best)
    # One extremely close pHash or two independent Commons files is required.
    if best_distance <= 2 or (
        len({item.page_id for item in best}) >= 2
        and max(item.phash_distance or 99 for item in best) <= 6
    ):
        if len(ranked) == 1 or len(ranked[1]) < len(best):
            return best
    return None


class LocalHybridGeolocator:
    """Select local observable evidence and quarantine visual-only guesses."""

    def __init__(
        self,
        input_path: str | Path,
        *,
        data_root: str | Path = "local-data",
        ocr_provider: Any | None = None,
        resolver: Any | None = None,
        landmark_provider: Any | None = None,
        visual_prior_factory: Callable[[], Any] | None = None,
        visual_prior_name: str = "PLONK OSV-5M",
    ) -> None:
        self.input_path = Path(input_path)
        self.data_root = Path(data_root).expanduser().resolve()
        self.ocr_provider = ocr_provider
        self.resolver = resolver
        self.landmark_provider = landmark_provider
        self.visual_prior_factory = visual_prior_factory
        self.visual_prior_name = visual_prior_name
        self._visual_prior: Any | None = None
        self._suppress_prior = False
        self._exif_checked = False
        self._exif_value: tuple[float, float] | None = None
        self.last_assessment: LocalEvidenceAssessment | None = None
        self.frame_assessments: list[LocalEvidenceAssessment] = []
        self.evidence_frame_predictions: list[FramePrediction] = []
        self.provenance = LocalHybridProvenance(
            data_root=str(self.data_root),
            visual_prior=visual_prior_name,
        )

    def _prior(self) -> Any:
        if self._visual_prior is None:
            if self.visual_prior_factory is None:
                raise RuntimeError("No visual-prior factory is configured")
            self._visual_prior = self.visual_prior_factory()
        return self._visual_prior

    def _local_evidence(
        self, image: Image.Image
    ) -> tuple[list[OcrLine], list[PlaceMatch], dict[str, Any], dict[str, str]]:
        status: dict[str, str] = {}
        lines: list[OcrLine] = []
        places: list[PlaceMatch] = []
        landmark_analysis: dict[str, Any] = {
            "perceptual_matches": [],
            "visual_matches": [],
            "groups": [],
        }
        resolver: Any | None = None
        try:
            if self.ocr_provider is None:
                self.ocr_provider = PaddleSceneText(self.data_root / "ocr-models")
            ocr = self.ocr_provider
            lines = list(ocr.analyze(image))
            status["paddleocr"] = f"ok: {len(lines)} line(s)"
        except Exception as exc:
            status["paddleocr"] = f"unavailable: {exc}"
        try:
            if self.resolver is None:
                self.resolver = GeoNamesResolver(
                    self.data_root / "geonames" / "geonames.sqlite3"
                )
            resolver = self.resolver
            places = list(resolver.resolve_lines([line.text for line in lines]))
            status["geonames_osm"] = f"ok: {len(places)} match(es)"
        except Exception as exc:
            status["geonames_osm"] = f"unavailable: {exc}"
        try:
            if self.landmark_provider is None:
                self.landmark_provider = LandmarkRetriever(
                    self.data_root / "landmarks" / "gallery"
                )
            landmarks = self.landmark_provider
            landmark_analysis = landmarks.analyze(image)
            status["wikimedia_landmarks"] = (
                f"ok: gallery={landmark_analysis.get('gallery_size', 0)}"
            )
        except Exception as exc:
            status["wikimedia_landmarks"] = f"unavailable: {exc}"
        groups = list(landmark_analysis.get("groups") or [])
        if resolver is not None and hasattr(resolver, "resolve_lines_near"):
            seen = {(place.source, place.source_id) for place in places}
            for group in groups[:5]:
                if not group:
                    continue
                nearby = resolver.resolve_lines_near(
                    [line.text for line in lines],
                    group[0].lat,
                    group[0].lon,
                    radius_km=OCR_LANDMARK_AGREEMENT_KM,
                )
                for place in nearby:
                    key = (place.source, place.source_id)
                    if key not in seen:
                        seen.add(key)
                        places.append(place)
        return lines, places, landmark_analysis, status

    def predict_image(self, image: Image.Image, top_k: int = 5) -> list[Candidate]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not self._exif_checked:
            try:
                self._exif_value = extract_exif_gps(self.input_path)
            except Exception as exc:
                LOGGER.debug("Could not parse EXIF GPS: %s", exc)
                self._exif_value = None
            self._exif_checked = True
        exif = self._exif_value
        if exif is not None:
            lat, lon = exif
            self.last_assessment = LocalEvidenceAssessment(
                decision="emit",
                reliability_tier="embedded_gps_metadata",
                reason=(
                    "The file contains valid embedded GPS. It is direct metadata "
                    "but is editable and not cryptographically authenticated."
                ),
                automation_accepted=True,
                state_country_accepted=True,
                evidence_agreement_radius_km=None,
                selected_source="exif_gps",
                exif_gps={"latitude": round(lat, 6), "longitude": round(lon, 6)},
            )
            return [Candidate(lat, lon, 1.0)]

        lines, places, landmark_analysis, status = self._local_evidence(image)
        perceptual = list(landmark_analysis.get("perceptual_matches") or [])
        visual = list(landmark_analysis.get("visual_matches") or [])
        groups = list(landmark_analysis.get("groups") or [])

        exact_group = _dominant_perceptual(perceptual)
        if exact_group:
            best = min(exact_group, key=lambda item: item.phash_distance or 0)
            self.last_assessment = LocalEvidenceAssessment(
                decision="emit",
                reliability_tier="near_exact_attributed_reference",
                reason=(
                    "The query is a near-exact perceptual match to an attributed, "
                    "geographically labeled Wikimedia Commons reference."
                ),
                automation_accepted=True,
                state_country_accepted=True,
                evidence_agreement_radius_km=None,
                selected_source="commons_perceptual_match",
                ocr_lines=lines,
                place_matches=places,
                perceptual_matches=exact_group,
                landmark_matches=visual[:10],
                provider_status=status,
            )
            return [Candidate(best.lat, best.lon, 1.0)]

        pair = _best_ocr_landmark_pair(places, groups, lines)
        if pair:
            place, group = pair
            self.last_assessment = LocalEvidenceAssessment(
                decision="emit",
                reliability_tier="corroborated_text_and_landmark",
                reason=(
                    "High-confidence local OCR resolved through the offline "
                    "gazetteer and an independent Commons image-retrieval group "
                    f"agree within {OCR_LANDMARK_AGREEMENT_KM:g} km."
                ),
                automation_accepted=True,
                state_country_accepted=True,
                evidence_agreement_radius_km=OCR_LANDMARK_AGREEMENT_KM,
                selected_source="paddleocr_geonames_plus_commons",
                ocr_lines=lines,
                place_matches=places,
                perceptual_matches=perceptual,
                landmark_matches=group,
                provider_status=status,
            )
            return [Candidate(place.lat, place.lon, 1.0)]

        strong_group = next(
            (
                credible
                for group in groups
                if (
                    credible := _credible_landmark_group(
                        group,
                        min_similarity=MIN_REVIEW_LANDMARK_SIMILARITY,
                    )
                )
                is not None
            ),
            None,
        )
        if strong_group:
            best = strong_group[0]
            self.last_assessment = LocalEvidenceAssessment(
                decision="review",
                reliability_tier="multi_reference_landmark",
                reason=(
                    "At least three local Commons references from multiple credited "
                    "artists support one landmark, but this similarity rule is not "
                    "calibrated strongly enough for automatic use."
                ),
                automation_accepted=False,
                state_country_accepted=True,
                evidence_agreement_radius_km=None,
                selected_source="commons_visual_retrieval",
                ocr_lines=lines,
                place_matches=places,
                perceptual_matches=perceptual,
                landmark_matches=list(strong_group),
                provider_status=status,
            )
            return [Candidate(best.lat, best.lon, best.similarity)]

        strong_places = [
            place
            for place in places
            if place.name_score >= 0.95
            and _ocr_confidence(place, lines) >= MIN_OCR_CONFIDENCE
            and len(normalize_name(place.query).split()) >= 2
        ]
        if strong_places:
            best = strong_places[0]
            self.last_assessment = LocalEvidenceAssessment(
                decision="review",
                reliability_tier="offline_text_place",
                reason=(
                    "Local OCR strongly matches an offline geographic name, but "
                    "there is no independent visual landmark evidence."
                ),
                automation_accepted=False,
                state_country_accepted=False,
                evidence_agreement_radius_km=None,
                selected_source="paddleocr_geonames",
                ocr_lines=lines,
                place_matches=places,
                perceptual_matches=perceptual,
                landmark_matches=visual[:10],
                provider_status=status,
            )
            return [Candidate(best.lat, best.lon, best.name_score)]

        if self._suppress_prior:
            candidates = [Candidate(0.0, 0.0, 0.0)]
        else:
            prior = self._prior()
            candidates = prior.predict_image(image, top_k=top_k)
        self.last_assessment = LocalEvidenceAssessment(
            decision="abstain",
            reliability_tier="visual_prior_only",
            reason=(
                "No accepted local text, metadata, or landmark evidence was found. "
                f"{self.visual_prior_name} is retained only as a manual hypothesis."
            ),
            automation_accepted=False,
            state_country_accepted=False,
            evidence_agreement_radius_km=None,
            selected_source="plonk_visual_prior",
            ocr_lines=lines,
            place_matches=places,
            perceptual_matches=perceptual,
            landmark_matches=visual[:10],
            provider_status=status,
        )
        return candidates

    def predict_frames(
        self,
        frames: Sequence[Any],
        top_k: int = 5,
        batch_size: int = 1,
        progress: Any = False,
    ) -> list[FramePrediction]:
        """Run fail-closed local evidence independently over sampled video frames."""
        predictions: list[FramePrediction] = []
        evidence_predictions: list[FramePrediction] = []
        assessments: list[LocalEvidenceAssessment] = []
        accepted_coordinates: list[tuple[float, float]] = []
        accepted_assessments: list[LocalEvidenceAssessment] = []
        review_assessments: list[LocalEvidenceAssessment] = []

        self._suppress_prior = True
        try:
            for position, frame in enumerate(frames, start=1):
                candidates = self.predict_image(frame.image, top_k=top_k)
                prediction = FramePrediction(
                    candidates=candidates,
                    frame_index=frame.index,
                    timestamp=frame.timestamp,
                    blur_score=frame.blur_score,
                )
                predictions.append(prediction)
                assessment = self.last_assessment
                if assessment is not None:
                    assessments.append(assessment)
                    if assessment.decision != "abstain":
                        evidence_predictions.append(prediction)
                    if assessment.automation_accepted and candidates:
                        accepted_assessments.append(assessment)
                        accepted_coordinates.append((candidates[0].lat, candidates[0].lon))
                    elif assessment.decision == "review":
                        review_assessments.append(assessment)
                _report_progress(
                    progress,
                    position,
                    len(frames),
                    logger=LOGGER,
                    message="Analyzed local evidence for %d/%d frames",
                )
        finally:
            self._suppress_prior = False

        if not evidence_predictions and frames:
            prior = self._prior()
            if hasattr(prior, "predict_frames"):
                predictions = prior.predict_frames(
                    frames,
                    top_k=top_k,
                    batch_size=batch_size,
                    progress=progress,
                )
            else:
                predictions = [
                    FramePrediction(
                        candidates=prior.predict_image(frame.image, top_k=top_k),
                        frame_index=frame.index,
                        timestamp=frame.timestamp,
                        blur_score=frame.blur_score,
                    )
                    for frame in frames
                ]

        self.frame_assessments = assessments
        self.evidence_frame_predictions = evidence_predictions
        if accepted_assessments:
            spread = max(
                (
                    haversine_km(lat1, lon1, lat2, lon2)
                    for index, (lat1, lon1) in enumerate(accepted_coordinates)
                    for lat2, lon2 in accepted_coordinates[index + 1 :]
                ),
                default=0.0,
            )
            representative = accepted_assessments[0]
            if spread <= OCR_LANDMARK_AGREEMENT_KM:
                self.last_assessment = LocalEvidenceAssessment(
                    decision="emit",
                    reliability_tier="corroborated_video_frame_evidence",
                    reason=(
                        f"{len(accepted_assessments)} sampled frame(s) contained "
                        "accepted local evidence and their coordinates agree within "
                        f"{spread:.1f} km."
                    ),
                    automation_accepted=True,
                    state_country_accepted=True,
                    evidence_agreement_radius_km=round(spread, 3),
                    selected_source="video_frame_local_evidence",
                    ocr_lines=representative.ocr_lines,
                    place_matches=representative.place_matches,
                    perceptual_matches=representative.perceptual_matches,
                    landmark_matches=representative.landmark_matches,
                    provider_status={
                        **representative.provider_status,
                        "video_frames": (
                            f"{len(accepted_assessments)} accepted/"
                            f"{len(frames)} analyzed"
                        ),
                    },
                )
            else:
                self.last_assessment = LocalEvidenceAssessment(
                    decision="abstain",
                    reliability_tier="conflicting_video_evidence",
                    reason=(
                        "Accepted frame-level signals disagree geographically "
                        f"(maximum separation {spread:.1f} km)."
                    ),
                    automation_accepted=False,
                    state_country_accepted=False,
                    evidence_agreement_radius_km=round(spread, 3),
                    selected_source="conflicting_video_frames",
                    provider_status={
                        "video_frames": (
                            f"{len(accepted_assessments)} conflicting/"
                            f"{len(frames)} analyzed"
                        )
                    },
                )
        elif review_assessments:
            representative = review_assessments[0]
            self.last_assessment = LocalEvidenceAssessment(
                decision="review",
                reliability_tier="video_frame_review",
                reason=(
                    f"{len(review_assessments)} sampled frame(s) produced review-level "
                    "local evidence, but none met the automatic acceptance gate."
                ),
                automation_accepted=False,
                state_country_accepted=all(
                    item.state_country_accepted for item in review_assessments
                ),
                evidence_agreement_radius_km=None,
                selected_source="video_frame_local_evidence",
                ocr_lines=representative.ocr_lines,
                place_matches=representative.place_matches,
                perceptual_matches=representative.perceptual_matches,
                landmark_matches=representative.landmark_matches,
                provider_status={
                    **representative.provider_status,
                    "video_frames": (
                        f"{len(review_assessments)} review/"
                        f"{len(frames)} analyzed"
                    ),
                },
            )
        elif assessments:
            representative = assessments[-1]
            self.last_assessment = LocalEvidenceAssessment(
                decision="abstain",
                reliability_tier="video_visual_prior_only",
                reason=(
                    f"No accepted metadata, OCR, or landmark evidence was found in "
                    f"{len(frames)} sampled frame(s). Visual-model outputs remain "
                    "unauthorized hypotheses."
                ),
                automation_accepted=False,
                state_country_accepted=False,
                evidence_agreement_radius_km=None,
                selected_source="plonk_video_visual_prior",
                ocr_lines=representative.ocr_lines,
                place_matches=representative.place_matches,
                perceptual_matches=representative.perceptual_matches,
                landmark_matches=representative.landmark_matches,
                provider_status={
                    **representative.provider_status,
                    "video_frames": f"0 accepted/{len(frames)} analyzed",
                },
            )
        return predictions
