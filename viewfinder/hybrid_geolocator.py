"""Evidence-first hybrid geolocation.

The visual models in this repository are useful hypotheses, but the benchmark
shows they are not reliable enough to authorize a location. This module puts
observable evidence ahead of learned visual priors:

1. Embedded EXIF GPS (structured metadata).
2. Cloud landmark coordinates and OCR-resolved Google Places (opt-in).
3. PLONK OSV-5M as a weak fallback that always causes abstention.

Online evidence is never enabled implicitly because it uploads the image to a
third-party service and can incur API charges.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from PIL import ExifTags, Image

from .aggregator import Candidate, haversine_km


logger = logging.getLogger(__name__)

LANDMARK_ACCEPT_SCORE = 0.90
LANDMARK_CORROBORATION_SCORE = 0.50
EVIDENCE_AGREEMENT_KM = 25.0
PLACE_NAME_TOKEN_OVERLAP = 0.80
MAX_OCR_QUERIES = 4


class HybridError(RuntimeError):
    """A malformed evidence response or unavailable required provider."""


@dataclass(frozen=True)
class LandmarkEvidence:
    name: str
    lat: float
    lon: float
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "latitude": round(self.lat, 6),
            "longitude": round(self.lon, 6),
            "provider_score": round(self.score, 6),
        }


@dataclass(frozen=True)
class PlaceEvidence:
    query: str
    name: str
    address: str
    lat: float
    lon: float
    place_id: str = ""
    token_overlap: float = 0.0
    query_source: str = "ocr"

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "name": self.name,
            "address": self.address,
            "latitude": round(self.lat, 6),
            "longitude": round(self.lon, 6),
            "place_id": self.place_id,
            "name_token_overlap": round(self.token_overlap, 4),
            "query_source": self.query_source,
        }


@dataclass(frozen=True)
class WebEvidence:
    label: str
    score: float
    kind: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": round(self.score, 6),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class WebPageEvidence:
    title: str
    url: str

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url}


@dataclass
class VisionAnalysis:
    landmarks: list[LandmarkEvidence] = field(default_factory=list)
    detected_text: str = ""
    web_entities: list[WebEvidence] = field(default_factory=list)
    matching_pages: list[WebPageEvidence] = field(default_factory=list)


@dataclass
class HybridAssessment:
    decision: str
    reliability_tier: str
    reason: str
    automation_accepted: bool
    state_country_accepted: bool
    evidence_agreement_radius_km: float | None
    selected_source: str
    exif_gps: dict[str, float] | None = None
    landmarks: list[LandmarkEvidence] = field(default_factory=list)
    detected_text: str = ""
    web_entities: list[WebEvidence] = field(default_factory=list)
    matching_pages: list[WebPageEvidence] = field(default_factory=list)
    places: list[PlaceEvidence] = field(default_factory=list)
    provider_status: dict[str, str] = field(default_factory=dict)

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
            "landmarks": [item.as_dict() for item in self.landmarks],
            "detected_text": self.detected_text,
            "web_entities": [item.as_dict() for item in self.web_entities],
            "matching_pages": [item.as_dict() for item in self.matching_pages],
            "places": [item.as_dict() for item in self.places],
            "provider_status": self.provider_status,
            "warning": (
                "Only valid embedded GPS or accepted landmark/text/web evidence can "
                "authorize a result. A visual-model fallback is a hypothesis and "
                "always produces an abstention."
            ),
        }


@dataclass(frozen=True)
class HybridProvenance:
    visual_fallback: str
    online_evidence_enabled: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": "evidence-first-hybrid",
            "visual_fallback": self.visual_fallback,
            "online_evidence_enabled": self.online_evidence_enabled,
            "evidence_order": [
                "exif_gps",
                "cloud_landmark_plus_ocr_place",
                "web_image_matches_plus_mapped_place",
                "high_score_cloud_landmark_review",
                "high_overlap_text_place_review",
                "visual_fallback_abstain",
            ],
        }


def _rational(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise HybridError(f"Invalid EXIF GPS value: {value!r}") from exc


def _degrees(values: Sequence[Any], reference: str) -> float:
    if len(values) != 3:
        raise HybridError("EXIF GPS coordinate must contain degrees, minutes, seconds")
    degrees, minutes, seconds = (_rational(value) for value in values)
    coordinate = degrees + minutes / 60.0 + seconds / 3600.0
    if reference.upper() in {"S", "W"}:
        coordinate = -coordinate
    return coordinate


def _reference(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="ignore")
    return str(value)


def gps_from_mapping(gps: Mapping[str, Any]) -> tuple[float, float] | None:
    """Decode a GPS IFD whose keys use Pillow's human-readable tag names."""
    required = ("GPSLatitude", "GPSLatitudeRef", "GPSLongitude", "GPSLongitudeRef")
    if any(key not in gps for key in required):
        return None
    lat = _degrees(gps["GPSLatitude"], _reference(gps["GPSLatitudeRef"]))
    lon = _degrees(gps["GPSLongitude"], _reference(gps["GPSLongitudeRef"]))
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise HybridError("EXIF GPS coordinate is outside valid latitude/longitude bounds")
    return lat, lon


def extract_exif_gps(path: str | Path) -> tuple[float, float] | None:
    """Read embedded GPS without changing image pixels or orientation."""
    with Image.open(path) as image:
        exif = image.getexif()
        if not exif:
            return None
        gps_tag = next(
            (tag for tag, name in ExifTags.TAGS.items() if name == "GPSInfo"),
            34853,
        )
        try:
            gps_ifd = exif.get_ifd(gps_tag)
        except (AttributeError, KeyError, TypeError):
            raw = exif.get(gps_tag)
            gps_ifd = raw if isinstance(raw, Mapping) else {}
        named = {
            ExifTags.GPSTAGS.get(key, str(key)): value
            for key, value in gps_ifd.items()
        }
        return gps_from_mapping(named)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[^\W_]{2,}", text.casefold(), flags=re.UNICODE)
        if token not in {"the", "and", "park", "road", "street"}
    }


def token_overlap(query: str, place_name: str) -> float:
    query_tokens = _tokens(query)
    place_tokens = _tokens(place_name)
    if not query_tokens or not place_tokens:
        return 0.0
    return len(query_tokens & place_tokens) / len(place_tokens)


def ocr_queries(text: str, limit: int = MAX_OCR_QUERIES) -> list[str]:
    """Extract conservative place-like lines instead of geocoding every word."""
    candidates: list[str] = []
    for raw in text.splitlines():
        cleaned = re.sub(r"\s+", " ", raw).strip(" -|,.;:")
        words = re.findall(r"[^\W_]+", cleaned, flags=re.UNICODE)
        if 5 <= len(cleaned) <= 100 and len(words) >= 2:
            candidates.append(cleaned)
    candidates.sort(key=lambda value: (-len(_tokens(value)), -len(value), value))
    unique: list[str] = []
    for candidate in candidates:
        if candidate.casefold() not in {item.casefold() for item in unique}:
            unique.append(candidate)
    return unique[:limit]


def web_queries(analysis: VisionAnalysis, limit: int = MAX_OCR_QUERIES) -> list[str]:
    """Extract high-information place queries from reverse-image web evidence."""
    candidates = [
        item.label.strip()
        for item in sorted(analysis.web_entities, key=lambda item: item.score, reverse=True)
        if item.score >= 0.40 and len(_tokens(item.label)) >= 2
    ]
    for page in analysis.matching_pages:
        for segment in re.split(r"\s+[|—–-]\s+", page.title):
            cleaned = re.sub(r"\s+", " ", segment).strip()
            if 5 <= len(cleaned) <= 100 and 2 <= len(_tokens(cleaned)) <= 12:
                candidates.append(cleaned)

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique[:limit]


def independent_web_domains(pages: Sequence[WebPageEvidence]) -> set[str]:
    return {
        hostname.removeprefix("www.")
        for page in pages
        if (hostname := (urlparse(page.url).hostname or "").casefold())
    }


class GoogleVisionEvidenceProvider:
    """Google Cloud Vision landmark, OCR, and web-match adapter using ADC."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                from google.cloud import vision
            except ImportError as exc:
                raise HybridError(
                    "Online evidence needs google-cloud-vision. Install "
                    "viewfinder/requirements-hybrid.txt."
                ) from exc
            try:
                client = vision.ImageAnnotatorClient()
            except Exception as exc:
                raise HybridError(
                    "Google Cloud Vision credentials are unavailable. Run "
                    "'gcloud auth application-default login' or configure ADC."
                ) from exc
        self.client = client

    def analyze(self, path: str | Path) -> VisionAnalysis:
        content = Path(path).read_bytes()
        try:
            landmark_response = self.client.landmark_detection(image={"content": content})
            text_response = self.client.text_detection(image={"content": content})
            web_response = self.client.web_detection(image={"content": content})
        except Exception as exc:
            raise HybridError(f"Google Cloud Vision request failed: {exc}") from exc

        for response in (landmark_response, text_response, web_response):
            error = getattr(getattr(response, "error", None), "message", "")
            if error:
                raise HybridError(f"Google Cloud Vision returned: {error}")

        landmarks: list[LandmarkEvidence] = []
        for annotation in getattr(landmark_response, "landmark_annotations", ()):
            locations = getattr(annotation, "locations", ())
            if not locations:
                continue
            lat_lng = locations[0].lat_lng
            landmarks.append(
                LandmarkEvidence(
                    name=str(annotation.description),
                    lat=float(lat_lng.latitude),
                    lon=float(lat_lng.longitude),
                    score=float(annotation.score),
                )
            )
        annotations = getattr(text_response, "text_annotations", ())
        detected_text = str(annotations[0].description) if annotations else ""
        web = getattr(web_response, "web_detection", None)
        web_entities = [
            WebEvidence(
                label=str(entity.description),
                score=float(entity.score),
                kind="entity",
            )
            for entity in getattr(web, "web_entities", ())
            if getattr(entity, "description", "")
        ]
        web_entities.extend(
            WebEvidence(label=str(label.label), score=0.5, kind="best_guess")
            for label in getattr(web, "best_guess_labels", ())
            if getattr(label, "label", "")
        )
        matching_pages = [
            WebPageEvidence(
                title=str(getattr(page, "page_title", "") or ""),
                url=str(getattr(page, "url", "") or ""),
            )
            for page in getattr(web, "pages_with_matching_images", ())
            if getattr(page, "url", "")
        ]
        return VisionAnalysis(
            landmarks=landmarks,
            detected_text=detected_text,
            web_entities=web_entities,
            matching_pages=matching_pages,
        )


class GooglePlacesEvidenceProvider:
    """Resolve conservative OCR queries with Places Text Search (New)."""

    ENDPOINT = "https://places.googleapis.com/v1/places:searchText"

    def __init__(self, api_key: str | None = None, session: Any | None = None) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY", "")
        if not self.api_key:
            raise HybridError("Set GOOGLE_MAPS_API_KEY to enable OCR place resolution")
        if session is None:
            import requests

            session = requests.Session()
        self.session = session

    def search(self, query: str) -> PlaceEvidence | None:
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": (
                "places.id,places.displayName,places.formattedAddress,places.location"
            ),
        }
        try:
            response = self.session.post(
                self.ENDPOINT,
                headers=headers,
                json={"textQuery": query, "maxResultCount": 1},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise HybridError(f"Google Places search failed for {query!r}: {exc}") from exc
        places = payload.get("places") or []
        if not places:
            return None
        place = places[0]
        location = place.get("location") or {}
        display = place.get("displayName") or {}
        try:
            lat = float(location["latitude"])
            lon = float(location["longitude"])
        except (KeyError, TypeError, ValueError):
            return None
        name = str(display.get("text") or "")
        return PlaceEvidence(
            query=query,
            name=name,
            address=str(place.get("formattedAddress") or ""),
            lat=lat,
            lon=lon,
            place_id=str(place.get("id") or ""),
            token_overlap=token_overlap(query, name),
        )


class HybridGeolocator:
    """Select the strongest available evidence and abstain on visual-only input."""

    def __init__(
        self,
        input_path: str | Path,
        *,
        online_evidence: bool = False,
        vision_provider: Any | None = None,
        places_provider: Any | None = None,
        visual_fallback_factory: Callable[[], Any] | None = None,
        visual_fallback_name: str = "PLONK OSV-5M",
    ) -> None:
        self.input_path = Path(input_path)
        self.online_evidence = bool(online_evidence)
        self.vision_provider = vision_provider
        self.places_provider = places_provider
        self.visual_fallback_factory = visual_fallback_factory
        self.visual_fallback_name = visual_fallback_name
        self._visual_fallback: Any | None = None
        self.last_assessment: HybridAssessment | None = None
        self.provenance = HybridProvenance(
            visual_fallback=visual_fallback_name,
            online_evidence_enabled=self.online_evidence,
        )

    def _fallback(self) -> Any:
        if self._visual_fallback is None:
            if self.visual_fallback_factory is None:
                raise HybridError("No visual fallback factory is configured")
            self._visual_fallback = self.visual_fallback_factory()
        return self._visual_fallback

    def _online(
        self,
    ) -> tuple[VisionAnalysis, list[PlaceEvidence], dict[str, str]]:
        status: dict[str, str] = {}
        if not self.online_evidence:
            status["online_evidence"] = "disabled by default; pass --online-evidence"
            return VisionAnalysis(), [], status

        try:
            vision = self.vision_provider or GoogleVisionEvidenceProvider()
            analysis = vision.analyze(self.input_path)
            status["google_cloud_vision"] = "ok"
        except Exception as exc:
            status["google_cloud_vision"] = f"unavailable: {exc}"
            return VisionAnalysis(), [], status

        places: list[PlaceEvidence] = []
        queries = [
            ("ocr", query) for query in ocr_queries(analysis.detected_text)
        ] + [("web", query) for query in web_queries(analysis)]
        if not queries:
            status["google_places"] = "skipped: no place-like OCR or web labels"
            return analysis, places, status
        try:
            provider = self.places_provider or GooglePlacesEvidenceProvider()
            seen: set[str] = set()
            for source, query in queries:
                if query.casefold() in seen:
                    continue
                seen.add(query.casefold())
                place = provider.search(query)
                if place is not None:
                    places.append(replace(place, query_source=source))
            status["google_places"] = "ok"
        except Exception as exc:
            status["google_places"] = f"unavailable: {exc}"
        return analysis, places, status

    @staticmethod
    def _landmark_place_pair(
        landmarks: Sequence[LandmarkEvidence],
        places: Sequence[PlaceEvidence],
    ) -> tuple[LandmarkEvidence, PlaceEvidence] | None:
        pairs = [
            (landmark, place)
            for landmark in landmarks
            for place in places
            if landmark.score >= LANDMARK_CORROBORATION_SCORE
            and place.token_overlap >= PLACE_NAME_TOKEN_OVERLAP
            and len(_tokens(place.name)) >= 2
            and haversine_km(landmark.lat, landmark.lon, place.lat, place.lon)
            <= EVIDENCE_AGREEMENT_KM
        ]
        if not pairs:
            return None
        return max(pairs, key=lambda pair: (pair[0].score, pair[1].token_overlap))

    def predict_image(self, image: Image.Image, top_k: int = 5) -> list[Candidate]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        try:
            exif = extract_exif_gps(self.input_path)
        except Exception as exc:
            logger.warning("Could not parse EXIF GPS: %s", exc)
            exif = None
        if exif is not None:
            lat, lon = exif
            self.last_assessment = HybridAssessment(
                decision="emit",
                reliability_tier="embedded_gps_metadata",
                reason=(
                    "The source image contains valid embedded EXIF GPS coordinates. "
                    "EXIF is direct capture metadata, but it is not cryptographically "
                    "authenticated and can be edited."
                ),
                automation_accepted=True,
                state_country_accepted=True,
                evidence_agreement_radius_km=None,
                selected_source="exif_gps",
                exif_gps={"latitude": round(lat, 6), "longitude": round(lon, 6)},
                provider_status={"online_evidence": "skipped: EXIF GPS is authoritative"},
            )
            return [Candidate(lat, lon, 1.0)]

        analysis, places, status = self._online()
        pair = self._landmark_place_pair(analysis.landmarks, places)
        if pair is not None:
            landmark, place = pair
            # The mapped place is the requested named-site coordinate; the
            # landmark coordinate corroborates it rather than being averaged
            # into a possibly less meaningful midpoint.
            lat, lon = place.lat, place.lon
            self.last_assessment = HybridAssessment(
                decision="emit",
                reliability_tier="corroborated_landmark_and_text",
                reason=(
                    "Cloud landmark coordinates and an independently resolved OCR "
                    f"place agree within {EVIDENCE_AGREEMENT_KM:g} km."
                ),
                automation_accepted=True,
                state_country_accepted=True,
                evidence_agreement_radius_km=EVIDENCE_AGREEMENT_KM,
                selected_source="landmark_plus_ocr_place",
                landmarks=list(analysis.landmarks),
                detected_text=analysis.detected_text,
                web_entities=list(analysis.web_entities),
                matching_pages=list(analysis.matching_pages),
                places=list(places),
                provider_status=status,
            )
            return [Candidate(lat, lon, max(landmark.score, place.token_overlap))]

        strong_web_places = [
            item
            for item in places
            if item.query_source == "web"
            and item.token_overlap >= PLACE_NAME_TOKEN_OVERLAP
            and len(_tokens(item.name)) >= 2
            and len(_tokens(item.query)) >= 2
        ]
        web_domains = independent_web_domains(analysis.matching_pages)
        if strong_web_places and len(web_domains) >= 2:
            best = max(strong_web_places, key=lambda item: item.token_overlap)
            self.last_assessment = HybridAssessment(
                decision="emit",
                reliability_tier="web_image_matches_and_mapped_place",
                reason=(
                    "Cloud Vision found the image on multiple independent web "
                    "domains, and its place label strongly matches a Google "
                    "Places result."
                ),
                automation_accepted=True,
                state_country_accepted=True,
                evidence_agreement_radius_km=None,
                selected_source="web_detection_plus_google_places",
                landmarks=list(analysis.landmarks),
                detected_text=analysis.detected_text,
                web_entities=list(analysis.web_entities),
                matching_pages=list(analysis.matching_pages),
                places=list(places),
                provider_status=status,
            )
            return [Candidate(best.lat, best.lon, best.token_overlap)]

        strong_landmarks = [
            item for item in analysis.landmarks if item.score >= LANDMARK_ACCEPT_SCORE
        ]
        if strong_landmarks:
            best = max(strong_landmarks, key=lambda item: item.score)
            self.last_assessment = HybridAssessment(
                decision="review",
                reliability_tier="provider_landmark",
                reason=(
                    f"Google Cloud Vision identified {best.name!r} with score "
                    f"{best.score:.3f}. This provider result is stronger than a "
                    "visual geolocation prior but is not a mathematical guarantee."
                ),
                automation_accepted=False,
                state_country_accepted=True,
                evidence_agreement_radius_km=None,
                selected_source="google_cloud_landmark",
                landmarks=list(analysis.landmarks),
                detected_text=analysis.detected_text,
                web_entities=list(analysis.web_entities),
                matching_pages=list(analysis.matching_pages),
                places=list(places),
                provider_status=status,
            )
            return [Candidate(best.lat, best.lon, best.score)]

        strong_places = [
            item
            for item in places
            if item.token_overlap >= PLACE_NAME_TOKEN_OVERLAP
            and len(_tokens(item.name)) >= 2
            and len(_tokens(item.query)) >= 2
        ]
        if strong_places:
            best = max(strong_places, key=lambda item: item.token_overlap)
            self.last_assessment = HybridAssessment(
                decision="review",
                reliability_tier="ocr_place_name",
                reason=(
                    "OCR text closely matches the returned Google Places name. "
                    "The coordinate refers to that mapped place."
                ),
                automation_accepted=False,
                state_country_accepted=True,
                evidence_agreement_radius_km=None,
                selected_source="ocr_google_places",
                landmarks=list(analysis.landmarks),
                detected_text=analysis.detected_text,
                web_entities=list(analysis.web_entities),
                matching_pages=list(analysis.matching_pages),
                places=list(places),
                provider_status=status,
            )
            return [Candidate(best.lat, best.lon, best.token_overlap)]

        fallback = self._fallback()
        candidates = fallback.predict_image(image, top_k=top_k)
        self.last_assessment = HybridAssessment(
            decision="abstain",
            reliability_tier="visual_hypothesis_only",
            reason=(
                "No verified metadata, accepted landmark, or resolvable place text "
                f"was found. {self.visual_fallback_name} candidates are included for "
                "manual review but are not authorized as a prediction."
            ),
            automation_accepted=False,
            state_country_accepted=False,
            evidence_agreement_radius_km=None,
            selected_source="visual_fallback",
            landmarks=list(analysis.landmarks),
            detected_text=analysis.detected_text,
            web_entities=list(analysis.web_entities),
            matching_pages=list(analysis.matching_pages),
            places=list(places),
            provider_status=status,
        )
        return candidates
