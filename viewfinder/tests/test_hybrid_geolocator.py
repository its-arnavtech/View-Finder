"""Unit tests for the evidence-first hybrid selector."""

from __future__ import annotations

from PIL import Image

from viewfinder.aggregator import Candidate
from viewfinder.hybrid_geolocator import (
    HybridGeolocator,
    LandmarkEvidence,
    PlaceEvidence,
    VisionAnalysis,
    WebEvidence,
    WebPageEvidence,
    gps_from_mapping,
    ocr_queries,
    token_overlap,
    web_queries,
)


class _Fallback:
    def predict_image(self, image: Image.Image, top_k: int = 5) -> list[Candidate]:
        return [Candidate(35.0, 33.0, 0.9)]


class _Vision:
    def __init__(self, analysis: VisionAnalysis) -> None:
        self.analysis = analysis

    def analyze(self, path):
        return self.analysis


class _Places:
    def __init__(self, result: PlaceEvidence | None) -> None:
        self.result = result

    def search(self, query: str) -> PlaceEvidence | None:
        return self.result


def test_gps_mapping_decodes_bytes_references() -> None:
    gps = {
        "GPSLatitude": (42, 40, 0),
        "GPSLatitudeRef": b"N",
        "GPSLongitude": (103, 28, 0),
        "GPSLongitudeRef": b"W",
    }
    assert gps_from_mapping(gps) == (42.666666666666664, -103.46666666666667)


def test_ocr_queries_and_overlap_are_conservative() -> None:
    text = "STOP\nFort Robinson State Park\nUS 20\n"
    assert ocr_queries(text)[0] == "Fort Robinson State Park"
    assert token_overlap("Fort Robinson State Park", "Fort Robinson State Park") == 1.0
    assert token_overlap("Visitor Center", "Fort Robinson State Park") == 0.0


def test_web_queries_prioritize_high_score_entities() -> None:
    analysis = VisionAnalysis(
        web_entities=[
            WebEvidence("Towers in Time Ponca State Park", 0.95, "entity"),
            WebEvidence("Outdoor recreation", 0.20, "entity"),
        ],
        matching_pages=[
            WebPageEvidence(
                "Towers in Time at Ponca State Park | Nebraska",
                "https://example.test/ponca",
            )
        ],
    )
    assert web_queries(analysis)[0] == "Towers in Time Ponca State Park"


def test_exif_authorizes_without_loading_visual_fallback(monkeypatch, tmp_path) -> None:
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8)).save(path)
    monkeypatch.setattr(
        "viewfinder.hybrid_geolocator.extract_exif_gps",
        lambda ignored: (42.668, -103.466),
    )
    fallback_called = False

    def fallback_factory():
        nonlocal fallback_called
        fallback_called = True
        return _Fallback()

    predictor = HybridGeolocator(path, visual_fallback_factory=fallback_factory)
    candidates = predictor.predict_image(Image.new("RGB", (8, 8)))

    assert candidates[0].lat == 42.668
    assert predictor.last_assessment is not None
    assert predictor.last_assessment.automation_accepted
    assert predictor.last_assessment.selected_source == "exif_gps"
    assert not fallback_called


def test_landmark_and_place_agreement_authorizes(monkeypatch, tmp_path) -> None:
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8)).save(path)
    monkeypatch.setattr("viewfinder.hybrid_geolocator.extract_exif_gps", lambda ignored: None)
    landmark = LandmarkEvidence("Fort Robinson", 42.668, -103.466, 0.92)
    place = PlaceEvidence(
        query="Fort Robinson State Park",
        name="Fort Robinson State Park",
        address="Soldier Creek Rd, Crawford, NE",
        lat=42.6682,
        lon=-103.4658,
        token_overlap=1.0,
    )
    predictor = HybridGeolocator(
        path,
        online_evidence=True,
        vision_provider=_Vision(
            VisionAnalysis(
                landmarks=[landmark],
                detected_text="Fort Robinson State Park",
            )
        ),
        places_provider=_Places(place),
        visual_fallback_factory=lambda: _Fallback(),
    )

    candidates = predictor.predict_image(Image.new("RGB", (8, 8)))

    assert abs(candidates[0].lat - 42.6681) < 0.001
    assert predictor.last_assessment is not None
    assert predictor.last_assessment.automation_accepted
    assert predictor.last_assessment.reliability_tier == "corroborated_landmark_and_text"


def test_single_landmark_requires_review(monkeypatch, tmp_path) -> None:
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8)).save(path)
    monkeypatch.setattr("viewfinder.hybrid_geolocator.extract_exif_gps", lambda ignored: None)
    landmark = LandmarkEvidence("Fort Robinson", 42.668, -103.466, 0.95)
    predictor = HybridGeolocator(
        path,
        online_evidence=True,
        vision_provider=_Vision(VisionAnalysis(landmarks=[landmark])),
        visual_fallback_factory=lambda: _Fallback(),
    )

    predictor.predict_image(Image.new("RGB", (8, 8)))

    assert predictor.last_assessment is not None
    assert predictor.last_assessment.decision == "review"
    assert not predictor.last_assessment.automation_accepted
    assert predictor.last_assessment.state_country_accepted


def test_multiple_web_matches_and_place_authorize(monkeypatch, tmp_path) -> None:
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8)).save(path)
    monkeypatch.setattr("viewfinder.hybrid_geolocator.extract_exif_gps", lambda ignored: None)
    place = PlaceEvidence(
        query="Towers in Time Ponca State Park",
        name="Ponca State Park",
        address="88090 Spur 26 E, Ponca, NE",
        lat=42.6022,
        lon=-96.7097,
        token_overlap=1.0,
    )
    analysis = VisionAnalysis(
        web_entities=[
            WebEvidence("Towers in Time Ponca State Park", 0.95, "entity")
        ],
        matching_pages=[
            WebPageEvidence("Towers in Time at Ponca State Park", "https://one.test"),
            WebPageEvidence("Ponca State Park Towers in Time", "https://two.test"),
        ],
    )
    predictor = HybridGeolocator(
        path,
        online_evidence=True,
        vision_provider=_Vision(analysis),
        places_provider=_Places(place),
        visual_fallback_factory=lambda: _Fallback(),
    )

    candidates = predictor.predict_image(Image.new("RGB", (8, 8)))

    assert candidates[0].lat == 42.6022
    assert predictor.last_assessment is not None
    assert predictor.last_assessment.automation_accepted
    assert (
        predictor.last_assessment.reliability_tier
        == "web_image_matches_and_mapped_place"
    )


def test_visual_only_fallback_abstains(monkeypatch, tmp_path) -> None:
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8)).save(path)
    monkeypatch.setattr("viewfinder.hybrid_geolocator.extract_exif_gps", lambda ignored: None)
    predictor = HybridGeolocator(
        path,
        visual_fallback_factory=lambda: _Fallback(),
    )

    candidates = predictor.predict_image(Image.new("RGB", (8, 8)))

    assert candidates[0].lat == 35.0
    assert predictor.last_assessment is not None
    assert predictor.last_assessment.decision == "abstain"
    assert not predictor.last_assessment.state_country_accepted
