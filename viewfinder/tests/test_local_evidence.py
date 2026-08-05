"""Tests for the fully local OCR/gazetteer/landmark evidence path."""

from __future__ import annotations

import sqlite3

from PIL import Image

from viewfinder.aggregator import Candidate
from viewfinder.local_gazetteer import GeoNamesResolver, normalize_name, phrase_candidates
from viewfinder.local_hybrid_geolocator import LocalHybridGeolocator
from viewfinder.local_landmarks import LandmarkMatch
from viewfinder.local_ocr import OcrLine


def _gazetteer(path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO metadata VALUES ('schema_version', '1');
        CREATE TABLE countries(code TEXT PRIMARY KEY, iso3 TEXT, name TEXT, continent TEXT);
        INSERT INTO countries VALUES ('US', 'USA', 'United States', 'NA');
        CREATE TABLE admin1(
            country_code TEXT, admin1_code TEXT, name TEXT, ascii_name TEXT,
            geoname_id INTEGER, PRIMARY KEY(country_code, admin1_code)
        );
        INSERT INTO admin1 VALUES ('US', 'NE', 'Nebraska', 'Nebraska', 1);
        CREATE TABLE places(
            geoname_id INTEGER PRIMARY KEY, name TEXT, ascii_name TEXT,
            lat REAL, lon REAL, feature_class TEXT, feature_code TEXT,
            country_code TEXT, admin1_code TEXT, population INTEGER
        );
        INSERT INTO places VALUES
          (5074958, 'Ponca State Park', 'Ponca State Park',
           42.6022, -96.7097, 'L', 'PRK', 'US', 'NE', 0);
        CREATE TABLE names(
            geoname_id INTEGER, normalized_name TEXT, name TEXT, language TEXT,
            is_preferred INTEGER, is_short INTEGER, source TEXT,
            PRIMARY KEY(geoname_id, normalized_name)
        );
        INSERT INTO names VALUES
          (5074958, 'ponca state park', 'Ponca State Park', 'en', 1, 0, 'test');
        CREATE TABLE osm_names(
            osm_id TEXT PRIMARY KEY, normalized_name TEXT, name TEXT,
            lat REAL, lon REAL, country_code TEXT, admin1_code TEXT,
            feature_class TEXT, feature_code TEXT
        );
        CREATE VIRTUAL TABLE names_fts USING fts5(
            normalized_name, display_name UNINDEXED, geoname_id UNINDEXED
        );
        INSERT INTO names_fts VALUES ('ponca state park', 'Ponca State Park', '5074958');
        """
    )
    connection.commit()
    connection.close()


class _Ocr:
    def analyze(self, image):
        return [OcrLine("Ponca State Park", 0.98, [])]


class _Landmarks:
    def analyze(self, image):
        matches = [
            LandmarkMatch(
                page_id=index,
                geoname_id=5074958,
                landmark_name="Ponca State Park",
                lat=42.6022,
                lon=-96.7097,
                country_code="US",
                admin1_code="NE",
                title=f"Ponca reference {index}",
                page_url=f"https://commons.test/{index}",
                license_short="CC BY-SA 4.0",
                artist=f"Test Artist {(index % 2) + 1}",
                similarity=score,
            )
            for index, score in enumerate((0.81, 0.76, 0.73), start=1)
        ]
        return {
            "perceptual_matches": [],
            "visual_matches": matches,
            "groups": [matches],
            "gallery_size": 3,
        }


class _Prior:
    def predict_image(self, image, top_k=5):
        return [Candidate(35.0, 33.0, 1.0)]


def test_normalization_and_phrases() -> None:
    assert normalize_name("  Mórfou—Bay! ") == "morfou bay"
    assert phrase_candidates(["Welcome to Ponca State Park"])[0] == (
        "welcome to ponca state park"
    )


def test_offline_gazetteer_exact_match(tmp_path) -> None:
    path = tmp_path / "places.sqlite3"
    _gazetteer(path)
    match = GeoNamesResolver(path).search_exact("Ponca State Park")[0]
    assert match.country_code == "US"
    assert match.admin1_name == "Nebraska"
    assert match.feature_code == "PRK"


def test_ocr_and_landmark_corroboration_emits(monkeypatch, tmp_path) -> None:
    database = tmp_path / "places.sqlite3"
    _gazetteer(database)
    image_path = tmp_path / "query.jpg"
    Image.new("RGB", (16, 16)).save(image_path)
    monkeypatch.setattr(
        "viewfinder.local_hybrid_geolocator.extract_exif_gps",
        lambda path: None,
    )
    predictor = LocalHybridGeolocator(
        image_path,
        data_root=tmp_path,
        ocr_provider=_Ocr(),
        resolver=GeoNamesResolver(database),
        landmark_provider=_Landmarks(),
        visual_prior_factory=lambda: _Prior(),
    )

    candidate = predictor.predict_image(Image.new("RGB", (16, 16)))[0]

    assert abs(candidate.lat - 42.6022) < 1e-6
    assert predictor.last_assessment is not None
    assert predictor.last_assessment.automation_accepted
    assert predictor.last_assessment.selected_source == (
        "paddleocr_geonames_plus_commons"
    )


def test_missing_local_evidence_keeps_prior_unauthorized(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "query.jpg"
    Image.new("RGB", (16, 16)).save(image_path)
    monkeypatch.setattr(
        "viewfinder.local_hybrid_geolocator.extract_exif_gps",
        lambda path: None,
    )

    class Empty:
        def analyze(self, image):
            return []

    predictor = LocalHybridGeolocator(
        image_path,
        data_root=tmp_path,
        ocr_provider=Empty(),
        visual_prior_factory=lambda: _Prior(),
    )
    predictor.predict_image(Image.new("RGB", (16, 16)))

    assert predictor.last_assessment is not None
    assert predictor.last_assessment.decision == "abstain"
    assert not predictor.last_assessment.automation_accepted


def test_video_frames_reuse_local_evidence_and_emit(monkeypatch, tmp_path) -> None:
    database = tmp_path / "places.sqlite3"
    _gazetteer(database)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "viewfinder.local_hybrid_geolocator.extract_exif_gps",
        lambda path: None,
    )

    class Frame:
        def __init__(self, index):
            self.index = index
            self.timestamp = float(index)
            self.blur_score = 100.0
            self.image = Image.new("RGB", (16, 16))

    predictor = LocalHybridGeolocator(
        video_path,
        data_root=tmp_path,
        ocr_provider=_Ocr(),
        resolver=GeoNamesResolver(database),
        landmark_provider=_Landmarks(),
        visual_prior_factory=lambda: _Prior(),
    )
    predictions = predictor.predict_frames([Frame(0), Frame(1)])

    assert len(predictions) == 2
    assert len(predictor.evidence_frame_predictions) == 2
    assert predictor.last_assessment is not None
    assert predictor.last_assessment.automation_accepted
    assert predictor.last_assessment.reliability_tier == (
        "corroborated_video_frame_evidence"
    )
