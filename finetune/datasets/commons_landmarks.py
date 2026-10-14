"""Wikimedia Commons geotagged landmark imagery, from the local catalog.

Source: ``local-data/landmarks/catalog.sqlite3``, built by the local-evidence
pipeline through the official MediaWiki API. It records each asset's source
page, licence, artist, credit, coordinates and hashes, so the licence question
is answerable per image rather than per corpus.

Licence: CLEAR_AFTER_FILTER. Every ``ready`` asset carries a named licence in
``license_short``, and a scan of all 21,852 of them found **zero** NonCommercial
and **zero** NoDerivatives items -- the corpus is CC BY / CC BY-SA / CC0 /
Public domain / GFDL. The adapter still filters explicitly rather than trusting
that, because the catalog grows.

Coordinate caveat -- read before interpreting a result
------------------------------------------------------
The catalog's ``camera_lat``/``camera_lon`` are **identical to**
``landmark_lat``/``landmark_lon`` for all 21,852 ready rows. These are
therefore **per-landmark coordinates, not per-image camera positions** -- the
same limitation that disqualified Google Landmarks v2. The error this
introduces is bounded by how far a photographer stands from a landmark
(typically tens to hundreds of metres) rather than being unbounded, which is
milder than GLv2's Wikidata-derived centroids, but it is an approximation and
any result trained on it must say so.

What it is good for: **location diversity**. 21,852 images across 10,298
distinct landmark targets is a far higher distinct-location count than the
dense Mapillary tiles that bound earlier runs (184-730 locations), and distinct
locations -- not image count -- is what binds this fine-tune.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

from .base import Availability, LicenceStatus, validate_frame

DEFAULT_CATALOG = os.path.join("local-data", "landmarks", "catalog.sqlite3")

# Substrings that disqualify a licence. NoDerivatives is the blocking one: it
# makes redistributing a checkpoint trained on the image questionable, which is
# why GSV-Cities (CC BY-NC-ND) was rejected earlier for the same reason.
BLOCKED_LICENCE_TOKENS = ("ND", "NODERIV", "NONCOMMERCIAL", "NC")

# Tokens that mark a licence as known-permissive.
ALLOWED_LICENCE_PREFIXES = (
    "CC BY", "CC0", "PUBLIC DOMAIN", "GFDL", "ATTRIBUTION",
    "COPYRIGHTED FREE USE", "NO RESTRICTIONS", "FAL", "CC SA",
)


def _licence_permits(licence: str | None) -> bool:
    """True when a Commons licence string permits training and redistribution."""
    if not licence:
        return False
    text = licence.upper().strip()
    # Tokenise so "CC BY-ND" is blocked but "CC BY-SA 3.0 nl" (…-nl) is not
    # mistaken for a NoDerivatives marker by a naive substring test.
    tokens = set(text.replace("-", " ").replace("_", " ").split())
    if tokens & {"ND", "NODERIVS", "NODERIVATIVES", "NC", "NONCOMMERCIAL"}:
        return False
    return any(text.startswith(prefix) for prefix in ALLOWED_LICENCE_PREFIXES)


class CommonsLandmarksAdapter:
    """Yield license-filtered Commons landmark images from the local catalog."""

    name = "commons-landmarks"

    def __init__(self, catalog: str | None = None, repo_root: str | None = None) -> None:
        # Resolved against the repository root, not the working directory: this
        # is normally run from finetune/, and a cwd-relative default silently
        # reports "catalog not found" from the wrong place.
        # This file is <repo>/finetune/datasets/commons_landmarks.py, so the
        # root is three levels up.
        root = repo_root or str(Path(__file__).resolve().parents[2])
        self.catalog = catalog or os.path.join(root, DEFAULT_CATALOG)

    def availability(self) -> Availability:
        present = os.path.isfile(self.catalog)
        return Availability(
            licence=LicenceStatus.CLEAR_AFTER_FILTER,
            licence_note=(
                "Wikimedia Commons; per-image licence recorded in the catalog. "
                "Filtered to CC BY / CC BY-SA / CC0 / Public domain / GFDL; "
                "NonCommercial and NoDerivatives items are excluded."
            ),
            access_ok=present,
            access_note=(
                f"local catalog present at {self.catalog}"
                if present
                else f"catalog not found at {self.catalog}; build it with "
                "local-data-pipeline/build_local_data.py"
            ),
        )

    def load(self, limit: int | None = None, seed: int = 1337) -> pd.DataFrame:
        availability = self.availability()
        if not availability.usable:
            raise PermissionError(
                f"{self.name} cannot be ingested: {availability.blocked_reason()}"
            )

        connection = sqlite3.connect(f"file:{self.catalog}?mode=ro", uri=True)
        try:
            frame = pd.read_sql_query(
                """
                SELECT page_id, target_id, license_short,
                       camera_lat, camera_lon, landmark_lat, landmark_lon, local_path
                FROM assets
                WHERE status = 'ready' AND local_path IS NOT NULL
                """,
                connection,
            )
        finally:
            connection.close()

        before = len(frame)
        frame = frame[frame["license_short"].map(_licence_permits)].copy()
        self.licence_filtered_out = before - len(frame)

        frame["lat"] = frame["camera_lat"].fillna(frame["landmark_lat"])
        frame["lon"] = frame["camera_lon"].fillna(frame["landmark_lon"])
        frame["image_id"] = "commons-" + frame["page_id"].astype(str)
        # target_id is the landmark. Using it as tile_id makes the existing
        # location-aware split treat one landmark as one indivisible unit, so a
        # second photo of the same landmark cannot leak across the split.
        frame["tile_id"] = "landmark-" + frame["target_id"].astype(str)
        frame["region"] = "commons"
        frame["dataset"] = self.name
        frame["image_path"] = frame["local_path"]

        frame = validate_frame(frame, self.name)
        if limit is not None and len(frame) > limit:
            frame = frame.sample(n=limit, random_state=seed).reset_index(drop=True)
        return frame

    def discard(self, frame: pd.DataFrame) -> int:
        """No-op: this imagery is a durable local evidence store, not a scratch pull.

        The hybrid predictor's landmark matcher reads the same files at
        inference time, so deleting them would break an active code path.
        """
        return 0
