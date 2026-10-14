"""YFCC100M geotagged subset (~48.4M geotagged of 100M items).

Status: **CLEAR_AFTER_FILTER on licence, but access is gated — blocked today.**

Licence, verified from the dataset paper (arXiv:1503.01817, Table 3). Every
item carries its own Creative Commons licence, and they are not uniform:

===========================  ============  ==========
licence                      photos        videos
===========================  ============  ==========
CC-BY                          17,210,144     137,503
CC-BY-SA                        9,408,154      72,116
CC-BY-ND                        4,910,766      37,542   <- NoDerivatives
CC-BY-NC                       12,674,885     102,288   <- NonCommercial
CC-BY-NC-SA                    28,776,835     235,319   <- NonCommercial
CC-BY-NC-ND                    26,225,780     208,668   <- both
===========================  ============  ==========

Roughly 67.7% of the corpus is NonCommercial and about 31.1M items carry
NoDerivatives. Training on the whole thing and redistributing the checkpoint
would be exactly the problem that disqualified GSV-Cities.

It is still usable, because the licence is recorded **per item** in the
metadata. This adapter filters to CC-BY and CC-BY-SA only -- 26.6M items, of
which the geotagged share is roughly half -- and never yields anything else.
Note the extracted *features* the project distributes are CC0; the restriction
is on the source photographs.

Access is the blocker. The metadata (``yfcc100m_dataset.bz2``, ~15 GB) and the
media in the ``multimedia-commons`` S3 bucket both require an **AWS account**,
and the Yahoo Webscope route requires a **Yahoo account**. Creating accounts is
not something this code or the agent running it should do on the owner's
behalf, so the adapter refuses until credentials or a local extract exist.

Coordinates are genuine per-image (Flickr EXIF or user-placed), not per-subject
centroids, which is what makes this the strongest candidate of the three
externally-sourced datasets once access is arranged.
"""

from __future__ import annotations

import os

import pandas as pd

from .base import Availability, LicenceStatus, validate_frame

# The only licence markers this adapter will accept.
PERMITTED_LICENCES = ("cc-by", "cc-by-sa")
BLOCKED_MARKERS = ("nc", "nd", "noncommercial", "noderiv")

METADATA_NOTE = (
    "yfcc100m_dataset.bz2 (~15 GB) from the multimedia-commons S3 bucket, "
    "or via Yahoo Webscope"
)


def licence_permits(licence_field: str | None) -> bool:
    """True only for CC-BY and CC-BY-SA; everything else is excluded."""
    if not licence_field:
        return False
    text = str(licence_field).lower()
    tokens = set(text.replace("/", " ").replace("-", " ").replace("_", " ").split())
    if tokens & {"nc", "nd", "noncommercial", "noderivs", "noderivatives"}:
        return False
    return any(marker in text for marker in PERMITTED_LICENCES)


class YFCC100MAdapter:
    """Ingest license-filtered, geotagged YFCC100M rows from a local extract."""

    name = "yfcc100m"

    def __init__(self, metadata_path: str | None = None, image_root: str | None = None) -> None:
        self.metadata_path = metadata_path
        self.image_root = image_root

    def availability(self) -> Availability:
        have_metadata = bool(self.metadata_path) and os.path.isfile(self.metadata_path or "")
        have_images = bool(self.image_root) and os.path.isdir(self.image_root or "")
        access_ok = have_metadata and have_images
        return Availability(
            licence=LicenceStatus.CLEAR_AFTER_FILTER,
            licence_note=(
                "Per-item CC licences; ~67.7% NonCommercial and ~31.1M items "
                "NoDerivatives. Adapter yields CC-BY and CC-BY-SA only."
            ),
            access_ok=access_ok,
            access_note=(
                "local extract present"
                if access_ok
                else (
                    f"needs {METADATA_NOTE} plus a local image root. Both routes "
                    "require an account (AWS or Yahoo); creating one is an owner "
                    "action, not an automated one."
                )
            ),
        )

    def load(self, limit: int | None = None, seed: int = 1337) -> pd.DataFrame:
        availability = self.availability()
        if not availability.usable:
            raise PermissionError(
                f"{self.name} cannot be ingested: {availability.blocked_reason()}"
            )

        # The metadata is a TSV; only the columns needed are read so a 15 GB
        # file does not have to fit in memory.
        usecols = ["photoid", "latitude", "longitude", "licensename", "accuracy"]
        frame = pd.read_csv(
            self.metadata_path, sep="\t", usecols=lambda c: c in usecols, low_memory=True
        )

        before = len(frame)
        frame = frame[frame["licensename"].map(licence_permits)].copy()
        self.licence_filtered_out = before - len(frame)

        frame = frame.dropna(subset=["latitude", "longitude"])
        # Flickr accuracy 1-16; below ~12 the coordinate is city-level at best,
        # which is the same per-subject approximation problem GLv2 has.
        if "accuracy" in frame.columns:
            frame = frame[pd.to_numeric(frame["accuracy"], errors="coerce").fillna(0) >= 12]

        frame["image_id"] = "yfcc-" + frame["photoid"].astype(str)
        frame["lat"] = frame["latitude"]
        frame["lon"] = frame["longitude"]
        frame["region"] = "yfcc"
        frame["dataset"] = self.name
        frame["image_path"] = frame["photoid"].map(
            lambda pid: os.path.join(self.image_root or "", f"{pid}.jpg")
        )
        # No natural grouping key, so each image is its own unit; the spacing
        # thinning downstream is what enforces real location separation.
        frame["tile_id"] = frame["image_id"]

        frame = validate_frame(frame, self.name)
        if limit is not None and len(frame) > limit:
            frame = frame.sample(n=limit, random_state=seed).reset_index(drop=True)
        return frame

    def discard(self, frame: pd.DataFrame) -> int:
        removed = 0
        for path in frame["image_path"]:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
        return removed
