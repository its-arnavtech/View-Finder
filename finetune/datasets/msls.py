"""Mapillary Street-Level Sequences (MSLS), ~1.6M street-level images.

Status: **UNCLEAR licence — blocked, will not auto-ingest.**

What was verified (2026-08-08, primary sources only):

* The GitHub repository ``mapillary/mapillary_sls`` is **MIT licensed**, but
  that LICENSE file covers the *evaluation code*, not the imagery. Copyright is
  held by Meta Platforms, Inc.
* Mapillary's general Terms of Use place user-contributed imagery under
  **CC BY-SA**, and explicitly reserve the right to release some content under
  **CC BY-NC-SA** instead -- so "Mapillary imagery is CC BY-SA" is not a safe
  blanket assumption for any particular corpus.
* Mapillary's own announcement states MSLS is "available both as a commercial
  edition and a research edition". A separate research edition strongly implies
  terms distinct from plain CC BY-SA, most likely research-only.
* No licence text for either edition could be retrieved from any primary page:
  the dataset page (``mapillary.com/dataset/places``) is a client-rendered app
  that serves no terms to a fetch, the CVPR 2020 paper PDF returns HTTP 403,
  and the Mapillary help article on CC BY-SA returns HTTP 403.

Because the specific terms could not be read, this adapter refuses to run. That
is the instructed behaviour for unclear licensing, and it is also the same test
that rejected GSV-Cities: an unread or NoDerivatives licence makes distributing
a checkpoint trained on the data questionable.

To unblock, a human needs to open the dataset page, read the edition terms, and
accept them if appropriate -- accepting terms is not something this code or the
agent running it should do on the owner's behalf. Once read, set
``licence_confirmed=True`` with the actual licence string.

Format (from the repository README, which is public): metadata ships as
``raw.csv`` with ``lon, lat, ca, captured_at, pano`` per image, plus
``postprocessed.csv`` with UTM easting/northing. Per-image GPS is genuine and
per-image, which is why this dataset is worth unblocking.
"""

from __future__ import annotations

import os

import pandas as pd

from .base import Availability, LicenceStatus, validate_frame

DATASET_URL = "https://www.mapillary.com/dataset/places"


class MSLSAdapter:
    """Ingest MSLS once its terms have been read and confirmed by a human."""

    name = "msls"

    def __init__(
        self,
        root: str | None = None,
        licence_confirmed: bool = False,
        confirmed_licence: str = "",
    ) -> None:
        self.root = root
        self.licence_confirmed = licence_confirmed
        self.confirmed_licence = confirmed_licence

    def availability(self) -> Availability:
        if self.licence_confirmed and self.confirmed_licence:
            licence = LicenceStatus.CLEAR
            note = f"confirmed by operator: {self.confirmed_licence}"
        else:
            licence = LicenceStatus.UNCLEAR
            note = (
                "MSLS ships as separate commercial and research editions and no "
                "licence text is retrievable from any primary source (dataset page "
                "is client-rendered; CVPR PDF and Mapillary help article both 403). "
                f"A human must read the terms at {DATASET_URL} and pass "
                "licence_confirmed=True with the actual licence."
            )
        present = bool(self.root) and os.path.isdir(self.root or "")
        return Availability(
            licence=licence,
            licence_note=note,
            access_ok=present,
            access_note=(
                f"local MSLS root {self.root}"
                if present
                else f"no local MSLS extract; download requires accepting terms at {DATASET_URL}"
            ),
        )

    def load(self, limit: int | None = None, seed: int = 1337) -> pd.DataFrame:
        availability = self.availability()
        if not availability.usable:
            raise PermissionError(
                f"{self.name} cannot be ingested: {availability.blocked_reason()}"
            )

        # Reached only after a human confirms the licence and supplies an extract.
        # MSLS lays out as <root>/<city>/database|query/{images,raw.csv}.
        frames = []
        for city_dir, _, _ in os.walk(self.root or ""):
            raw_csv = os.path.join(city_dir, "raw.csv")
            if not os.path.isfile(raw_csv):
                continue
            city = os.path.basename(os.path.dirname(city_dir))
            part = pd.read_csv(raw_csv)
            part["image_path"] = part["key"].map(
                lambda key: os.path.join(city_dir, "images", f"{key}.jpg")
            )
            part["image_id"] = "msls-" + part["key"].astype(str)
            part["region"] = city
            # Sequences are the natural leak unit: consecutive frames of one
            # sequence are the same place seconds apart.
            part["tile_id"] = "seq-" + part.get(
                "sequence_key", pd.Series(city, index=part.index)
            ).astype(str)
            part["dataset"] = self.name
            frames.append(part.rename(columns={"lat": "lat", "lon": "lon"}))

        if not frames:
            raise FileNotFoundError(f"No raw.csv found under {self.root}")

        frame = validate_frame(pd.concat(frames, ignore_index=True), self.name)
        if limit is not None and len(frame) > limit:
            frame = frame.sample(n=limit, random_state=seed).reset_index(drop=True)
        return frame

    def discard(self, frame: pd.DataFrame) -> int:
        """Delete the raw images for these rows, keeping only checkpoints."""
        removed = 0
        for path in frame["image_path"]:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
        return removed
