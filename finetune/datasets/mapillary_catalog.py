"""The existing gallery-pipeline Mapillary pulls, as an adapter.

Status: CLEAR. This is imagery the project already holds, pulled through the
authorised Mapillary API with the project's own token. Mapillary's Terms of Use
place user-contributed imagery under CC BY-SA, which permits derivatives with
attribution and share-alike.

This adapter exists so the pre-existing training path is reachable through the
same interface as new sources, rather than being a special case in the loop.
It wraps :mod:`dataset_index`, which already handles the WSL-to-Windows path
translation and the cumulative-superset dedupe across gallery versions.

Known limitation, measured: this data is spatially near-duplicated. 77% of
images have another within 10 m, median nearest neighbour 4.8 m, so 1365 images
collapsed to 184 distinct places at 100 m spacing. That is what bounded earlier
runs, and it is why the loop thins by spacing before splitting.
"""

from __future__ import annotations

import os

import pandas as pd

from .base import Availability, LicenceStatus, validate_frame


class MapillaryCatalogAdapter:
    """Yield rows from gallery-pipeline's extract catalogs or built galleries."""

    name = "mapillary-catalog"

    def __init__(self, catalog: str | None = None) -> None:
        self.catalog = catalog

    def availability(self) -> Availability:
        if self.catalog:
            ok = os.path.isfile(self.catalog)
            note = f"catalog {self.catalog}" if ok else f"catalog not found: {self.catalog}"
        else:
            # build_index() globs the committed gallery metadata.
            ok = True
            note = "using gallery-pipeline gallery metadata via dataset_index.build_index"
        return Availability(
            licence=LicenceStatus.CLEAR,
            licence_note="Mapillary user imagery under CC BY-SA per Mapillary Terms of Use",
            access_ok=ok,
            access_note=note,
        )

    def load(self, limit: int | None = None, seed: int = 1337) -> pd.DataFrame:
        availability = self.availability()
        if not availability.usable:
            raise PermissionError(
                f"{self.name} cannot be ingested: {availability.blocked_reason()}"
            )

        from dataset_index import build_index, build_index_from_catalog

        if self.catalog:
            frame, _ = build_index_from_catalog(self.catalog, dataset_name=self.name)
        else:
            frame, _ = build_index()

        frame = frame.copy()
        frame["dataset"] = self.name
        frame = validate_frame(frame, self.name)
        if limit is not None and len(frame) > limit:
            frame = frame.sample(n=limit, random_state=seed).reset_index(drop=True)
        return frame

    def discard(self, frame: pd.DataFrame) -> int:
        """No-op: these pulls are committed project data, not scratch."""
        return 0
