"""Adapter contract shared by every training data source."""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

# The frame shape the training loop consumes. Matches dataset_index.build_index
# so existing Mapillary paths and new sources are interchangeable.
REQUIRED_COLUMNS = ("image_id", "lat", "lon", "tile_id", "region", "dataset", "image_path")


class LicenceStatus(enum.Enum):
    """How confident we are that training on a source is permitted.

    Recorded from primary-source verification, not from secondhand claims.
    """

    CLEAR = "clear"
    """Verified permissive terms, no NonCommercial or NoDerivatives item."""

    CLEAR_AFTER_FILTER = "clear_after_filter"
    """Mixed per-item licences, but each item's licence is known and the
    adapter filters to the permissive subset before yielding anything."""

    UNCLEAR = "unclear"
    """No licence text found on any primary source. Never auto-ingested."""

    RESTRICTIVE = "restrictive"
    """Terms found and they prohibit this use (typically NoDerivatives, which
    makes redistributing a checkpoint trained on it questionable)."""


@dataclass(frozen=True)
class Availability:
    """Whether an adapter may run right now, and if not, what would unblock it."""

    licence: LicenceStatus
    licence_note: str
    access_ok: bool
    access_note: str

    @property
    def usable(self) -> bool:
        return (
            self.licence in (LicenceStatus.CLEAR, LicenceStatus.CLEAR_AFTER_FILTER)
            and self.access_ok
        )

    def blocked_reason(self) -> str:
        parts = []
        if self.licence not in (LicenceStatus.CLEAR, LicenceStatus.CLEAR_AFTER_FILTER):
            parts.append(f"licence [{self.licence.value}]: {self.licence_note}")
        if not self.access_ok:
            parts.append(f"access: {self.access_note}")
        return " | ".join(parts)


class DatasetAdapter(Protocol):
    """Turn a source into training rows, and clean up after itself."""

    name: str

    def availability(self) -> Availability:
        """Report licence and access status without touching the network."""
        ...

    def load(self, limit: int | None = None, seed: int = 1337) -> pd.DataFrame:
        """Return up to ``limit`` rows in :data:`REQUIRED_COLUMNS` shape.

        Raises:
            PermissionError: If :meth:`availability` says this may not run.
        """
        ...

    def discard(self, frame: pd.DataFrame) -> int:
        """Delete raw files for these rows after training. Returns count removed.

        The batch pattern is ingest -> train -> checkpoint -> discard, so total
        disk stays flat no matter how many batches a session runs. Adapters
        backed by data that already lives in the repo do not delete anything.
        """
        ...


def validate_frame(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Normalise and sanity-check an adapter's output.

    Drops rows with unusable coordinates or missing files, which is cheaper to
    do once here than to discover inside a dataloader worker mid-epoch.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{dataset} adapter produced a frame missing {missing}")

    frame = frame.dropna(subset=["lat", "lon", "image_path"]).copy()
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    frame = frame.dropna(subset=["lat", "lon"])

    frame = frame[frame["lat"].between(-90, 90) & frame["lon"].between(-180, 180)]
    # (0, 0) is the classic "coordinate missing" placeholder, not Null Island.
    frame = frame[~(frame["lat"].eq(0) & frame["lon"].eq(0))]
    frame = frame[frame["image_path"].map(os.path.exists)]
    frame = frame.drop_duplicates(subset=["image_id"])

    return frame[list(REQUIRED_COLUMNS)].reset_index(drop=True)
