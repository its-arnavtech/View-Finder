"""Location-diversity sampling, redundancy measurement, and leak-proof splits.

The batch01 run established the failure mode: 1365 images across 72 dense tiles
overfit within ~2 epochs because effective *location* diversity was far lower
than the image count suggested (77% of images had a neighbour within 10 m). The
fix is to select training data by location spacing rather than by image count.

This module is source-agnostic. It works on any frame carrying `lat`, `lon`, and
`image_path`, so the same sampling and split logic applies to Mapillary
gallery-pipeline data or to an external dataset such as GSV-Cities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Accepts broadcastable arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _pairwise_km(coords: np.ndarray, chunk: int = 512):
    """Yield (row_offset, distance_block) so large frames don't allocate N^2 at once."""
    n = len(coords)
    for start in range(0, n, chunk):
        block = coords[start : start + chunk]
        yield start, haversine_km(
            block[:, 0][:, None], block[:, 1][:, None], coords[:, 0][None, :], coords[:, 1][None, :]
        )


@dataclass
class RedundancyReport:
    n: int
    median_nn_m: float
    mean_nn_m: float
    share_within_10m: float
    share_within_50m: float
    share_within_100m: float
    mean_neighbours_within_50m: float

    def format(self, label: str) -> str:
        return (
            f"--- redundancy: {label} (n={self.n}) ---\n"
            f"  nearest-neighbour distance : median {self.median_nn_m:.1f} m | mean {self.mean_nn_m:.1f} m\n"
            f"  share with neighbour <=10 m: {self.share_within_10m:.3f}\n"
            f"  share with neighbour <=50 m: {self.share_within_50m:.3f}\n"
            f"  share with neighbour <=100m: {self.share_within_100m:.3f}\n"
            f"  mean negatives within 50 m : {self.mean_neighbours_within_50m:.1f}"
        )


def measure_redundancy(frame: pd.DataFrame) -> RedundancyReport:
    """Quantify spatial near-duplication -- the thing that drove batch01's overfitting."""
    coords = frame[["lat", "lon"]].to_numpy(dtype=np.float64)
    n = len(coords)
    if n < 2:
        raise ValueError("Need at least two rows to measure redundancy.")

    nearest = np.empty(n)
    within50 = np.empty(n)
    for start, block in _pairwise_km(coords):
        rows = np.arange(start, start + block.shape[0])
        block[np.arange(block.shape[0]), rows] = np.inf  # exclude self
        nearest[start : start + block.shape[0]] = block.min(axis=1)
        within50[start : start + block.shape[0]] = (block <= 0.050).sum(axis=1)

    return RedundancyReport(
        n=n,
        median_nn_m=float(np.median(nearest) * 1000),
        mean_nn_m=float(nearest.mean() * 1000),
        share_within_10m=float(np.mean(nearest <= 0.010)),
        share_within_50m=float(np.mean(nearest <= 0.050)),
        share_within_100m=float(np.mean(nearest <= 0.100)),
        mean_neighbours_within_50m=float(within50.mean()),
    )


def thin_by_spacing(frame: pd.DataFrame, min_spacing_m: float, seed: int = 1337) -> pd.DataFrame:
    """Greedily keep rows that are at least `min_spacing_m` apart.

    Mirrors GSV-Cities' own design constraint (>=100 m between any pair of
    places). Applied to dense Mapillary data it converts a redundant batch into
    a location-diverse one. Rows are shuffled first so the retained set is not
    biased toward whichever tile happens to sort first.
    """
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    coords = shuffled[["lat", "lon"]].to_numpy(dtype=np.float64)

    keep_idx: list[int] = []
    kept_coords = np.empty((0, 2))
    threshold_km = min_spacing_m / 1000.0

    for i, coord in enumerate(coords):
        if len(kept_coords) == 0:
            keep_idx.append(i)
            kept_coords = coord[None, :]
            continue
        d = haversine_km(coord[0], coord[1], kept_coords[:, 0], kept_coords[:, 1])
        if d.min() >= threshold_km:
            keep_idx.append(i)
            kept_coords = np.vstack([kept_coords, coord])

    return shuffled.iloc[keep_idx].reset_index(drop=True)


def location_split(
    frame: pd.DataFrame,
    val_fraction: float = 0.15,
    min_gap_m: float = 100.0,
    seed: int = 1337,
    location_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Split by location with a hard guarantee of spatial separation.

    Held-out locations are removed from training, and then any training row
    within `min_gap_m` of a held-out row is *also* dropped. Without that second
    step a held-out location can still have a near-duplicate in training, which
    is exactly the leakage that makes validation numbers meaningless.
    """
    frame = frame.reset_index(drop=True)

    if location_col and location_col in frame.columns:
        groups = frame[location_col].unique()
        rng = np.random.default_rng(seed)
        val_groups = set(rng.choice(groups, size=max(1, int(round(len(groups) * val_fraction))), replace=False))
        val = frame[frame[location_col].isin(val_groups)].copy()
        train = frame[~frame[location_col].isin(val_groups)].copy()
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(frame))
        n_val = max(1, int(round(len(frame) * val_fraction)))
        val = frame.iloc[idx[:n_val]].copy()
        train = frame.iloc[idx[n_val:]].copy()

    # Enforce the spatial buffer between the two sides.
    val_coords = val[["lat", "lon"]].to_numpy(dtype=np.float64)
    train_coords = train[["lat", "lon"]].to_numpy(dtype=np.float64)
    threshold_km = min_gap_m / 1000.0

    too_close = np.zeros(len(train_coords), dtype=bool)
    for start in range(0, len(train_coords), 512):
        block = train_coords[start : start + 512]
        d = haversine_km(
            block[:, 0][:, None], block[:, 1][:, None], val_coords[:, 0][None, :], val_coords[:, 1][None, :]
        )
        too_close[start : start + len(block)] = d.min(axis=1) < threshold_km

    dropped = int(too_close.sum())
    train = train[~too_close].reset_index(drop=True)
    val = val.reset_index(drop=True)

    return {"train": train, "val": val, "dropped_for_proximity": dropped}


def verify_split_separation(train: pd.DataFrame, val: pd.DataFrame, min_gap_m: float = 100.0) -> dict:
    """Confirm no held-out location sits within `min_gap_m` of a training location."""
    train_coords = train[["lat", "lon"]].to_numpy(dtype=np.float64)
    val_coords = val[["lat", "lon"]].to_numpy(dtype=np.float64)

    mins = []
    for start in range(0, len(val_coords), 512):
        block = val_coords[start : start + 512]
        d = haversine_km(
            block[:, 0][:, None], block[:, 1][:, None], train_coords[:, 0][None, :], train_coords[:, 1][None, :]
        )
        mins.append(d.min(axis=1))
    mins = np.concatenate(mins) * 1000  # metres

    return {
        "min_gap_m_observed": float(mins.min()),
        "median_gap_m": float(np.median(mins)),
        "violations": int((mins < min_gap_m).sum()),
        "passed": bool((mins >= min_gap_m).all()),
    }
