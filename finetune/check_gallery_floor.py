"""Compute the oracle error floor imposed by the candidate GPS gallery.

GeoCLIP predicts by argmax over a fixed gallery of candidate coordinates, so the
best achievable error for an image is the distance from its true location to the
nearest gallery coordinate. If that floor is large, fine-grained metrics like
'accuracy @ 1 km' are structurally unreachable and would be misleading to report
as the headline number.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from geoclip.model.misc import file_dir, load_gps_data

from dataset_index import build_index
from split import make_split


def nearest_gallery_distance_km(coords: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Great-circle distance to the nearest gallery point, vectorised."""
    lat1 = np.radians(coords[:, 0])[:, None]
    lon1 = np.radians(coords[:, 1])[:, None]
    lat2 = np.radians(gallery[:, 0])[None, :]
    lon2 = np.radians(gallery[:, 1])[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def report(name: str, coords: np.ndarray, gallery: np.ndarray) -> None:
    # Chunk over images to keep the distance matrix bounded.
    mins = []
    for start in range(0, len(coords), 64):
        d = nearest_gallery_distance_km(coords[start : start + 64], gallery)
        mins.append(d.min(axis=1))
    mins = np.concatenate(mins)

    print(f"\n--- {name} (n={len(coords)}) ---")
    print(f"  nearest gallery point: median {np.median(mins):.2f} km | "
          f"mean {mins.mean():.2f} km | min {mins.min():.2f} km | max {mins.max():.2f} km")
    for threshold in (1, 25, 200, 750, 2500):
        print(f"  max achievable acc @ {threshold:>4} km : {np.mean(mins <= threshold):.4f}")


if __name__ == "__main__":
    gallery = load_gps_data(os.path.join(file_dir, "gps_gallery", "coordinates_100K.csv")).numpy()
    print(f"gallery size: {gallery.shape[0]:,} coordinates")

    index, _ = build_index()
    split = make_split(index)

    for name in ("train", "val_same_region", "val_untrained_region"):
        frame = split[name]
        report(name, frame[["lat", "lon"]].to_numpy(dtype=np.float64), gallery)
