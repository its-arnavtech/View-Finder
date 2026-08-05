"""Quantify near-duplicate GPS coordinates in the training batch.

The contrastive loss treats every non-matching location in (batch + GPS queue)
as a negative. Mapillary sequences are dense -- many images sit metres apart --
so a positive's "negatives" often include locations that are physically the same
place. That puts an irreducible floor under the loss and adds gradient noise.
Knowing the scale of it is what separates "training is broken" from "training is
working against a data property".
"""

from __future__ import annotations

import numpy as np

from dataset_index import build_index
from split import make_split


def pairwise_km(coords: np.ndarray) -> np.ndarray:
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


if __name__ == "__main__":
    index, _ = build_index()
    train = make_split(index)["train"]
    coords = train[["lat", "lon"]].to_numpy(dtype=np.float64)

    d = pairwise_km(coords)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)

    print(f"training images: {len(coords)}")
    print(f"nearest same-batch neighbour distance:")
    print(f"  median {np.median(nearest) * 1000:.1f} m | mean {nearest.mean() * 1000:.1f} m")
    for metres in (10, 25, 50, 100, 250, 1000):
        share = np.mean(nearest <= metres / 1000.0)
        print(f"  share with a neighbour within {metres:>4} m : {share:.3f}")

    # How many negatives are effectively indistinguishable from the positive?
    for metres in (50, 100):
        collisions = (d <= metres / 1000.0).sum(axis=1)
        print(f"\nnegatives within {metres} m of the positive (per image):")
        print(f"  median {np.median(collisions):.0f} | mean {collisions.mean():.1f} | max {collisions.max()}")
