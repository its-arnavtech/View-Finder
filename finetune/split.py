"""Held-out split construction.

Leakage matters a lot here. Mapillary imagery inside one tile is a sequence of
street-level captures taken metres apart, so two images from the same tile are
often near-duplicates. A random per-image split would put near-duplicates of
every validation image into the training set and report a meaninglessly
optimistic validation number.

So we split by *tile*, holding out whole tiles. To satisfy "unseen images from
the same regions", held-out tiles are drawn from the large multi-tile areas that
are also represented in training -- the model trains on Puget Sound / Avenue of
the Giants / Chicago Loop / global_0007 tiles and is validated on *different*
tiles from those same areas.

We additionally carve a small control set from areas that are NOT trained on at
all, to detect whether fine-tuning on a geographically concentrated batch
degrades the location encoder elsewhere in the world.
"""

from __future__ import annotations

import re

import pandas as pd

# Areas with enough tiles to hold some out while still training on the area.
SAME_REGION_AREAS = ("puget_hood_canal", "avenue_of_giants", "global_0007", "chicago_loop")

# Fraction of each large area's tiles reserved for validation.
VAL_TILE_FRACTION = 0.18


def area_of(tile_id: str) -> str:
    """'puget_hood_canal_p47d40_m122d70_p0d05' -> 'puget_hood_canal'."""
    return re.sub(r"_[pm]\d+d\d+_[pm]\d+d\d+_[pm]?\d+d\d+$", "", tile_id)


def make_split(index: pd.DataFrame, seed: int = 1337) -> dict[str, pd.DataFrame]:
    """Return {'train', 'val_same_region', 'val_untrained_region'} frames.

    The split is deterministic given `seed`.
    """
    index = index.copy()
    index["area"] = index["tile_id"].map(area_of)

    rng = pd.Series(range(len(index)))  # placeholder to keep pandas import honest
    del rng

    val_tiles: list[str] = []
    for area in SAME_REGION_AREAS:
        tiles = sorted(index.loc[index["area"] == area, "tile_id"].unique())
        if len(tiles) < 3:
            # Not enough tiles to hold any out without losing the area from training.
            continue
        n_val = max(1, int(round(len(tiles) * VAL_TILE_FRACTION)))
        # Deterministic sample, stable across runs.
        chosen = pd.Series(tiles).sample(n=n_val, random_state=seed).tolist()
        val_tiles.extend(chosen)

    val_tile_set = set(val_tiles)

    # Untrained-region control: small areas that we exclude from training entirely.
    trained_areas = set(SAME_REGION_AREAS)
    control_mask = ~index["area"].isin(trained_areas)
    control = index[control_mask].copy()

    same_region_val = index[index["tile_id"].isin(val_tile_set)].copy()
    train = index[~index["tile_id"].isin(val_tile_set) & index["area"].isin(trained_areas)].copy()

    # Sanity: no tile may appear on both sides.
    overlap = set(train["tile_id"]) & set(same_region_val["tile_id"])
    if overlap:
        raise RuntimeError(f"Tile leakage between train and val: {sorted(overlap)[:5]}")
    overlap_imgs = set(train["image_id"]) & set(same_region_val["image_id"])
    if overlap_imgs:
        raise RuntimeError(f"Image leakage between train and val: {len(overlap_imgs)} ids")

    return {
        "train": train.reset_index(drop=True),
        "val_same_region": same_region_val.reset_index(drop=True),
        "val_untrained_region": control.reset_index(drop=True),
    }


def describe_split(split: dict[str, pd.DataFrame]) -> str:
    lines = []
    for name, frame in split.items():
        areas = frame.groupby("area").size().sort_values(ascending=False)
        area_str = ", ".join(f"{a}={n}" for a, n in areas.items())
        lines.append(
            f"{name:>22} : {len(frame):>5} images | {frame['tile_id'].nunique():>3} tiles | {area_str}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    from dataset_index import build_index

    index, _ = build_index()
    split = make_split(index)
    print(describe_split(split))
