"""Summarize the index by 'area' (the tile_id name prefix) to design a leak-free split."""

import re

from dataset_index import build_index


def area_of(tile_id: str) -> str:
    """tile_id looks like '<area_name>_p47d40_m122d70_p0d05' -> '<area_name>'."""
    return re.sub(r"_[pm]\d+d\d+_[pm]\d+d\d+_[pm]?\d+d\d+$", "", tile_id)


if __name__ == "__main__":
    index, _ = build_index()
    index["area"] = index["tile_id"].map(area_of)

    grouped = index.groupby("area").agg(
        images=("image_id", "size"),
        tiles=("tile_id", "nunique"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        region=("region", "first"),
    ).sort_values("images", ascending=False)

    print(grouped.to_string())
    print()
    print(f"total areas: {len(grouped)}   total images: {len(index)}")
    print()
    print("images-per-tile distribution:")
    print(index.groupby("tile_id").size().describe().to_string())
