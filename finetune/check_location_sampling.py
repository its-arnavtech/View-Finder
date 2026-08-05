"""Validate location_sampling against the Puget/Humboldt data with known properties.

The batch01 training set measured 77% of images having a neighbour within 10 m.
This exercises the redundancy measurement, the >=100 m thinning, and the
leak-proof split against that known baseline before any of it is pointed at a
new dataset.
"""

from __future__ import annotations

from dataset_index import build_index
from location_sampling import (
    location_split,
    measure_redundancy,
    thin_by_spacing,
    verify_split_separation,
)
from split import make_split

MIN_SPACING_M = 100.0

if __name__ == "__main__":
    index, _ = build_index()
    train = make_split(index)["train"]

    before = measure_redundancy(train)
    print(before.format("batch01 training set (as trained)"))
    print()

    thinned = thin_by_spacing(train, min_spacing_m=MIN_SPACING_M)
    print(f"thinning to >={MIN_SPACING_M:.0f} m spacing: {len(train)} -> {len(thinned)} images "
          f"({len(thinned) / len(train):.1%} retained)")
    print()

    after = measure_redundancy(thinned)
    print(after.format(f"after >={MIN_SPACING_M:.0f} m thinning"))
    print()

    split = location_split(thinned, val_fraction=0.15, min_gap_m=MIN_SPACING_M)
    print(f"location split: train={len(split['train'])} val={len(split['val'])} "
          f"(dropped {split['dropped_for_proximity']} training rows within {MIN_SPACING_M:.0f} m of held-out)")

    check = verify_split_separation(split["train"], split["val"], min_gap_m=MIN_SPACING_M)
    print(f"separation check: passed={check['passed']} | closest train/val pair "
          f"{check['min_gap_m_observed']:.1f} m | median {check['median_gap_m']:.1f} m | "
          f"violations {check['violations']}")
