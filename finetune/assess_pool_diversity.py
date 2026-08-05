"""How many spatially distinct locations does the existing pool actually contain?

batch01 trained on 1365 images that collapsed to 184 distinct locations at 100 m
resolution. Before scraping anything new, measure whether the *full* pool --
including the deliberately globally-spread global-breadth / global-test /
live-smoke datasets -- already yields a location-diverse batch.
"""

from __future__ import annotations

import pandas as pd

from dataset_index import build_index
from location_sampling import measure_redundancy, thin_by_spacing

SPACINGS_M = (50.0, 100.0, 250.0, 1000.0)

if __name__ == "__main__":
    index, stats = build_index()
    print(f"full pool: {len(index)} images across {index['tile_id'].nunique()} tiles\n")

    report = measure_redundancy(index)
    print(report.format("full pool, unthinned"))
    print()

    rows = []
    for spacing in SPACINGS_M:
        thinned = thin_by_spacing(index, min_spacing_m=spacing)
        by_dataset = thinned["dataset"].value_counts().to_dict()
        rows.append({
            "min_spacing_m": spacing,
            "locations": len(thinned),
            "retained": f"{len(thinned) / len(index):.1%}",
            **by_dataset,
        })

    print("distinct locations retained at each spacing threshold:")
    print(pd.DataFrame(rows).fillna(0).to_string(index=False))
    print()

    # Where does the diversity actually come from?
    thinned100 = thin_by_spacing(index, min_spacing_m=100.0)
    print("at >=100 m spacing, locations per dataset:")
    print(thinned100["dataset"].value_counts().to_string())
    print()
    print("at >=100 m spacing, locations per region:")
    print(thinned100["region"].value_counts().to_string())
