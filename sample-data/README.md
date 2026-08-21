# sample-data

Held-out evaluation set collected from external public data sources and run
through View Finder's predictors. Nothing here is used for training or
calibration; it is a small, transparent sanity check that the program works
end-to-end against real-world imagery the project did not produce itself.

## `commons/`

16 geotagged JPEG thumbnails downloaded from the
[Wikimedia Commons](https://commons.wikimedia.org/) MediaWiki `geosearch` API
(the same endpoint `local-data-pipeline/landmark_builder.py` uses to build the
landmark gallery). Eight well-known landmarks across six continents were
queried with a 1–3 km radius; the top 2 JPEG/PNG results per landmark were
downloaded at 800 px width.

The downloaded thumbnails do not carry their original EXIF GPS, so the true
coordinates from the search query are re-embedded with `piexif` so the hybrid
predictor's EXIF path can be exercised and the ground truth is self-contained
in each file. Per-image provenance (Commons file title, license, source URL,
true lat/lon) is recorded in `commons/manifest.json`.

Landmarks and their licenses:

| Label              | True lat, lon        | License(s)            |
|--------------------|----------------------|-----------------------|
| eiffel_paris       | 48.8584, 2.2945      | CC BY 3.0             |
| golden_gate_sf     | 37.8199, -122.4783   | CC0                   |
| mount_rainier      | 46.8523, -121.7603   | CC BY-SA 4.0, CC0     |
| colosseum_rome     | 41.8902, 12.4922     | CC BY 2.0             |
| tokyo_tower        | 35.6586, 139.7454    | CC BY-SA 2.0, CC BY 2.0 |
| statue_liberty_nyc | 40.6892, -74.0445    | CC BY-SA 3.0, CC BY-SA 4.0 |
| sydney_opera       | -33.8568, 151.2153   | CC0                   |
| bigben_london      | 51.5007, -0.1246     | CC BY 2.0, CC BY 3.0  |

## `results/`

Per-image JSON output from `viewfinder.main` for each predictor, plus three
roll-up files:

- `results.csv` — hybrid + geoclip rows (one per image × predictor)
- `retrieval_results.csv` — retrieval rows (run separately because the
  OpenCLIP ViT-B-32 encoder segfaults under MPS on this Apple Silicon host,
  so retrieval was forced to CPU via a `sitecustomize.py` shim that
  monkeypatches `resolve_device`)
- `combined_results.csv` — all three predictors merged
- `combined_summary.json` — per-image, per-predictor structured view
- `aggregate_metrics.json` — median/mean error and within-25/200/750 km
  counts per predictor

### Headline numbers (16 images, 8 landmarks)

| Predictor | Median err (km) | Mean err (km) | ≤25 km | ≤200 km | ≤750 km |
|-----------|-----------------|---------------|--------|---------|---------|
| hybrid    | 0.0             | 0.0           | 16/16  | 16/16   | 16/16   |
| geoclip   | 0.6             | 1238.1        | 12/16  | 12/16   | 14/16   |
| retrieval | 7685.9          | 6394.5        | 0/16   | 1/16    | 2/16    |

### How to read these

- **hybrid** is 0 km on every image because every sample carries embedded
  EXIF GPS, and the hybrid predictor's first evidence check is "emit
  directly when valid EXIF GPS is present". This confirms the fail-closed
  contract works on real files; it is not a meaningful accuracy number for
  the visual pipeline.
- **geoclip** lands within 25 km on 12/16 landmarks and within 750 km on
  14/16. The two large misses (Eiffel #2 → Algiers 1346 km, Tokyo Tower #2
  → Hiroshima 679 km, Sydney Opera #2 → Valencia 17 415 km) are the
  known GeoCLIP failure mode of snapping to a learned coordinate that looks
  vaguely similar; the README already documents this.
- **retrieval** is structurally wrong on 14/16 images because the only
  committed gallery (`gallery-pipeline/data/target-regions/gallery/v2`)
  covers Puget Sound and Avenue of the Giants — 1305 reference photos in
  two US regions. Every query from Paris, Rome, Tokyo, NYC, Sydney, or
  London has no matching reference, so the nearest neighbour is whichever
  Washington-state photo looks least dissimilar. This is exactly the
  "tiny-gallery" failure the README warns about; the confidence score
  stays high (74–88%) because the returned neighbours still agree with
  each other. The two sub-750 km hits (Mount Rainier, 119 km and 748 km)
  are the only queries whose true location is even roughly near the
  gallery's coverage.

## Reproducing

```bash
# Collect (needs network; respects Wikimedia rate limits with sleeps)
python sample-data/scripts/collect_commons.py sample-data/commons 2

# Run hybrid + geoclip
python sample-data/scripts/run_samples.py sample-data/commons/manifest.json \
    gallery-pipeline/data/target-regions/gallery/v2 sample-data/results

# Run retrieval (CPU-forced on Apple Silicon)
PYTHONPATH=sample-data/scripts KMP_DUPLICATE_LIB_OK=TRUE \
    python sample-data/scripts/run_retrieval_only.py sample-data/commons/manifest.json \
    gallery-pipeline/data/target-regions/gallery/v2 sample-data/results

# Merge
python sample-data/scripts/merge_results.py
```

`sample-data/scripts/sitecustomize.py` is the CPU-forcing shim; it is picked
up automatically when `sample-data/scripts` is on `PYTHONPATH` (the runner
scripts set this themselves).
