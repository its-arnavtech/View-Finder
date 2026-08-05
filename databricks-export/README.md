# View Finder — Databricks export

Summary and per-image analysis data from the GeoCLIP fine-tuning, gallery-growth
and ensemble-calibration work, flattened into small CSVs for manual upload to
Databricks Community Edition (drag-and-drop table creation — no API, no cloud
storage, no credentials involved).

All seven files together are well under 1 MB. Every file has a header row and
uses UTF-8 with `,` separators, so Databricks' "first row is header" +
"infer schema" defaults work as-is.

Regenerate the seven CSVs from committed inputs (no GPU needed):

```bash
python databricks-export/_build_exports.py
```

`_confidences.csv` is committed because rebuilding it means re-running both
predictors, which needs a GPU and the cached model weights. Only redo that step
if the predictors or the sample change:

```bash
python databricks-export/_build_confidences.py
```

Both scripts resolve paths relative to their own location, so they work from any
checkout directory.

---

## 1. `benchmark_results.csv` — 1,000 rows

Per-image results on the external validation set: 1,000 globally diverse images
(median nearest-neighbour spacing ~97 km) that the retrieval gallery does **not**
cover and that were never used for training. Scored against **gallery v2**.

| Column | Meaning |
|---|---|
| `image_id` | image filename stem |
| `source_dataset` | `kaggle/google-street-view` (712) or `kaggle/4k-benchmark-images` (288) |
| `true_lat` / `true_lon` | ground truth |
| `geoclip_pred_lat` / `_lon` | GeoCLIP top-1 prediction |
| `geoclip_confidence` | aggregator confidence, 0–100 |
| `geoclip_error_km` | haversine error |
| `retrieval_pred_lat` / `_lon` | retrieval top-1 prediction |
| `retrieval_confidence` | aggregator confidence, 0–100 |
| `retrieval_top_similarity` | raw top-1 cosine similarity vs the closest gallery photo |
| `retrieval_error_km` | haversine error |
| `ensemble_pred_lat` / `_lon` | ensemble output |
| `ensemble_source_used` | `retrieval` if `top_similarity >= 0.825`, else `geoclip` |
| `ensemble_error_km` | haversine error of the chosen prediction |

**Suggested visualization:** scatter/box comparison of `*_error_km` by predictor.

⚠️ The most informative chart here is **`retrieval_confidence` vs
`retrieval_top_similarity`, colored by `retrieval_error_km`.** On this set
retrieval's confidence averages **85.5** (minimum 74.3) while its median error is
**9,059 km** — confidence measures agreement among returned neighbours, so a
small gallery can agree strongly and be entirely wrong. `top_similarity` is the
signal that actually discriminates; confidence is not. This is why the ensemble
gates on similarity rather than confidence.

---

## 2. `accuracy_summary.csv` — 126 rows (tidy/long)

| Column | Meaning |
|---|---|
| `predictor` | `geoclip_alone`, `retrieval_alone`, `ensemble` |
| `gallery_version` | `v1` (1,719 images) or `v2` (3,314) |
| `population` | `external`, `probe_a`, `probe_b` — **filter on this** |
| `n` | images in that population |
| `metric` | `acc@1km`, `acc@25km`, `acc@200km`, `acc@750km`, `acc@2500km`, `median_error_km`, `mean_error_km` |
| `value` | metric value |

**Suggested visualization:** grouped bar chart — `metric` on x, `value` on y,
grouped by `predictor`, faceted or series-split by `gallery_version`.

⚠️ `population` is not in the originally requested schema but is **required for
correctness** — the three populations answer different questions and must not be
averaged together. Filter to `population = 'external'` for the honest
global-accuracy comparison; `probe_a` for in-coverage behaviour. `probe_b` is a
near-duplicate upper bound, useful as a ceiling but not a realistic estimate.

---

## 3. `threshold_calibration.csv` — 30 rows

The similarity-vs-error data behind the 0.825 trust threshold, for both galleries.

| Column | Meaning |
|---|---|
| `population` | `external` (not covered), `probe_a` (in coverage, honest), `probe_b` (near-duplicate upper bound) |
| `gallery_version` | `v1` / `v2` |
| `top_similarity_bucket` | label, e.g. `[0.65,0.80)` |
| `bucket_low` / `bucket_high` | numeric edges, for sorting/plotting on a real axis |
| `n` | images in bucket |
| `median_error_km` | retrieval's median error in that bucket |
| `geoclip_median_error_km` | GeoCLIP's median on the same images, as baseline |
| `retrieval_win_rate` | fraction where retrieval beat GeoCLIP |

**Suggested visualization:** line/scatter of `bucket_low` vs `median_error_km`,
one series per `population × gallery_version`, so the v1 and v2 distributions
overlay and the shift is visible.

The key readings: out of coverage, retrieval's median error is ~9,000–10,000 km
in **every** bucket from 0.50 to 0.80 — `top_similarity` carries no quality
signal there. It only becomes meaningful at the very top of the range, which is
what makes a high threshold work at all.

---

## 4. `false_accept_ceiling.csv` — 2 rows

The headline finding, isolated.

| Column | v1 | v2 |
|---|---|---|
| `gallery_size` | 1,719 | 3,314 |
| `out_of_coverage_max_similarity` | 0.8029 | **0.8296** |
| `out_of_coverage_mean_similarity` | 0.6683 | 0.7035 |
| `out_of_coverage_p99_similarity` | 0.7774 | 0.8034 |
| `in_coverage_median_similarity` | 0.8506 | 0.8629 |
| `in_coverage_n` | 24 | 93 |
| `external_images_above_threshold` | **0** / 1000 | **1** / 1000 |
| `in_coverage_above_threshold` | 15 / 24 | 68 / 93 |

**Suggested visualization:** `gallery_size` on x, two lines —
`out_of_coverage_max_similarity` and `external_images_above_threshold`.

**What it shows:** doubling the gallery raised the *false-accept ceiling*
(0.803 → 0.830, now above the 0.825 threshold) while moving external clearance
only from 0/1000 to 1/1000. A bigger gallery gives every query more chances at a
spuriously similar photo, so the ceiling rises with gallery **size** regardless
of which regions are added — while the benefit accrues only where coverage
actually lands. The threshold held at 0.825 because the in-coverage median rose
too (0.851 → 0.863) and the two shifts cancelled; that is a coincidence of this
particular growth step, not a stable property.

⚠️ Only two points. Treat as directional evidence, not a fitted scaling law.

---

## 5. `gallery_coverage.csv` — 3,314 rows

Every image in benchmark-gallery v2, labelled by when it entered.

| Column | Meaning |
|---|---|
| `image_id` | Mapillary image id |
| `lat` / `lon` | location |
| `region_label` | `puget-sound` (1088), `europe` (1064), `norcal` (274), `humboldt` (217), `north-america-other` (151), `chicago` (140), `east-asia` (124), `south-asia` (118), `africa` (70), `south-america` (51), `oceania` (17) |
| `gallery_version` | `v1` = present in the 1,719-image gallery; `v2` = added in the expansion (1,595 new) |
| `tile_id` | source scrape tile |

**Suggested visualization:** world map, points colored by `gallery_version` (or
`region_label`), to show coverage growth. Africa and South America appear only in
`v2` — both were at literally zero before.

Named regions come from the hand-configured scrape areas; `global_*` tiles are
labelled from coordinates, so boundaries are approximate.

---

## 6. `training_trend.csv` — 4 rows

The three fine-tuning runs plus the out-of-distribution control, deliberately in
one table.

| Column | Meaning |
|---|---|
| `run_name` | `batch01_es`, `breadth01`, `breadth02`, `external_transfer` |
| `training_locations` | distinct locations at ≥100 m spacing |
| `best_epoch` / `useful_steps` | where held-out error bottomed before regressing |
| `n` | held-out set size |
| `held_out_median_error_before_km` / `_after_km` | pretrained vs fine-tuned |
| `pct_improvement` | positive = better |
| `sign_test_z` | paired sign test, external row only |
| `result` | `in_distribution_improvement` or `null_effect` |

**Suggested visualization:** bar chart of `pct_improvement` by `run_name`, or
`training_locations` vs `useful_steps` for the scaling trend — with the
`external_transfer` bar included so both stories are on one axis.

**Read these two facts together.** In-distribution, more locations bought
proportionally more useful training (184→543→730 locations; 20→48→88 steps before
overfitting). On 1,000 independent images the same checkpoint produced a **null
effect**: 250 images closer, 255 farther, 495 identical, sign-test z = −0.22.
The `-11.26%` on that row is a median shift driven by a few large moves and is
**not** a significant regression — the honest reading is "no effect", which is
why `result` says `null_effect` rather than something derived from the percentage.

---

## 7. `pipeline_reliability.csv` — 8 rows

Real bugs found and fixed, for a summary table (**not** a chart).

Columns: `bug_description`, `session_found`, `component`, `fix_description`,
`impact`.

Covers the shared retry budget, the single-tile run abort, undividable dense
tiles, fixed-step checkpointing missing the optimum, loss-vs-held-out divergence,
the ImageNet-vs-CLIP normalization mismatch, benchmarking retrieval against its
own gallery, and the oversized-binary gitignore gap.

---

## Provenance and caveats

Derived from `finetune/work/` calibration CSVs, validation JSONs, run summaries,
and gallery metadata. `_build_confidences.py` recomputes the aggregator
confidence values, which the original calibration runs did not persist (they kept
only top-1 candidates, and confidence needs the top-k list) — predictions are
unchanged, only the confidence columns are new.

Two values are hand-entered rather than computed here, both annotated in
`_build_exports.py`: the sign-test z of −0.22 (from
`finetune/work/external_paired.json`) and the `pipeline_reliability` narrative.

Sample-size caveats worth carrying into any conclusion:

- `probe_a` is **n=24 for v1** and **n=93 for v2**. These are the only images
  available that sit inside gallery coverage without *being* gallery entries
  (everything else there was either a gallery member — retrieval returns the
  query itself at similarity 1.0 and 0.00 km, measuring memorisation — or was
  dropped by the pipeline's own filters). They were excluded for blur/indoor/
  tile-cap, so they are if anything harder than typical.
- `false_accept_ceiling.csv` has two data points.
- The external set is hard for everything: pretrained GeoCLIP reaches only
  `acc@25km = 0.007` on it. Predictor comparisons are fair (identical images),
  but absolute values reflect a difficult domain of random global street scenes.
