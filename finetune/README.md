# GeoCLIP location-encoder fine-tuning

Lightweight fine-tuning of GeoCLIP for View Finder on a single 8 GB GPU
(RTX 4060 Laptop). Only the **location encoder** is trained; the CLIP visual
tower and its projection MLP stay frozen at their pretrained weights.

## Why only the location encoder

`geoclip.model.GeoCLIP` has three parameter groups:

| Component | Params | Trained here |
|---|---|---|
| `image_encoder.CLIP` (ViT-L/14 visual tower) | ~303.9 M | no — frozen |
| `image_encoder.mlp` (768→768→512 projection) | 984 K | no — frozen |
| `location_encoder` (GPS → 512-d) | 9.45 M | **yes** |
| `logit_scale` (contrastive temperature) | 1 | no — frozen |

That is **9,449,472 / 314,400,002 = 3.006%** of the model.

Two freezing details are easy to get wrong:

- `ImageEncoder.__init__` already sets `requires_grad = False` on CLIP's
  parameters, but **not** on `image_encoder.mlp`. That MLP is pretrained image-side
  weight (`image_encoder_mlp_weights.pth`), so leaving it trainable would mean
  fine-tuning the image side too. `model_setup.freeze_image_side` freezes it explicitly.
- `logit_scale` is a bare `nn.Parameter` and is trainable by default. It is frozen
  so the similarity temperature is identical between the pretrained and fine-tuned
  models, keeping the before/after comparison attributable to the location encoder alone.

`model_setup.assert_freeze_contract` hard-fails if anything outside
`location_encoder.*` is trainable.

The location encoder is three parallel "capsules" (random Fourier features at
σ = 2⁰, 2⁴, 2⁸, each feeding a 3-layer 1024-wide MLP and a 512-d head), summed.

## Measured GPU envelope

Peak reserved VRAM for a real training step (frozen CLIP forward under `no_grad`
+ location encoder forward/backward + AdamW), GPS queue 1024:

| Batch | fp32 peak | fp32 img/s | AMP peak | AMP img/s |
|---|---|---|---|---|
| 64  | 2.80 GB | 39.5 | 2.18 GB | 105.6 |
| 128 | 4.01 GB | 35.3 | **3.14 GB** | **107.0** |
| 256 | 6.66 GB | 34.7 | 5.15 GB | 106.6 |
| 512 | 11.81 GB ✗ | — | 8.67 GB ✗ | — |

**AMP (fp16 autocast on the frozen forward) is ~3× faster and uses less memory**,
with no effect on predictions — `verify_eval_parity.py` gets identical gallery
picks under fp32 and fp16.

⚠️ **On Windows WDDM, exceeding VRAM does not raise `CUDA out of memory`.** The
driver silently spills into shared system RAM and keeps running, just far slower.
Batch 512 reported "ok" while reserving 11.8 GB on an 8 GB card. Peak *reserved
bytes*, not an OOM exception, is the real budget signal here.

Operating point: **batch 128 + AMP** — 3.14 GB peak leaves headroom for eval, and
smaller batches give more optimizer steps on a small dataset. 256 is the tested
maximum that comfortably fits.

## Data and the split

Source is existing gallery-pipeline Mapillary data (already validated
`(image, lat, lon)`), consolidated by `dataset_index.py`: 4303 rows across all
gallery versions → 1941 unique images present on disk across 113 tiles.

`split.py` splits **by tile, not by image**. Mapillary imagery within a tile is a
sequence of captures metres apart; a random per-image split would put
near-duplicates of every validation image into training and report a
meaninglessly optimistic number.

| Slice | Images | Tiles | Purpose |
|---|---|---|---|
| `train` | 1365 | 72 | the batch |
| `val_same_region` | 236 | 15 | **held-out** — same areas, tiles never trained on |
| `val_untrained_region` | 340 | 26 | control — areas never trained on at all |

The control slice detects whether fine-tuning on geographically concentrated data
degrades the location encoder elsewhere in the world.

## Two data properties that bound the results

Both are measured, not assumed, and both matter for reading the numbers.

**1. Gallery resolution floor** (`check_gallery_floor.py`). GeoCLIP predicts by
argmax over a fixed 100 K-coordinate gallery, so the nearest gallery point caps
achievable accuracy. Near our regions that spacing is ~5–6 km median:

- median achievable error: **~5.5 km** — median error can never beat this
- max achievable `acc @ 1 km`: **~13.6%** — structurally near-unreachable, not a useful headline
- max achievable `acc @ 25 km`: ~88%

So **median error and `acc @ 25 km` are the meaningful metrics**; `acc @ 1 km` is
reported only for completeness.

**2. Near-duplicate coordinates** (`check_coord_collisions.py`). The contrastive
loss treats every other location in (batch + queue) as a negative, but:

- 77% of training images have another within **10 m**; median nearest neighbour **4.8 m**
- each image has ~11 other images within 50 m

Those "negatives" are physically the same place and the location encoder cannot
separate them. This puts an irreducible floor on the loss of roughly
`ln(12) ≈ 2.5`. **A loss plateauing near ~2.5 is the data's intrinsic ambiguity,
not a broken loop.**

## Evaluation

`evaluate.py` reproduces GeoCLIP's deployed inference: score the image embedding
against location embeddings of the 100 K GPS gallery, take argmax, measure
geodesic error.

Because the image side is frozen and never updated, a validation image's
embedding is **bit-identical before and after** fine-tuning. Features are encoded
once and reused for both measurements, so the only thing differing between
"before" and "after" is the location encoder's weights.

`verify_eval_parity.py` confirms this path matches `GeoCLIP.predict()` on
**23/24** images (the one disagreement is a near-tie in a flat softmax).

Note `clip_transforms.py`: geoclip's own `train/dataloader.py` normalizes with
**ImageNet** statistics while `GeoCLIP.predict()` uses **CLIP** statistics. Using
ImageNet stats would feed the frozen tower out-of-distribution inputs, understate
the pretrained baseline, and inflate the apparent improvement. We use CLIP
statistics everywhere.

## Batch-ingest-train-discard

`train_batch.py` processes exactly one batch per invocation:

```
stage batch → eval pretrained baseline → train → checkpoint
   → eval fine-tuned → report before/after → optionally discard raw data
```

The **location encoder checkpoint is the only artifact that persists** across
batches (`checkpoints/<run>/`). Per-step loss goes to `work/<run>/train_log.jsonl`;
the run summary including the held-out trajectory goes to `work/<run>/summary.json`.

### Early stopping is mandatory here, not optional

Training loss on this data keeps falling long after held-out error starts
regressing. The first 30-epoch run (`batch01`) demonstrates it:

| Epoch | Train loss | Held-out median | acc @ 25 km |
|---|---|---|---|
| 4 | 4.219 | **45.1 km** | 0.339 |
| 9 | 3.851 | 166.0 km | 0.259 |
| 14 | 3.635 | 234.6 km | 0.242 |
| 29 | 3.434 | 239.4 km | 0.199 |

Loss fell monotonically the whole time. The final weights were **worse than
pretrained** (median 126.5 → 239.4 km), and the control slice degraded too
(mean 982 → 1270 km), so this is over-specialization, not a harmless overfit.

Consequently:

- checkpoints are written at the **end of every epoch** (a fixed step interval
  silently skipped the optimum — `batch01`'s best epoch was step 50 with saves
  only at 100/200/300),
- `location_encoder_best.pth` tracks the best held-out median and is overwritten
  on improvement,
- `--early-stop-patience` (default 3) ends the run, and the headline before/after
  **reloads the best checkpoint** rather than reporting final weights.

### Result with early stopping (`batch01_es`)

Stopped at epoch 4; best state was **epoch 1 (20 optimizer steps)**.

| Held-out (same regions, unseen tiles, n=236) | Pretrained | Fine-tuned | Δ |
|---|---|---|---|
| median error | 126.53 km | **40.48 km** | −68% |
| mean error | 390.60 km | 138.89 km | −64% |
| p75 error | 403.24 km | 101.06 km | −75% |
| acc @ 25 km | 0.0890 | **0.3856** | 4.3× |
| acc @ 200 km | 0.5975 | 0.9068 | +0.31 |

The control slice stayed roughly neutral (median 283 → 275 km, mean 982 → 897 km)
versus the severe degradation in the 30-epoch run — **early stopping prevented
global drift, not just local overfitting**. `acc @ 1 km` fell (0.055 → 0.030) but
is capped at ~13.6% by gallery resolution and carries little signal.

## Location diversity is the binding constraint

Image count overstates how much the location encoder actually sees
(`location_sampling.py`, `assess_pool_diversity.py`):

| Set | Images | Distinct locations at ≥100 m |
|---|---|---|
| `batch01` training set | 1365 | **184** (13.5%) |
| entire gallery-pipeline pool | 1941 | **325** (16.7%) |

184 places is why 20 optimizer steps was the optimum. More images from the same
tiles cannot fix this; only geographic breadth can.

`config/tiling_config_finetune_breadth.yaml` in gallery-pipeline targets that
directly — worldwide `bounds: null`, `ordering: spread` (one tile per populated
supercell per pass), and `max_images_per_tile: 6` instead of 20, buying ~3× more
distinct locations per image downloaded. `pull_breadth_batch.py` runs the extract
stage alone; validate/transform/load build FAISS gallery artifacts that
fine-tuning does not need.

`train_breadth_batch.py` then thins to a minimum spacing, splits **by distance**,
drops any training row inside the buffer around a held-out row, and verifies the
separation before training — refusing to run on a leaking split.

### Result: breadth batch (`breadth01`)

First worldwide pull: 624 images across 303 tiles in 28 min (Europe, Indonesia,
Mexico, Philippines, China, Australia, US, India, Turkey). Combined with the
existing pool: 2580 images → **639 locations** at ≥100 m (24.8% retained, vs
16.7% for the existing pool alone). Split verified at 101.1 m closest train/val
pair, 0 violations.

| Held-out (≥100 m separated locations, n=96) | Pretrained | Fine-tuned | Δ |
|---|---|---|---|
| median error | 270.58 km | **117.49 km** | −57% |
| mean error | 799.10 km | 471.79 km | −41% |
| p25 error | 116.93 km | 27.91 km | −76% |
| acc @ 1 km | 0.0208 | 0.0521 | +150% |
| acc @ 25 km | 0.0833 | **0.2396** | 2.9× |
| acc @ 200 km | 0.3854 | 0.6146 | +0.23 |

**All nine metrics improved**, including `acc @ 1 km` — which regressed in the
dense-tile run.

The headline finding is *when* overfitting starts:

| Run | Training locations | Best epoch | Useful steps |
|---|---|---|---|
| `batch01_es` (dense tiles) | 184 | 1 | 20 |
| `breadth01` (worldwide) | 543 | 5 | 48 |

2.9× the locations bought 2.4× more useful training before the held-out metric
turned. That is the diversity hypothesis confirmed on the metric that matters.

### Result: second breadth batch (`breadth02`)

Second pull added 512 images / 290 tiles (two sessions; the first crashed on a
connection reset — see "Pull resilience" below). Breadth catalog: 642 → **1154
images / 599 tiles**. Combined with the existing pool: 3092 images → **859
locations** at ≥100 m (27.8% retained, up from 24.8%). Split verified at 102.6 m
closest pair, 0 violations.

| Held-out (≥100 m separated, n=129) | Pretrained | Fine-tuned | Δ |
|---|---|---|---|
| median error | 286.16 km | **160.33 km** | −44% |
| mean error | 1270.76 km | 766.40 km | −40% |
| p75 error | 753.12 km | 463.67 km | −38% |
| acc @ 25 km | 0.0620 | **0.1163** | 1.9× |
| acc @ 200 km | 0.4186 | 0.5504 | +0.13 |
| acc @ 2500 km | 0.8992 | 0.9457 | +0.05 |

### The trend holds

| Run | Training locations | Best epoch | Useful steps |
|---|---|---|---|
| `batch01_es` | 184 | 1 | 20 |
| `breadth01` | 543 | 5 | 48 |
| `breadth02` | 730 | 7 | 88 |

Locations up 4.0× from batch01 to breadth02; useful optimizer steps before the
held-out metric turns up 4.4×. The relationship has stayed close to linear across
three runs spanning a 4× range, so location count — not image count — remains the
binding constraint, and more breadth data is still buying more trainable signal.

⚠️ Each run has a **different held-out set** (they are re-split from a growing
pool), so absolute km are not comparable run-to-run. Best-epoch and relative gain
are the comparable quantities. Note `breadth02`'s harder val slice: its
pretrained baseline is 286 km vs `breadth01`'s 271 km on the same metric.

## Ensemble trust threshold (calibrated, not assumed)

`viewfinder/ensemble_geolocator.py` lets retrieval override GeoCLIP only where
the gallery genuinely has a reference match. The cut is calibrated from
per-image data, because the intuitive 0.6–0.65 is badly wrong here.

### The calibration data had to be rebuilt

Neither cached benchmark file could answer this. `external_paired.json` holds
per-image errors for the two GeoCLIP variants only; `external_benchmark_results.json`
holds aggregate metrics with no per-image rows. Neither ever recorded retrieval's
`top_similarity` — the earlier harness kept `candidate.lat`/`lon` and discarded
`candidate.score`. `calibrate_ensemble.py` re-derives all of it.

### There is no clean in-distribution held-out data

Every image inside `benchmark-gallery/v1`'s footprint is either a gallery entry
(retrieval returns the query itself: similarity 1.0, 0.00 km — memorisation, not
geolocation) or was dropped by the pipeline's filters. The nearest non-gallery
image in the whole pool sits **36.9 km** from any gallery point; zero within
25 km. So breadth01/breadth02's held-out slices cannot serve — 28.7% of
breadth02's val slice are gallery members, and the other 71.3% are all far
outside coverage.

Two separate probes replace it (`build_incoverage_probes.py`):

| Probe | n | What it is | Use |
|---|---|---|---|
| **A** | 24 | dropped for blur/indoor/tile-cap — genuinely different photos | picks the threshold |
| **B** | 282 | perceptual-hash near-duplicates of gallery images | upper bound only |

### Measured populations

| population | min | median | p99 | max | retrieval median error |
|---|---|---|---|---|---|
| External, not covered (n=1000) | 0.487 | 0.672 | 0.777 | **0.803** | 10,009 km |
| Probe A, in-coverage (n=24) | 0.632 | **0.851** | — | 0.962 | 0.02 km |
| Probe B, near-duplicate (n=282) | 0.892 | 0.976 | — | 1.000 | 0.01 km |

**Out of coverage, `top_similarity` carries no quality signal at all.** Retrieval's
median error is ~10,000 km in every bucket from 0.50 to 0.80, and it beats GeoCLIP
on only 9–11% of images regardless of score. The populations separate only at the
very top, because no out-of-coverage image in 1000 exceeded 0.803.

### Threshold trade-off

| threshold | external accepted | external median cost | in-coverage accepted | in-coverage gain |
|---|---|---|---|---|
| 0.600 | 903/1000 | **+8,495 km** | 24/24 | +160 km |
| 0.650 | 649/1000 | +7,887 km | 23/24 | +160 km |
| 0.750 | 45/1000 | +149 km | 21/24 | +160 km |
| 0.800 | 1/1000 | +6 km | 18/24 | +160 km |
| **0.825** | **0/1000** | **0.00 km** | **15/24 (62.5%)** | **+159.8 km** |
| 0.900 | 0/1000 | 0.00 km | 6/24 | +70 km |

`DEFAULT_TRUST_THRESHOLD = 0.825` — above the observed out-of-coverage ceiling,
below the in-coverage median. A 0.6 cut would route 903 of 1000 external images
to retrieval and push their median error from 1,292 km to 9,787 km.

This is a *different quantity* from `aggregator.DEFAULT_WEAK_SIMILARITY_THRESHOLD`
(also 0.6), which re-ranks clusters within a retrieval result and does not choose
between predictors.

`DEFAULT_MENTION_FLOOR = 0.65`: below this, retrieval beat GeoCLIP on 10.6% of
external images at 10,048 km median error, so its answer is dropped rather than
offered as an alternative. Between the floor and the threshold it is kept but
explicitly ranked below GeoCLIP.

⚠️ The in-coverage side rests on **n=24** — the only images available inside
gallery coverage that are not gallery entries. They were excluded by blur/indoor/
tile-cap filters, so they are if anything harder than typical, making 0.825
conservative rather than optimistic. Re-calibrate when the gallery grows.

### Validation results

`validate_ensemble.py`. The ensemble routes to one of two already-recorded
per-image predictions, so its error is computable exactly; a parity check ran the
real `EnsembleGeolocator` over samples from each set and matched the offline
reconstruction **10/10 on all three**.

**External, 1000 global images the gallery does not cover:**

| metric | GeoCLIP alone | retrieval alone | ensemble |
|---|---|---|---|
| median err | 1,292.53 km | 10,008.72 km | **1,292.53 km** |
| acc @ 25 km | 0.0070 | 0.0000 | 0.0070 |
| acc @ 200 km | 0.0930 | 0.0010 | 0.0930 |
| acc @ 2500 km | 0.6320 | 0.0250 | 0.6320 |

**0/1000 routed to retrieval.** The ensemble is byte-identical to GeoCLIP alone
here — it does no harm and adds no global improvement. It avoids retrieval's
8,716 km median penalty.

**In-coverage Probe A (honest, n=24):**

| metric | GeoCLIP alone | retrieval alone | ensemble |
|---|---|---|---|
| median err | 160.19 km | 0.02 km | **0.36 km** |
| acc @ 25 km | 0.2917 | 0.9167 | **0.6667** |
| acc @ 200 km | 0.5417 | 0.9583 | 0.7500 |

15/24 (62.5%) routed to retrieval; median error improves **−159.83 km** vs
GeoCLIP. The 9 images left with GeoCLIP are those below 0.825, which is why the
ensemble trails retrieval-alone by 0.34 km here — the price of admitting zero
false accepts globally.

**In-coverage Probe B (near-duplicate upper bound, n=282):** 282/282 routed;
median 322.93 → 0.01 km, identical to retrieval alone.

**Honest summary: the ensemble does no harm globally and adds large value inside
the gallery's few small regions.** It is not a broad global improvement, and with
a 1,719-image North-America-only gallery it cannot be — 0% of a globally diverse
sample clears the trust threshold. Its value scales with gallery coverage.

## Pull resilience

`src/extract/mapillary_client.py` retries HTTP 500s but lets transport failures
propagate. A `ConnectionResetError` (WinError 10054) killed a 500-tile pull
outright after ~17 min stalled on a retry. Every completed tile was already
checkpointed, so nothing was lost, but the run ended early.

`pull_breadth_batch.py` wraps `run_extract` in a resume loop (`--max-attempts`):
because completed tiles are checkpointed, re-entering resumes rather than
repeats, so each attempt makes forward progress. The underlying client should
still handle it.

Also observed: with `subdivision_factor: 2` and `max_subdivision_depth: 3`, one
dense metro tile (Cologne, lat 51.0 lon 7.0) expanded to 0.00375° sub-cells and
consumed a large share of a 500-tile budget returning mostly zero images. For
breadth pulls this is counterproductive — the goal is many places, not exhaustive
coverage of one metro.

⚠️ The `batch01`/`breadth01` runs use **different held-out sets** and their
absolute numbers are not comparable — `breadth01`'s 96 globally-scattered locations are a much harder
test than `batch01`'s 236 Puget-region images, as the baselines show (270.6 km vs
126.5 km pretrained). Compare trajectory shape and relative gain, not raw km.

## External benchmark (validation only, never trained on)

`benchmark_external.py` scores three predictors on an independent image set:
pretrained GeoCLIP, GeoCLIP with a fine-tuned location-encoder checkpoint, and
`RetrievalGeolocator`. Errors are haversine great-circle distance to ground
truth. Input is any CSV with `image_path`, `lat`, `lon`.

This answers a different question from the in-run held-out slices. Those are
carved from our own scraped data and share its collection biases; this checks
whether fine-tuning gains survive contact with a fully independent dataset.
**Do not blend the two — they are not measuring the same thing.**

Source: `benchmark-data/manifest.csv`, 14,002 geotagged images already on disk
from two Kaggle datasets (`google-street-view`, `4k-benchmark-images`) spanning
every longitude band. `sample_external_benchmark.py` thins to a minimum spacing
and samples a diverse slice — the default 1000-image sample has a median
nearest-neighbour distance of ~97 km.

### Result: fine-tuning does not transfer off-distribution

1000 globally-diverse images, never trained on, `breadth01` best checkpoint:

| metric | pretrained | fine-tuned | retrieval |
|---|---|---|---|
| median error | 1292.53 km | 1438.07 km | 10008.72 km |
| mean error | 3750.23 km | 3927.99 km | 10306.64 km |
| acc @ 25 km | 0.0070 | 0.0070 | 0.0000 |
| acc @ 200 km | 0.0930 | 0.0860 | 0.0010 |
| acc @ 2500 km | 0.6320 | 0.6150 | 0.0250 |

Aggregate medians suggest the fine-tuned model is worse, but the **paired
per-image comparison shows the effect is null**:

```
fine-tuned closer : 250 (25.0%)
fine-tuned farther: 255 (25.5%)
identical         : 495 (49.5%)
sign-test z       : -0.22   (|z| > 1.96 would be significant at p<0.05)
```

Half the images get an identical gallery pick and the rest split evenly, so the
median shift comes from a handful of large moves rather than systematic
degradation. **The fine-tuning neither helps nor harms on out-of-distribution
data — the large in-house gains (median 270 → 117 km) are confined to the
geographic and visual distribution it was trained on.**

That is the whole point of an external benchmark: the in-run control slice
looked neutral, but it is still our own scraped data with the same biases, so it
could not have surfaced this.

Caveat: this external set is hard for everything — random global street-level
scenes with few distinctive cues. Pretrained GeoCLIP itself only reaches
`acc @ 25 km = 0.007` here, far below its published Im2GPS3k numbers. The
pretrained-vs-fine-tuned comparison is fair (identical images), but absolute
values reflect a difficult domain.

### Retrieval predictor: expected global failure, confirmed

Median error 10,008 km, `acc @ 2500 km = 0.025`. This is **expected, not a bug**:
`benchmark-gallery/v1` holds 1719 images, 100% North America. A retrieval
predictor cannot locate an image whose region is absent from its gallery, and
essentially every one of these 1000 images is outside it. The number quantifies
a known architectural limitation.

⚠️ **Do not benchmark the retrieval predictor against our own scraped held-out
slices.** `benchmark-gallery/v1` is built *from* that same scraped data, so the
gallery contains the validation images themselves and retrieval returns 0.00 km
for exact hits. A harness check on `breadth01`'s held-out slice produced a
bimodal artifact — 43.75% at 0.00 km error, the rest ~8000 km — which measures
gallery membership, not geolocation. The external set has no such overlap.

### On GSV-Cities

Evaluated and rejected as a source. It is genuinely well-suited on the technical
merits (~530k images, 62k places, ≥100 m apart by construction, lat/lon in both
filenames and per-city CSVs) but is licensed **CC BY-NC-ND 4.0**: NonCommercial
bars commercial use, and NoDerivatives makes distributing a checkpoint trained on
it legally questionable — which is precisely the artifact this pipeline produces.
Mapillary imagery is CC BY-SA, so the breadth path above has no such constraint.

### On discarding raw data

`purge.py` deletes **only files inside the run's own staging directory** and
refuses anything outside it.

This matters: the first batch reuses gallery-pipeline data, which is **committed
source data, not scratch**. The FAISS galleries' `metadata.parquet` stores
`raw_file_path` pointers into those same image files, so deleting them would
silently break the retrieval side of View Finder and destroy data that is not
recoverable without re-scraping. A batch assembled by reference stages no copies,
so purging it is correctly a no-op and says so.

`--purge-after` is opt-in and was **not** used for the first run.

## Running it

Existing gallery data, tile-based split:

```bash
C:/View-Finder/.venv/Scripts/python.exe train_batch.py --run-name batch01_es --epochs 14 --batch-size 128 --lr 3e-5 --eval-every-epochs 1 --early-stop-patience 3
```

Worldwide breadth batch — pull, then train with distance-based split:

```bash
C:/View-Finder/.venv/Scripts/python.exe pull_breadth_batch.py --max-tiles 300 --max-seconds 2400
```

```bash
C:/View-Finder/.venv/Scripts/python.exe train_breadth_batch.py --min-spacing-m 100 --batch-size 64
```

Supporting checks:

| Script | Purpose |
|---|---|
| `probe_vram.py` | peak VRAM sweep across batch sizes |
| `bench_throughput.py` | throughput sweep; finds the WDDM spill cliff |
| `check_gallery_floor.py` | oracle error floor from gallery resolution |
| `check_coord_collisions.py` | near-duplicate coordinate density |
| `assess_pool_diversity.py` | distinct locations retained at each spacing |
| `check_location_sampling.py` | validates thinning/split against known data |
| `verify_eval_parity.py` | our eval vs `GeoCLIP.predict()` |

## On `gpuboost`

`gpuboost doctor` / `benchmark --quick --recommend` / `analyze` / `agent optimize
--trial` were run against this setup. Assessment against direct measurement:

- **AMP** — confirmed, and adopted. gpuboost estimated 1.12× from a synthetic
  workload; measured gain on the real ViT-L/14 step is **3.0×** (35 → 107 img/s).
  Its suggested `GradScaler` is not applicable: our fp16 region sits inside
  `torch.no_grad()` (frozen tower), so no gradients flow through it.
- **`cudnn.benchmark = True`** — applied. Shapes are fixed; marginal for a ViT.
- **"Use batch size 4"** — *not* followed. gpuboost's own warning flags the sweep
  as CPU/Python-overhead limited; direct measurement shows throughput flat across
  batch 64–256 and batch 4 would be far slower for this model.
- **"num_workers=0 fastest"** — *not* followed, for the same reason. That was a
  synthetic in-memory benchmark; the real loop is JPEG-decode bound, and
  `num_workers=4` reaches 111 img/s against a 107 img/s pure-GPU ceiling.
- **`.item()` in the loop** — kept deliberately; per-step loss logging is
  required, and a sub-millisecond sync against a 1.15 s step is negligible.
- Its static analyzer reported "no AMP autocast" twice — a false positive; the
  loop uses `torch.autocast("cuda", dtype=torch.float16)`.
