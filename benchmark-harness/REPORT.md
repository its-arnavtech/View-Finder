# View Finder benchmark report

## Phase 2: PLONK and trust-gate evaluation

This phase reran the exact same seed-42 sample at the production setting of
**32 posterior samples per image**. Both PLONK predictors processed all 1,000
images with zero failures on an RTX 4060 Laptop GPU.

| Predictor | ≤25 km | ≤200 km | ≤750 km | ≤2500 km | Mean km | Median km |
|---|---:|---:|---:|---:|---:|---:|
| GeoCLIP | 1.2% | 9.4% | 34.9% | 64.9% | 3,560 | 1,374 |
| PLONK OSV-5M | 0.6% | **14.4%** | **42.3%** | **69.7%** | **2,954** | **996** |
| PLONK YFCC | 0.6% | 9.0% | 27.8% | 51.6% | 4,773 | 2,263 |

PLONK OSV-5M is the strongest tested predictor. It was closer than GeoCLIP on
**595/1,000 images (59.5%)**, improved regional accuracy by 5 percentage points,
and reduced median error by 378 km. PLONK YFCC was worse than both on this
mostly street-view benchmark, which is consistent with its general-photo
training distribution being less well matched to these inputs.

### Trust-gate result

The predeclared rule required at least two of GeoCLIP, PLONK OSV-5M, and PLONK
YFCC to agree within 200 km; success meant the consensus was within 200 km of
ground truth.

| Evaluated | Accepted | Coverage | Correct | Observed success | 95% Wilson lower |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 223 | 22.3% | 73 | 32.7% | 26.9% |

This fails the required 95% lower bound by a very large margin. Consequently,
**no default `models/trust_calibration.json` was installed** and trusted mode
continues to abstain. A grid of stricter fixed-pair and three-model agreement
rules also found no defensible 95%-precision subset. The best lower bound in
that exploratory grid was 44.4% (PLONK OSV + YFCC within 50 km, only 16
accepted images).

The oracle upper bound is also sobering: at least one of the three global models
was within 200 km for only **249/1,000 images (24.9%)**. Model selection alone
cannot turn this particular benchmark into a near-certain locator.

### Phase 2 conclusion

- Use `plonk-osv` when optional PLONK dependencies and model storage are
  available; it is measurably better overall than the GeoCLIP baseline here.
- Keep GeoCLIP as the lightweight core fallback.
- Do not use PLONK YFCC as the primary model for random street-view imagery.
- Do not represent any current confidence score or model agreement as a
  probability of correctness.
- Keep `trusted` fail-closed until a new, independently validated signal
  achieves the configured reliability target.

Phase 2 artifacts:

- `heldout-results-1000-s32.csv` — exact-path merged per-image results
- `heldout-metrics-1000-s32.md` — accuracy tables
- `ensemble-analysis-1000-s32.md` — head-to-head and agreement grid
- `trust-calibration-1000-s32.json` — rejected calibration statistics retained
  as an audit artifact, not installed for runtime use

## Phase 1: GeoCLIP and retrieval

First quantitative accuracy measurement for both View Finder predictors. Prior to
this, the project had no accuracy numbers at all — only anecdotal test images.

## Setup

| Item | Value |
|---|---|
| Test images | **1,000**, stratified random sample of the 14,002-image manifest |
| Sources | `kaggle/google-street-view` (714), `kaggle/4k-benchmark-images` (286) |
| Sampling | Proportional to each source's share, seed 42 (largest-remainder allocation) |
| Predictors | GeoCLIP (bundled 100k GPS gallery) and Retrieval (merged benchmark gallery) |
| Retrieval gallery | **1,719 images** — live-smoke v1 (140) + target-regions v2 (1,305) + global-test v1 (274) |
| Failures | **0 / 1,000** |
| Throughput | ~1.1 s/image for both predictors combined, after one-time model load (~5.6 min) |

Why 1,000 and not all 14,002: measured cost was ~2.15 s/image, so the full set
would take ~8.4 h. At n=1,000 the 95% confidence interval on a percentage metric
is about ±3 pp, versus ±0.8 pp at n=14,002 — 14× the runtime for ~2 pp of
precision, on effects that turn out to differ by tens of points. `--sample-size 0`
runs the full manifest if ever needed.

## GeoCLIP accuracy

| Subset | n | ≤1km | ≤25km | ≤200km | ≤750km | ≤2500km | Mean km | Median km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 1000 | 0.0% | 1.2% | 9.4% | 34.9% | 64.9% | 3,560 | 1,374 |
| kaggle/4k-benchmark-images | 286 | 0.0% | 3.1% | 17.8% | 51.0% | 72.0% | 2,845 | 707 |
| kaggle/google-street-view | 714 | 0.0% | 0.4% | 6.0% | 28.4% | 62.0% | 3,847 | 1,687 |

The 4k-benchmark subset scores roughly **2–3× better at every threshold**. It is
continent-balanced and curated; the street-view subset is uniformly random over
the globe, so it contains far more anonymous roadside scenes with no distinctive
cues.

## Retrieval accuracy

| Subset | n | ≤1km | ≤25km | ≤200km | ≤750km | ≤2500km | Mean km | Median km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 1000 | 0.0% | 0.0% | 0.0% | 0.1% | 3.0% | 10,306 | 10,197 |
| kaggle/4k-benchmark-images | 286 | 0.0% | 0.0% | 0.0% | 0.3% | 4.9% | 10,276 | 10,590 |
| kaggle/google-street-view | 714 | 0.0% | 0.0% | 0.0% | 0.0% | 2.2% | 10,319 | 10,017 |

**A median error of ~10,200 km is essentially "random point on Earth."** This is
the predicted and unavoidable consequence of a 1,719-image gallery covering only
Chicago, NorCal, Humboldt and Puget Sound being asked about globally sampled
photos: it can only ever answer with a coordinate it holds a reference photo for,
so nearly every test image is thousands of km from anything in the index. **This
is a structural property of retrieval at this gallery scale, not a bug.**

## Confidence calibration

### GeoCLIP

| Confidence bucket | n | Mean error km | Median error km |
|---|---:|---:|---:|
| 0–25% | 0 | – | – |
| 25–50% | 31 | 1,473 | 323 |
| 50–75% | 857 | 3,683 | 1,495 |
| 75–100% | 112 | 3,200 | 1,191 |

**Confidence is not usefully calibrated, and is mildly inverted.** The
*lowest*-confidence bucket (25–50%) has by far the *lowest* error — median 323 km
versus 1,191 km for the highest-confidence bucket. A user filtering for
"high-confidence" results would be selecting worse predictions, not better ones.
(Caveat: n=31 in that bucket, so the effect is suggestive rather than firmly
established; what *is* firm is the absence of the expected downward trend.)

### Retrieval

| Confidence bucket | n | Mean error km | Median error km |
|---|---:|---:|---:|
| 0–25% | 0 | – | – |
| 25–50% | 0 | – | – |
| 50–75% | 0 | – | – |
| 75–100% | **1000** | 10,306 | 10,197 |

**Every single one of the 1,000 predictions landed in the 75–100% bucket while
being ~10,000 km wrong.** Retrieval confidence carries no information whatsoever
here. This confirms, at scale, the failure mode observed anecdotally earlier
(an 86%-confidence prediction that was completely wrong).

### Does `top_similarity` help?

| top_similarity | n | Mean error km | Median error km |
|---|---:|---:|---:|
| 0.0–0.5 | 2 | 11,593 | 11,593 |
| 0.5–0.6 | 106 | 10,401 | 10,711 |
| 0.6–0.7 | 641 | 10,252 | 10,075 |
| 0.7–0.8 | 251 | 10,395 | 10,331 |

No. Error is **flat at ~10,300 km across every similarity band**. Raw match
quality is no more diagnostic than confidence when the gallery lacks coverage —
a visually similar photo from the wrong continent is still the wrong continent.

## Head-to-head

- Images scored by both predictors: **1,000**
- **GeoCLIP closer: 901 (90.1%)**
- Retrieval closer: 99 (9.9%)

## Comparison to GeoCLIP's published Im2GPS3k results

**This is directional context, not an apples-to-apples claim.** Our test set is
*not* Im2GPS3k; different data, different difficulty.

| Threshold | GeoCLIP paper (Im2GPS3k) | This benchmark (GeoCLIP) |
|---|---:|---:|
| ≤1 km (street) | ~14% | 0.0% |
| ≤25 km (city) | ~34% | 1.2% |
| ≤200 km (region) | ~50% | 9.4% |
| ≤750 km (country) | ~70% | 34.9% |
| ≤2500 km (continent) | ~84% | 64.9% |

Our numbers are substantially lower, and the test-set difference is the most
likely explanation rather than a broken implementation:

- **Im2GPS3k is Flickr tourist photography** — heavily weighted toward landmarks,
  monuments and famous places that are near-uniquely identifiable.
- **Our set is randomly sampled street-view imagery** — a random point on a road
  in rural Brazil or Poland, typically with no landmark, no signage and no
  distinctive architecture. This is close to the hardest realistic case for image
  geolocation.

The one internal check consistent with this: the curated, continent-balanced
4k-benchmark subset scores 2–3× better than the uniformly random street-view
subset on the same model. **The right reading is "GeoCLIP scores 64.9% at
continent level on a deliberately hard random-street-view set", not "GeoCLIP is
worse than published."** Running the actual Im2GPS3k set would be needed for a
true comparison.

## What this means for the project

The measurable result is that **GeoCLIP is the only predictor currently worth
using**: it wins head-to-head on 90.1% of images and gets two thirds of them onto
the right continent, while the retrieval predictor — the project's own
end-to-end contribution — is effectively guessing at a median error of ~10,200 km
and never once beats a coin flip on coverage. That is not a defect in the
retrieval code, which self-retrieves correctly and was verified at 1.0000 cosine
on in-gallery images; it is a direct consequence of a 1,719-image gallery against
a global test set, and no amount of tuning changes it — only orders of magnitude
more gallery coverage would. The honest conclusion is that **the ensemble/
retrieval advantage this project set out to demonstrate is not supported by the
data at current gallery scale**, and the confidence scores of *both* predictors
should not be presented to users as reliability indicators, because neither
correlates with actual error (GeoCLIP's is mildly inverted; retrieval's is
constant at 75–100% while being globally wrong). The value delivered so far is
the engineering system — pipeline, harness, reproducibility — and this benchmark
is the first evidence-based statement of where it actually stands.

## Reproducing

```bash
python benchmark-harness/merge_galleries.py
python benchmark-harness/run_benchmark.py --sample-size 1000
python benchmark-harness/analyze_results.py
```

Artifacts: `results.csv` (per-image), `metrics.md` (tables), this report.
