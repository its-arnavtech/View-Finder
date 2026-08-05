# View Finder

View Finder is a local Python CLI that estimates where a travel or landscape
photo or video was captured, by comparing scenery cues such as architecture,
signage, vegetation, and terrain. It does not identify people.

The default hybrid is fully local: PP-OCRv6 reads visible text in its supported
50-language set, an offline
GeoNames/OSM gazetteer resolves place names, and a local Wikimedia Commons
OpenCLIP/FAISS gallery searches known landmarks. Images never leave the
computer. Model weights and public reference data must be downloaded once
during setup.

## Distinct predictors and trusted abstention

View Finder ships several geolocation systems. They are genuinely separate
approaches, not variants of one another, and `--predictor` selects between
them.

### `hybrid` (default for images)

This is the recommended, fail-closed path. It checks evidence in this order:

1. embedded EXIF GPS;
2. a near-exact perceptual match in the attributed local Commons gallery;
3. agreement between local PaddleOCR/GeoNames text evidence and an independent
   local OpenCLIP landmark-retrieval group;
4. PLONK OSV-5M as a visual prior only.

Valid EXIF can emit directly. A near-exact local reference match can emit
directly. OCR and visual landmark retrieval may emit only when they independently
agree within 25 km. A strong multi-reference landmark group or exact text match
alone is marked `REVIEW`.
With visual evidence alone the top-level `prediction` is `null`; PLONK
candidates are retained under `visual_hypotheses` for manual inspection.

This improves trustworthiness by refusing unsupported answers. It does **not**
make arbitrary photo geolocation perfectly reliable: images without metadata,
recognizable landmarks, or readable location text may be impossible to locate
within a small radius. `--require-evidence` makes this contract machine
enforceable by returning exit code `3` for both `REVIEW` and `ABSTAIN`.
The local-evidence thresholds are deliberately conservative and still require
held-out calibration before the CLI can report a probability-of-correctness.
`ABSTAIN` means the available independent signals did not justify an answer.

### `geoclip`

The query image is embedded, and that embedding is compared against **learned
location embeddings** generated from raw GPS coordinates. No photograph of the
place is involved at any stage: the model has internalised what locations look
like during training. Its consequence is that GeoCLIP can only ever answer with
a coordinate present in its GPS gallery — the bundled gallery holds ~100k points
spread worldwide, so a remote area may have no nearby entry and the prediction
gets snapped to the closest one that exists.

### `retrieval`

The query image is embedded with OpenCLIP and matched by nearest-neighbour
search against the embeddings of **other real photographs** whose coordinates we
know — reference photos scraped from Mapillary and indexed by this project's own
`gallery-pipeline` ETL. A prediction means "this looks like that specific photo,
which was taken here". This path is end-to-end ours: our scraped photos, our
embeddings, our FAISS index.

**The retrieval predictor is currently experimental.** A retrieval system can
only ever propose a location it holds a reference photo for. The current
galleries contain only a few hundred images covering a handful of regions, so a
query from anywhere else *cannot* be answered correctly — the nearest neighbour
will be whichever gallery photo looks most similar, and its coordinates will be
confidently wrong. This is structural, and the confidence score cannot detect
it: the score measures agreement among the returned neighbours, and neighbours
drawn from a tiny gallery can agree strongly while all being wrong. The CLI
prints a warning whenever the selected gallery holds fewer than 1000 photos.

The two systems are never merged or averaged. With `--predictor both` they run
independently and both results are reported side by side; where they disagree,
that disagreement is the useful signal.

### `plonk-osv`, `plonk-yfcc`, and `trusted`

The optional PLONK predictors sample a probability distribution over the
Earth instead of collapsing uncertainty into a single coordinate.
`plonk-osv` uses the official model trained on 5.1 million OSV-5M street-view
images from 225 countries and territories. `plonk-yfcc` uses the official
general-photo YFCC model. Their sampled modes are geographically clustered and
the fraction of posterior samples in each mode is retained as evidence.

`--predictor trusted` runs GeoCLIP and both PLONK models separately. It emits
a top-level `prediction` only when:

1. at least two predictors form an all-pairs geographic consensus; and
2. that exact rule has a held-out calibration whose 95% Wilson lower bound
   reaches `--required-success-rate`.

Without calibration, agreement is visible as `provisional`, but the program
abstains. This is intentional: **no worldwide image-geolocation model can be
trusted fully on every input**. Blur, copied images, adversarial edits,
out-of-distribution scenes, and visually similar places make a universal
guarantee impossible. The safe behavior is measured selective accuracy plus
abstention, not an uncalibrated confidence label.

On this repository's fixed 1,000-image benchmark at 32 PLONK samples, OSV-5M
was the strongest tested predictor: 14.4% within 200 km, 42.3% within 750 km,
and 996 km median error, versus GeoCLIP's 9.4%, 34.9%, and 1,374 km. It was
closer than GeoCLIP on 59.5% of images. This is a meaningful improvement, not a
reliability guarantee. The 200 km trust rule achieved only 32.7% success among
accepted images, so no runtime calibration was installed and `trusted` remains
fail-closed. See [the benchmark report](benchmark-harness/REPORT.md).

## Requirements

- Python 3.10 or newer
- Enough memory to run GeoCLIP's CLIP ViT-L/14 backbone
- A CUDA GPU is used automatically when available; CPU inference also works
- A local video codec supported by OpenCV for video inputs
- FFmpeg and `yt-dlp` for public Instagram/TikTok URL inputs

## Quickstart from a fresh clone

These are the exact, verified steps to go from a clean `git clone` to a working
prediction. No Mapillary token is needed — the retrieval galleries are committed
(index + metadata only), and model weights download once on first run.

```bash
git clone https://github.com/its-arnavtech/View-Finder.git
cd View-Finder
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r viewfinder/requirements.txt
python -m pip install -r viewfinder/requirements-local-evidence.txt
python -m pip install -r viewfinder/requirements-social-video.txt
python -m pip install --no-deps --no-build-isolation -e .
```

The default image command is fail-closed:

```bash
view-finder locate images/IMG_6790.png --require-evidence
```

It can emit immediately when the original file has GPS metadata. Otherwise it
returns `prediction: null` unless online evidence is configured as described
below. PLONK is preferred for manual visual hypotheses when its optional runtime
is installed; the cached GeoCLIP model is used if PLONK is unavailable.

Run the retrieval predictor against a committed gallery (works fully offline
after the one-time OpenCLIP weight download):

```bash
python -m viewfinder.main --input images/IMG_6790.png --predictor retrieval \
  --gallery-path gallery-pipeline/data/target-regions/gallery/v2
```

Or run both predictors side by side (GeoCLIP downloads its CLIP ViT-L/14 backbone
once, ~1.7 GB):

```bash
python -m viewfinder.main --input images/IMG_6790.png --predictor both \
  --gallery-path gallery-pipeline/data/target-regions/gallery/v2
```

Expected on a correct setup: retrieval reports **Silverdale / Belfair, Washington**
for the bundled test images against the `v2` gallery, and GeoCLIP reports a
California result — the same numbers this repo documents. The dependencies in
`viewfinder/requirements.txt` are pinned to the exact versions these results were
produced with, so a fresh clone reproduces them.

## Installation

The Quickstart above is the short path. In full, from the project root:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r viewfinder/requirements.txt
```

Cache the GeoCLIP backbone once while network access is available:

```powershell
python -c "from viewfinder.geolocator import GeoLocator; GeoLocator()"
```

After that setup step, runs do not make network calls unless
`--online-evidence` is explicitly supplied. If a visual model is not cached,
the CLI exits with an actionable error instead of downloading it silently.

`viewfinder/requirements.txt` already includes the retrieval predictor's
dependencies (`faiss-cpu`, `open-clip-torch`, `torchvision`, `pyarrow`). To use
`--predictor retrieval`, also cache its OpenCLIP checkpoint once and point it at
a gallery (see [Galleries](#galleries)):

```powershell
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"
```

Install the optional PLONK adapter (tested in this repo's Python 3.13
environment):

```powershell
python -m pip install --no-deps diff-plonk==0.4
python -m pip install -r viewfinder/requirements-plonk-runtime.txt
```

`diff-plonk 0.4` declares `scipy==1.13.1` and an unused `geos` package in its
upstream metadata. SciPy 1.13.1 has no Python 3.13 wheel, so the command above
intentionally bypasses that metadata and uses the newer SciPy already installed
by the core environment. Both official PLONK paths were smoke-tested here, but
`pip check` will still report those two upstream metadata discrepancies. For a
dependency-clean PLONK environment, use Python 3.10–3.12 and install
`diff-plonk==0.4` normally.

The first PLONK run downloads its official checkpoint and feature backbone.
Those assets are cached locally and can be reused offline. The OSV and YFCC
models have different backbones, so enabling both requires additional disk and
memory.

Install the fully local evidence stack:

```powershell
python -m pip install -r viewfinder/requirements-local-evidence.txt
python local-data-pipeline/build_local_data.py all
```

The build is resumable, keeps all downloaded media under the ignored
`local-data` directory, and defaults to a 40 GiB Commons thumbnail budget.
GeoNames is about 600 MiB compressed; the expanded search database and landmark
gallery require substantially more room. Check progress at any time:

```powershell
python local-data-pipeline/build_local_data.py status
python -m viewfinder.main --input .\photos\street.jpg --predictor hybrid `
  --local-data .\local-data --require-evidence
```

The previous Google-backed experiment remains available as
`--predictor cloud-hybrid --online-evidence`, but it is not the default and is
not needed by the local system.

Run all three global models through the conservative gate:

```powershell
python -m viewfinder.main --input .\photos\street.jpg --predictor trusted `
  --plonk-samples 32 --require-trust
```

Before a calibration artifact exists, this command writes all raw predictions,
sets the top-level `prediction` to `null`, and exits `3`. Generate a calibration
only from an untouched evaluation split:

```powershell
python benchmark-harness\run_benchmark.py --predictors all --sample-size 1000 `
  --out benchmark-harness\heldout-results.csv
python benchmark-harness\calibrate_trust.py `
  --results benchmark-harness\heldout-results.csv
```

Do not use the same images to choose thresholds and to calibrate them. Any
model, preprocessing, dataset, or threshold change invalidates the artifact.
See [DATASETS.md](DATASETS.md) for dataset provenance and licensing notes.

### Galleries

The retrieval predictor searches a FAISS gallery built by `gallery-pipeline`.
Prebuilt galleries are checked into `gallery-pipeline/data/*/gallery/vN/` (the
FAISS index + metadata + manifest — not the raw source images, which are not
required at query time), so retrieval reproduces the documented results out of
the box:

```powershell
python -m viewfinder.main --input .\images\IMG_6790.png --predictor both `
  --gallery-path .\gallery-pipeline\data\target-regions\gallery\v2
```

To build your own gallery (denser or elsewhere), run the pipeline — see
`gallery-pipeline/README.md`. That step needs a Mapillary token; querying an
existing gallery does not.

## Usage

Analyze an image:

```powershell
view-finder locate .\photos\coast.jpg
```

Analyze a local video at two sampled frames per second:

```powershell
view-finder locate .\clips\walk.mov --fps 2 --max-frames 24
```

Download and analyze a public Instagram reel or TikTok clip:

```powershell
view-finder locate "https://www.instagram.com/reel/POST_ID/" --require-evidence
view-finder locate "https://www.tiktok.com/@USER/video/VIDEO_ID" --require-evidence
```

The downloaded clip is cached under `local-data/social-video-cache`, its source
URL and public post metadata are recorded in the result, and no account,
cookies, browser session, or API key is used. Private/login-gated posts are not
supported. Platform extractors can temporarily break or an IP may be blocked;
these cases return an actionable download error. Only download media you are
authorized to access and process.

Inspect the installation and local datasets:

```powershell
view-finder doctor
view-finder formats
view-finder data status
```

Choose an output path and enable stage-level debug logging:

```powershell
view-finder locate .\photos\street.png --output .\results\street.json -v
```

Produce only the final prediction on stdout (as one JSON line):

```powershell
view-finder locate .\clips\panorama.mp4 --predictor geoclip --quiet
```

Use the retrieval predictor against a gallery built by `gallery-pipeline`:

```powershell
python -m viewfinder.main --input .\photos\redwoods.jpg --predictor retrieval --gallery-path .\gallery-pipeline\data\target-regions\gallery\v1
```

Run both systems and compare them side by side:

```powershell
python -m viewfinder.main --input .\photos\redwoods.jpg --predictor both --gallery-path .\gallery-pipeline\data\target-regions\gallery\v1
```

CLI options:

```text
SOURCE             Local image/video path or public Instagram/TikTok URL
--predictor NAME   hybrid (default) | cloud-hybrid | geoclip | retrieval |
                   both | plonk-osv | plonk-yfcc | trusted
--local-data PATH  Local evidence root; default ./local-data
--online-evidence  cloud-hybrid only: use Cloud Vision + Google Places
--require-evidence Hybrid modes: exit 3 unless evidence authorizes automation
--gallery-path PATH  Gallery version directory; required for retrieval/both
--plonk-samples INT  Posterior samples per PLONK model; default 32
--trust-radius-km N  Maximum pairwise consensus spread; default 200
--required-success-rate N  Required held-out Wilson lower bound; default .95
--trust-success-km N  Distance defining a calibrated success; default 200
--trust-calibration PATH  Optional calibration artifact path
--require-trust     Exit 3 when trusted mode abstains
--fps FLOAT        Video sampling rate; default 1.0
--max-frames INT   Maximum retained video frames; default 24
--download-dir PATH  Social-video cache; default local-data/social-video-cache
--max-video-seconds INT  Download/video duration cap; default 600
--max-download-mb INT  Social download size cap; default 500
--top-k INT        Ranked candidates retained/reported; default 5
                   (also the k for the retrieval nearest-neighbour search)
--max-predictions INT  Distinct location clusters returned; default 5
--output PATH      JSON path; defaults beside the input file
--gallery PATH     Optional custom GPS gallery for GeoCLIP (see below)
--gallery-mode M   augment (default) | replace, for --gallery
--verbose, -v      Debug logging for pipeline stages
--quiet, -q        No progress logging; print only the final prediction
```

`--gallery-path` must point at a gallery **version** directory produced by
`gallery-pipeline`'s load stage, containing `gallery.faiss`,
`metadata.parquet`, and `manifest.json`. Omitting it with
`--predictor retrieval` or `--predictor both` is an error (exit `2`).

The query encoder is read from the gallery's own `manifest.json` rather than
hardcoded, so a gallery is always searched with the model that built it. A
mismatch between query and gallery embedding spaces would make retrieval
meaningless, so a gallery built with a non-OpenCLIP encoder is rejected
outright.

Note `--gallery`/`--gallery-mode` are a separate, GeoCLIP-only feature: they
extend GeoCLIP's *coordinate* pool with coordinates harvested by the pipeline.
That is unrelated to `--predictor retrieval`, which does not use GeoCLIP at all.

Supported formats depend on the installed Pillow/OpenCV decoders. Run
`view-finder formats` for the exact image and video
extensions available in the current environment. Format detection is
extension-based and case-insensitive.

## Pipeline

For an image, View Finder loads and orients the image, obtains ranked
coordinates from the selected predictor, geographically clusters the top-k
results, independently scores each distinct location hypothesis, and
reverse-geocodes every returned cluster.

Both predictors share the clustering, confidence scoring, and reverse-geocoding
stages, so they produce the same shape of ranked result. Scores are normalised
per frame against that frame's best candidate, which is what lets GeoCLIP's
softmax probabilities and retrieval's cosine similarities both flow through the
same scoring code unchanged.

For a video, it samples at the requested rate, caps retained frames, rejects
blurry and near-duplicate frames, and runs the selected predictor. The local
hybrid evaluates OCR/GeoNames and Commons evidence per frame, reuses its loaded
models, and emits only when accepted frame coordinates agree; otherwise it
returns `REVIEW` or `ABSTAIN`. Visual predictors apply connected-component
clustering with haversine distance to the complete frame x top-k candidate pool. Every cluster
gets its own score-weighted spherical centroid, confidence breakdown, and count
of supporting frames. Clusters remain separate, are ranked by confidence, and
are reverse-geocoded in one offline batch. Progress is reported after inference
batches unless quiet mode is enabled.

## Output

Unless `--output` is supplied, `photo.jpg` produces
`photo_viewfinder.json` in the same directory. A video result resembles:

```json
{
  "input": "C:\\photos\\trip.mp4",
  "input_type": "video",
  "prediction": {
    "latitude": 41.9028,
    "longitude": 12.4964,
    "country": "IT",
    "admin1": "Latium",
    "city": "Rome",
    "confidence_percent": 78.4
  },
  "predictions": [
    {
      "rank": 1,
      "lat": 41.9028,
      "lon": 12.4964,
      "confidence": 78.4,
      "support_count": 6,
      "candidate_count": 12,
      "country": "IT",
      "admin1": "Latium",
      "city": "Rome",
      "confidence_details": {}
    },
    {
      "rank": 2,
      "lat": 43.7696,
      "lon": 11.2558,
      "confidence": 45.2,
      "support_count": 2,
      "candidate_count": 3,
      "country": "IT",
      "admin1": "Tuscany",
      "city": "Florence",
      "confidence_details": {}
    }
  ],
  "alternatives": [
    {
      "rank": 2,
      "lat": 43.7696,
      "lon": 11.2558,
      "confidence": 45.2,
      "support_count": 2,
      "candidate_count": 3,
      "country": "IT",
      "admin1": "Tuscany",
      "city": "Florence",
      "confidence_details": {}
    }
  ],
  "processing": {
    "video_fps": 30.0,
    "duration_seconds": 12.0,
    "total_frames": 360,
    "frames_sampled": 12,
    "frames_kept": 8,
    "frames_skipped_blurry": 2,
    "frames_skipped_duplicate": 2,
    "blur_fallback_used": false,
    "frames_analyzed": 8,
    "frame_dispersion_km": 14.2,
    "confidence_details": {}
  }
}
```

With `--predictor both`, the top level instead holds a `predictors` list and a
`results` object keyed by predictor name, each containing the same structure
shown above. The blocks are kept strictly separate — nothing is merged or
averaged across the two systems:

```json
{
  "input": "C:\\photos\\redwoods.jpg",
  "input_type": "image",
  "predictors": ["geoclip", "retrieval"],
  "results": {
    "geoclip": { "predictor": "geoclip", "prediction": {}, "predictions": [] },
    "retrieval": {
      "predictor": "retrieval",
      "prediction": {},
      "predictions": [],
      "warning": "EXPERIMENTAL: the retrieval gallery holds only 299 reference photos ...",
      "gallery": {
        "version": 1,
        "image_count": 299,
        "embedding_model": "open_clip:ViT-B-32:laion2b_s34b_b79k",
        "experimental": true
      },
      "processing": { "gallery_images_matched": 5, "gallery_size": 299 }
    }
  }
}
```

Retrieval candidates additionally carry `source_image_id`, the gallery photo
each match came from, so a prediction can be traced back to the reference
photograph that produced it.

Retrieval runs also report `top_similarity`, the raw cosine similarity of the
single best match. **Read it alongside the confidence score, not instead of
it** — the two answer different questions. Confidence measures how well the
returned neighbours agree with each other; `top_similarity` measures how much
the query actually resembles the closest reference photo. A high confidence
paired with a low `top_similarity` is the signature of a query whose true
location is absent from the gallery: the neighbours agree because they all come
from the same small region, not because they are right.

Single-predictor runs (including the default) keep the flat shape shown
earlier, so existing consumers of the JSON are unaffected.

`prediction` remains the highest-confidence result for compatibility with
single-answer consumers. `predictions` contains every ranked cluster up to
`--max-predictions`, and `alternatives` contains the same objects from rank 2
onward. `support_count` is the number of independent source frames backing the
cluster; `candidate_count` is the number of retained top-k entries it contains.
For a still image, support is necessarily one frame. View Finder confidence is
a relative ranking/reliability indicator, not a calibrated probability or a
guarantee that the true location lies within a particular distance.

## Errors and exit codes

- Exit `0`: result written successfully
- Exit `1`: a pipeline stage failed (decoding, model loading/inference,
  aggregation, reverse geocoding, or output writing)
- Exit `2`: invalid CLI input, such as a missing file or unsupported extension
- Exit `3`: `--require-evidence` or `--require-trust` requested a fail-closed
  result and the evidence gate abstained

Failures are reported as concise messages without raw tracebacks. Common fixes
are included in the message where possible.
