# View Finder

View Finder is a local Python CLI that estimates where a travel or landscape
photo or video was captured, by comparing scenery cues such as architecture,
signage, vegetation, and terrain. It does not identify people.

Inference is offline. The CLI forces Hugging Face/Transformers offline mode,
and place names come from `reverse_geocoder`'s bundled dataset. Model weights
must therefore be downloaded and cached once during setup.

## Two independent predictors

View Finder ships two geolocation systems. They are genuinely separate
approaches, not variants of one another, and `--predictor` selects between
them.

### `geoclip` (default)

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

## Requirements

- Python 3.10 or newer
- Enough memory to run GeoCLIP's CLIP ViT-L/14 backbone
- A CUDA GPU is used automatically when available; CPU inference also works
- A local video codec supported by OpenCV for video inputs

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
```

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

After that setup step, normal CLI runs do not make network calls. If the model
is not cached, the CLI exits with an actionable error instead of downloading it
silently.

`viewfinder/requirements.txt` already includes the retrieval predictor's
dependencies (`faiss-cpu`, `open-clip-torch`, `torchvision`, `pyarrow`). To use
`--predictor retrieval`, also cache its OpenCLIP checkpoint once and point it at
a gallery (see [Galleries](#galleries)):

```powershell
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"
```

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
python -m viewfinder.main --input .\photos\coast.jpg
```

Analyze a video at two sampled frames per second and retain ten candidates:

```powershell
python -m viewfinder.main --input .\clips\walk.mov --fps 2 --top-k 10
```

Choose an output path and enable stage-level debug logging:

```powershell
python -m viewfinder.main --input .\photos\street.png --output .\results\street.json -v
```

Produce only the final prediction on stdout (as one JSON line):

```powershell
python -m viewfinder.main --input .\clips\panorama.mp4 --quiet
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
--input PATH       Required image or video path
--predictor NAME   geoclip (default) | retrieval | both
--gallery-path PATH  Gallery version directory; required for retrieval/both
--fps FLOAT        Video sampling rate; default 1.0
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

Supported images are `.jpg`, `.jpeg`, and `.png`. Supported videos are `.mp4`,
`.mov`, `.avi`, and `.mkv`. Format detection is extension-based and
case-insensitive.

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

For a video, it samples at the requested rate, rejects blurry and near-duplicate
frames, runs GeoCLIP in batches, and applies connected-component clustering with
haversine distance to the complete frame x top-k candidate pool. Every cluster
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

Failures are reported as concise messages without raw tracebacks. Common fixes
are included in the message where possible.
