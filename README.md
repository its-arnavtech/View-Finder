# View Finder

View Finder is a local Python CLI that estimates where a travel or landscape
photo or video was captured, by comparing scenery cues such as architecture,
signage, vegetation, and terrain. It does not identify people.

Everything runs locally. Images never leave the computer; model weights and
public reference data must be downloaded once during setup.

The default predictor is `geoclip`, a visual model that will estimate a
location for any photograph. The `hybrid` predictor remains available and is
the fail-closed, evidence-first path:
PP-OCRv6 reads visible text in its supported 50-language set, an offline
GeoNames/OSM gazetteer resolves place names, and a local Wikimedia Commons
OpenCLIP/FAISS gallery searches known landmarks. Choose it with
`--predictor hybrid` whenever an unsupported guess is worse than no answer.

There is also a small desktop GUI — see [Desktop GUI](#desktop-gui).

## Distinct predictors and trusted abstention

View Finder ships several geolocation systems. They are genuinely separate
approaches, not variants of one another, and `--predictor` selects between
them.

### `hybrid` (fail-closed evidence path)

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

View Finder loads the fine-tuned OSV-5M location encoder at
`finetune/checkpoints/osv5m_reg_narrow_30/location_encoder_best_external.pth`
by default. The CLIP vision tower and the rest of GeoCLIP remain stock. Use
`--use-stock-geoclip` for an explicit stock-encoder comparison. The default
fails clearly if the fine-tuned checkpoint is missing instead of silently
changing models. Checkpoints are local training artifacts and remain gitignored,
so a fresh clone must copy this winning checkpoint into that exact path (or set
`VIEWFINDER_LOCATION_ENCODER` to its location) before using the fine-tuned
default.

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

## Retrieval and ensemble: explored and shelved

Two predictors were built, calibrated and tested, then removed from the active
code path. They are **not deleted** — they live in `experiments/retrieval/`
with their tests, and the reasoning is recorded here because the negative
result is the useful part.

### What was built

`retrieval_geolocator.py` embedded a query with OpenCLIP and matched it by
nearest neighbour against a FAISS index of real photographs whose coordinates
were known — reference imagery scraped from Mapillary by this project's own
`gallery-pipeline` ETL. A prediction meant "this looks like that specific
photo, which was taken here". End-to-end ours: our photos, our embeddings, our
index.

`ensemble_geolocator.py` composed it with GeoCLIP. Neither model was modified;
the ensemble only decided which answer led, based on retrieval's top-1 cosine
similarity.

### How the 0.825 threshold was calibrated

The obvious gate — retrieval's own confidence score — is unusable. That score
measures agreement among the returned neighbours, so a small gallery can
produce strong agreement while every neighbour is wrong. An Alaska query
scored 86.1% confidence at 0.523 similarity and named the wrong region.

Raw top-1 cosine similarity does work, but only at a surprisingly high cut.
Measured on 1000 globally diverse images the gallery does not cover (negative
class) against images inside coverage that were not gallery entries (positive
class):

| population | min | median | p99 | max |
|---|---|---|---|---|
| not covered (n=1000) | 0.487 | 0.672 | 0.777 | **0.803** |
| in coverage (n=24) | 0.632 | **0.851** | — | 0.962 |

The populations separate only at the very top of the range. `0.825` sits above
the observed out-of-coverage ceiling and below the in-coverage median. At that
cut the external set contributed zero false accepts while 62.5% of in-coverage
images routed to retrieval, taking their median error from 160.19 km to 0.36 km.

A 0.6–0.65 cut looks reasonable a priori and would have been a disaster: it
accepts 903 of those 1000 external images and pushes their median error from
1292 km to 9787 km.

Reproduce with `experiments/retrieval/calibrate_ensemble.py` and
`pick_threshold.py`.

### Why it was shelved

**It worked in-gallery.** Inside coverage the ensemble cut median error from
160.19 km to 0.36 km. That result stands.

**It contributed nothing globally.** On the 1000-image external diverse set the
ensemble was *identical* to GeoCLIP alone — 0 of 1000 images routed to
retrieval. No harm, and no gain. Its value scaled with gallery coverage and
nothing else, and the gallery covered 3314 photos against a planet.

**The scaling limit is the real reason.** Recalibrating against a 1.9x larger
gallery (v1 1719 → v2 3314 images) left the threshold unmoved, but only because
two opposing shifts cancelled:

| population | v1 (1719 img) | v2 (3314 img) |
|---|---|---|
| out-of-coverage max | 0.803 | 0.830 |
| out-of-coverage mean | 0.668 | 0.704 |
| in-coverage median | 0.851 | 0.863 |

A larger gallery gives every query more chances at a spuriously similar photo,
so **the false-accept ceiling rises with gallery size**. That is a property of
size itself, not of which regions were added. Scaling the gallery toward global
coverage therefore raises the very floor the trust threshold has to clear — the
approach fights itself as it grows, and the threshold would need re-deriving on
every significant expansion.

A concrete illustration of the coverage limit: for a photograph of the Penang
Second Bridge, the gallery held no image within 972 km, and zero in Malaysia at
all. Retrieval cannot help outside its footprint, and expanding the footprint
enough to matter is what the scaling limit blocks.

### What remains

`viewfinder/openclip_support.py` holds the OpenCLIP loading helpers this code
introduced. They are still in the active path: the `hybrid` predictor's local
Wikimedia Commons landmark matcher uses the same embedding-space conventions.

The shelved tests run in place, with the CLI-level ones skipped since the
behaviour they assert was removed:

```bash
python -m pytest experiments/retrieval/tests/
```

## Requirements

- Python 3.10 or newer
- Enough memory to run GeoCLIP's CLIP ViT-L/14 backbone
- A CUDA GPU is used automatically when available; CPU inference also works
- A local video codec supported by OpenCV for video inputs
- FFmpeg and `yt-dlp` for public Instagram/TikTok URL inputs

## Quickstart from a fresh clone

These are the exact, verified steps to go from a clean `git clone` to a working
prediction. No Mapillary token is needed; model weights download once on first
run.

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

The default command runs GeoCLIP and needs no flags beyond the file:

```bash
python -m viewfinder.main images/IMG_6790.png
```

For the fail-closed evidence path, which returns `prediction: null` rather than
an unsupported guess, ask for it explicitly:

```bash
view-finder locate images/IMG_6790.png --predictor hybrid --require-evidence
```

It can emit immediately when the original file has GPS metadata. Otherwise it
returns `prediction: null` unless online evidence is configured as described
below. PLONK is preferred for manual visual hypotheses when its optional runtime
is installed; the cached GeoCLIP model is used if PLONK is unavailable.

Expected on a correct setup: GeoCLIP reports a California result for the bundled
test image. The dependencies in `viewfinder/requirements.txt` are pinned to the
exact versions these results were produced with, so a fresh clone reproduces
them.

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

`viewfinder/requirements.txt` includes `faiss-cpu`, `open-clip-torch`,
`torchvision` and `pyarrow`. These are still required: the `hybrid` predictor's
local Commons landmark matcher uses OpenCLIP and FAISS. Cache its checkpoint
once with:

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

The `hybrid` predictor's landmark matcher searches a local FAISS gallery of
Wikimedia Commons imagery, built by `gallery-pipeline`. Prebuilt galleries are
checked into `gallery-pipeline/data/*/gallery/vN/` (the FAISS index + metadata +
manifest — not the raw source images, which are not required at query time).

The now-shelved retrieval predictor used the same galleries; see
[Retrieval and ensemble](#retrieval-and-ensemble-explored-and-shelved).

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

CLI options:

```text
SOURCE             Local image/video path or public Instagram/TikTok URL
--predictor NAME   geoclip (default) | hybrid | cloud-hybrid | plonk-osv |
                   plonk-yfcc | trusted
--local-data PATH  Local evidence root; default ./local-data
--online-evidence  cloud-hybrid only: use Cloud Vision + Google Places
--require-evidence Hybrid modes: exit 3 unless evidence authorizes automation
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
--max-predictions INT  Distinct location clusters returned; default 5
--output PATH      JSON path; defaults beside the input file
--gallery PATH     Optional custom GPS gallery for GeoCLIP (see below)
--gallery-mode M   augment (default) | replace, for --gallery
--crop-screen M    auto (default) | off. Crop to a detected display/framed
                   picture; auto acts only on a confident detection
--verbose, -v      Debug logging for pipeline stages
--quiet, -q        Print only the machine-readable prediction on stdout
```

### Photos of screens (`--crop-screen`)

A photo *of a display* is not a photo of a place. Handed one, the pipeline is
asked "where is this laptop sitting in a room", and the scene on screen reaches
the model at roughly a hundred pixels of vertical resolution after CLIP's
`Resize(224) + CenterCrop(224)` — alongside the bezel, the desk, the clock and
the OS's news tiles.

`--crop-screen auto` (the default) detects a display or framed picture and crops
to its content, perspective-correcting it on the way. Measured on a controlled
test — a synthetic lock-screen photo whose wallpaper is a known image, so the
right answer is known:

| run | prediction | error vs. the wallpaper alone |
|---|---|---|
| screen photo, `--crop-screen off` | A Coruña, Spain | 7393 km |
| screen photo, `--crop-screen auto` | Lawrence, Kansas | 445 km |

**Auto mode only acts on a confident detection.** Cropping an ordinary
photograph would discard most of the frame on a false positive, so every gate in
`viewfinder/screen_crop.py` must pass: area, aspect ratio, interior/exterior
contrast, rectangularity, corner squareness, opposite-side equality, and — the
gate that does most of the work — *edge support*, the fraction of the quad's
perimeter that lies on a real detected boundary. A bezel scores near 1.0; four
scenery features that happen to line up do not.

The first four gates alone cropped 18 of 400 real photographs (4.5%). With the
full set that is **4 of 900 (0.44%)**, while both a real and a synthetic screen
photo still crop. Re-run that measurement if you touch any threshold: loosening
one to make a real photograph detect at all pushed the rate back to 2.5% until
the aspect, area and angle gates were tightened to compensate.

### What cropping does not fix

Cropping fixes the input, not the model. On a real Windows lock-screen photo of
the Penang Second Bridge the crop succeeds and the prediction does not move:

| run | prediction | error |
|---|---|---|
| `--crop-screen off` | Mountain View, California | 13549 km |
| `--crop-screen auto` | Mountain View, California | 13549 km |
| `--crop-screen auto --remove-text` | Pingtung, Taiwan | 2903 km |

The overlaid clock and date dominate whatever is left after cropping — removing
the tile strip alone moves the answer by 47 km, removing the clock and date
moves it 10,000 km. And even fully cleaned the answer is East Asia, not
Malaysia, because **the gallery holds no photograph within 972 km of Penang**
(zero entries in Malaysia at all), and GeoCLIP does not know this bridge. For a
location like this the fix is training data or gallery coverage, not
preprocessing. This measurement was taken while the retrieval/ensemble
predictor was still active; it is one of the results that led to shelving it.

When a crop happens it is reported in the summary and recorded under
`processing.screen_crop` in the JSON, so it is never a silent change:

```text
  Preprocessing cropped to a detected screen covering 74% of the photo
```

Two limits worth knowing: this applies to still images, not video frames, and
cropping fixes the *input* — the scene still has to be something the model or
the gallery actually knows. Compose it with `--remove-text` to also inpaint
overlay text left inside the cropped region.

### Progress and the result summary

Video runs draw a progress bar for frame sampling and for model inference. Bars
are written to stderr and only when stderr is a terminal, so piped output and
log files stay clean. They are suppressed under `--verbose`, where the per-batch
debug lines already report progress.

A compact summary is always the last thing printed, at every verbosity level, so
a long run's answer is never buried in stage logs:

```text
==================================================================
  RESULT: Erlands Point-Kitsap Lake, Washington, US
==================================================================
  Coordinates  47.523348, -122.745066
  Confidence   74.3%
  Predictor    geoclip
  Alternatives Mill Valley, California, US (37%)
==================================================================
```

Under `--quiet` the summary goes to **stderr**, leaving
stdout as exactly the JSON that scripts already parse:

```bash
python -m viewfinder.main photo.jpg --quiet > result.json
```

## Desktop GUI

A small Tkinter window covers the common case — pick a file, pick a predictor,
read the answer — without a terminal. Tkinter ships with Python, so there is
nothing extra to install:

```bash
python viewfinder/gui.py
```

It also runs as a module:

```bash
python -m viewfinder.gui
```

The window offers a file picker, radio buttons for `geoclip` (default) and
`hybrid`, a Run button, and a progress bar that switches from indeterminate to a
real percentage once the pipeline knows the frame count. The results panel shows
the place name, coordinates, confidence, and the ranked alternatives, plus the
path of the JSON written beside the input file (the same default the CLI uses).

The GUI imports `viewfinder.main` and calls the same orchestration the CLI
calls — it does not shell out — so the predictor default, screen-crop step, and
error messages are identical in both. Inference runs on a worker thread, so the
window stays responsive; failures are shown in the results panel rather than
raised as a traceback into a console nobody is watching.

The landmark gallery's query encoder is read from its own `manifest.json`
rather than hardcoded, so a gallery is always searched with the model that built
it. A mismatch between query and gallery embedding spaces would make the search
meaningless, so a gallery built with a non-OpenCLIP encoder is rejected outright.

Note `--gallery`/`--gallery-mode` are a separate, GeoCLIP-only feature: they
extend GeoCLIP's *coordinate* pool with coordinates harvested by the pipeline.

Supported formats depend on the installed Pillow/OpenCV decoders. Run
`view-finder formats` for the exact image and video
extensions available in the current environment. Format detection is
extension-based and case-insensitive.

## 3D World Engine

`view-finder world` turns a single photo into an explorable 3D world you can
move through in a browser, including on a phone. It estimates depth locally
(Depth Anything V2 Small, Apache-2.0), builds a layered mesh (foreground,
the hidden background behind silhouettes, sky, dome and ground) and exports a
standard `.glb` with a dependency-free WebGL viewer.

```bash
view-finder world setup                          # once: download the depth model (~100 MB)
view-finder world build photo.jpg --html         # photo_world.glb/.json/.html
view-finder world serve --host 0.0.0.0           # open the printed link on your phone
```

There is also a **desktop app** ([desktop/README.md](desktop/README.md)): the
same engine in its own window, with no network port, running only on the
NVIDIA GPU:

```bash
cd desktop && npm install && npm start
```

The phone path resizes photos on the device before upload and protects the
LAN server with an access token. Everything stays on your network. See
[docs/3d-engine/README.md](docs/3d-engine/README.md) for usage and
[docs/3d-engine/ARCHITECTURE.md](docs/3d-engine/ARCHITECTURE.md) for how it
works.

## Discord bot

`viewfinder/discord_bot.py` is a standalone, always-on Discord process. It is
not imported by, merged with, or dependent on the News bot. Use a **new Discord
application and token** so the two bots retain separate identities.

1. Open [Discord Developer Portal](https://discord.com/developers/applications),
   choose **New Application**, give it a View Finder-specific name, and add a bot
   on the **Bot** page.
2. On that Bot page, enable **Message Content Intent**. Attachments can arrive
   without it, but URL text cannot.
3. Reset/copy the bot token. Copy `.env.example` to `.env`, then set
   `DISCORD_BOT_TOKEN=...` there. `.env` is gitignored; the token is a password
   and must never be committed or pasted into logs.
4. Under **OAuth2 -> URL Generator**, select the `bot` scope and grant **View
   Channels**, **Send Messages**, and **Read Message History**. Open the generated
   URL to invite this new bot to a server. Sharing a server also lets you DM it
   when your Discord privacy settings permit DMs from server members.
5. Install and run it from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r viewfinder\requirements-discord.txt
.\.venv\Scripts\python.exe -m viewfinder.discord_bot
```

Keep that process running for the bot to respond. In either a DM or an accessible
channel, send an image/video attachment or a recognized media URL. The bot first
replies that it is processing, queues concurrent requests to stay within GPU
memory, then edits the reply with the location, coordinates, confidence, and top
alternatives. Link downloads use `yt-dlp`, are capped at 200 MB and 10 minutes by
default, and report private/login-gated/rate-limited Instagram failures plainly.
The limits can be changed with the variables documented in `.env.example`.

## Discord bot

A standalone bot that geolocates any image or video you send it. Send a photo
as an attachment, or paste a link, in a DM or in any channel the bot can see;
it replies with the place name, coordinates, a map link, confidence, and the
top alternatives.

It is a **separate process with its own bot identity**. If you already run
another bot, this one needs its own application and its own token — two
processes cannot share a token, and Discord will disconnect both if they try.

### 1. Create the application

1. Go to <https://discord.com/developers/applications> and click
   **New Application**. Name it whatever you like (e.g. `View Finder`).
2. Open the **Bot** tab and click **Add Bot**.
3. Under **Privileged Gateway Intents**, enable **Message Content Intent**.
   This is required: without it the bot still sees attachments, but reads every
   message body as empty, so pasted links silently stop working.
4. Click **Reset Token**, then **Copy**. This is the only time it is shown.

### 2. Put the token in `.env`

```bash
cp .env.example .env
```

Then set the token in `.env`:

```
DISCORD_BOT_TOKEN=your-token-here
```

`.env` is gitignored. Never commit a real token — anyone holding it can act as
your bot. The bot never prints or logs it, including in error messages.

### 3. Invite it

In the developer portal, open **OAuth2 → URL Generator**:

- **Scopes:** `bot`
- **Bot Permissions:** `Send Messages`, `Read Message History`,
  `Attach Files`, and `View Channel`

Open the generated URL and pick a server. For DMs, you and the bot only need to
share one server — you do not need to invite it to a channel specifically.

### 4. Install and run

```bash
python -m pip install -r viewfinder/requirements-discord.txt
python viewfinder/discord_bot.py
```

**It only responds while it is running.** There is no hosted component; close
the terminal and the bot goes offline. Keep it in its own terminal, or register
it as a background service.

### What it does and does not handle

| input | behaviour |
|---|---|
| Image or video attachment | Downloaded and predicted |
| Instagram / TikTok / YouTube / Twitter / Reddit / Vimeo link | Fetched with yt-dlp, then predicted |
| Direct link ending in a media extension | Fetched and predicted |
| Any other link, or plain text | Ignored, so the bot stays quiet in a busy channel |
| Message from another bot | Ignored, so two bots cannot talk to each other forever |

Safeguards: attachments are capped at 50 MB and link downloads at 200 MB and
600 seconds (all configurable in `.env`), a "working on it" reply goes out
immediately since inference takes real time, and predictions are serialised —
one GeoCLIP instance nearly fills the GPU, so simultaneous messages queue
rather than loading a second copy.

**Instagram links fail unpredictably**, and not because of anything you did:
Instagram blocks anonymous downloads, rate-limits, and periodically breaks
yt-dlp's extractor. The bot says so plainly and suggests downloading the video
and attaching it directly, which always works. It never replies with a stack
trace or goes silent.

### Relationship to the news bot

This mirrors the conventions of the separate news bot — `DISCORD_BOT_TOKEN`
loaded from `.env` via python-dotenv, a token that is never logged — but shares
no code and no identity with it. One difference: the news bot calls Discord's
REST API directly and polls, because it only ever pushes a digest on a
schedule. This bot has to react to arbitrary incoming messages, so it uses
discord.py's gateway client instead.

## Pipeline

For an image, View Finder loads and orients the image, obtains ranked
coordinates from the selected predictor, geographically clusters the top-k
results, independently scores each distinct location hypothesis, and
reverse-geocodes every returned cluster.

Every predictor shares the clustering, confidence scoring, and reverse-geocoding
stages, so they produce the same shape of ranked result. Scores are normalised
per frame against that frame's best candidate, which is what lets differently
scaled model outputs flow through the same scoring code unchanged.

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
  "predictors": ["geoclip", "plonk-osv", "plonk-yfcc"],
  "results": {
    "geoclip":    { "predictor": "geoclip",    "prediction": {}, "predictions": [] },
    "plonk-osv":  { "predictor": "plonk-osv",  "prediction": {}, "predictions": [] },
    "plonk-yfcc": { "predictor": "plonk-yfcc", "prediction": {}, "predictions": [] }
  }
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
