# View Finder Gallery Pipeline

A standalone, resumable ETL pipeline that collects geotagged Mapillary images, removes bad and repetitive data, creates OpenCLIP image embeddings, and publishes immutable FAISS gallery versions. Airflow provides orchestration, retries, history, and task-level operational logs; SQLite is the durable control plane and metadata catalog.

## Architecture

```text
Mapillary Graph API
        |
        v
 extract --> validate --> transform --> load --> quality_check
    |          |             |           |            |
  tiled     decode +      balance +   FAISS vN/    JSON + text
 requests   pHash dedup   quality +    manifest     reports
    |                      OpenCLIP
    +---------- SQLite catalog/checkpoints ----------+
```

Every Airflow stage is a separate TaskFlow task. The DAG applies three retries with exponential backoff and a 15-minute maximum delay. `max_active_runs=1` prevents concurrent gallery publication from the same local installation.

## Engineering decisions

- **Deliberate coverage:** `src/extract/tiling.py` creates stable latitude/longitude cells. Three tiling modes: `regions` (hand-listed bounding boxes, the development default), `world` (uniform 1-degree grid), and `global` (population/land-aware programmatic grid — skips ocean/uninhabited cells and scales cell size with local city density; see "Incremental global coverage"). `max_tiles_per_run` (and, for `global`, `extract.max_runtime_seconds`) bounds each run, so collection advances incrementally instead of issuing tens of thousands of requests at once.
- **Safe resume:** a tile is set to `in_progress` before its first request and `completed` only after its final page and download. An interruption leaves it eligible for retry. Image IDs are unique in SQLite, existing files are skipped, and downloads use `.part` files followed by an atomic rename.
- **API behavior:** Mapillary calls are bounded by tile, page size, and per-tile image cap. HTTP 429 and transient 5xx responses use exponential backoff with jitter; `Retry-After` is honored. The access token is read only from the environment.
- **Validation:** Pillow fully decodes every file. Per-tile perceptual hashes remove near duplicates while retaining the first stable `image_id` ordering. Failed raw files remain on disk for audit but cannot enter later stages.
- **Quality and indoor filtering:** a NumPy Laplacian-variance score removes blurred images. Because the pipeline already loads OpenCLIP, its image/text similarity provides a lightweight indoor/outdoor signal; images above a conservative 0.65 indoor probability are excluded by default.
- **Embeddings:** `open_clip_torch` with `ViT-B-32` and `laion2b_s34b_b79k` produces normalized vectors in batches. CUDA is selected automatically; CPU uses a smaller batch. Each tile is persisted as compressed Parquet before publication.
- **Geographic balance:** after quality filtering, the sharpest 200 images per tile are retained by default. Stable image IDs break score ties, making reruns repeatable.
- **Coverage labels:** manifest continent-scale labels use transparent coordinate bounding heuristics and are intended for distribution alerts, not geopolitical analysis.
- **Immutable builds:** load atomically reserves the next unused `data/gallery/vN/` directory and never reuses it. Quality checks change only the new manifest's build status from pending to passed/failed.

## Project layout

```text
gallery-pipeline/
  dags/gallery_etl_dag.py
  src/{extract,validate,transform,load,quality_check}/
  config/tiling_config.yaml
  data/                       # runtime-only; gitignored
  tests/
  requirements.txt
  .env.example
```

## Setup (WSL, Linux, or macOS)

Airflow does not support native Windows as a production/runtime environment. On Windows, run these commands in WSL 2 or an Airflow container. Python 3.11 is recommended.

```bash
cd gallery-pipeline
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

AIRFLOW_VERSION=3.3.0
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
pip install -r requirements.txt
```

The constraint file is used only for Airflow's reproducible base installation. Project dependencies are installed in the second command without that constraint while retaining the exact Airflow pin, following [Apache's documented custom-dependency workflow](https://airflow.apache.org/docs/apache-airflow/3.3.0/installation/installing-from-pypi.html).

OpenCLIP downloads its pretrained weights on the first transform run, so that run needs internet access and several hundred MB of free cache space. CPU execution works but is much slower than CUDA.

Copy the environment template and set the Mapillary client token—never commit the resulting file:

```bash
cp .env.example .env
# Edit .env and set MAPILLARY_ACCESS_TOKEN.
```

Mapillary's developer documentation and token setup are at <https://www.mapillary.com/developer>.

## Initialize and run Airflow

Use an absolute `AIRFLOW_HOME`; keeping it inside this project makes the installation self-contained.

```bash
export AIRFLOW_HOME="$(pwd)/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/dags"
export AIRFLOW__CORE__EXECUTOR=SequentialExecutor

airflow db migrate
airflow standalone
```

The command prints an admin password and starts the UI at <http://localhost:8080>. Enable and trigger `gallery_etl` in the UI, or use another shell with the same virtual environment and variables:

```bash
airflow dags unpause gallery_etl
airflow dags trigger gallery_etl
```

For a synchronous small-region check, leave the checked-in regional config and lower `max_tiles_per_run` to `1`, then run:

```bash
airflow dags test gallery_etl 2026-01-01
```

Task logs include tile bounding boxes, images discovered and newly inserted, validation reason counts, images surviving each transform filter, embedding batch progress, and final build/report metrics.

## Checkpoint and resume behavior

`data/catalog.db` contains `tiles`, `images`, `stage_results`, and `transformed_assets`. On a rerun, extraction ignores only `completed` tiles. A `failed` or interrupted `in_progress` tile is requested again; already cataloged image IDs are not downloaded or inserted again. Do not delete `catalog.db` if you want to retain progress. SQLite uses WAL mode and short transactions so metadata survives normal process interruption.

To inspect progress:

```bash
sqlite3 data/catalog.db \
  "select status, count(*) as tiles, sum(images_found) as images from tiles group by status;"
```

## Incremental global coverage

`tiling.mode: global` builds a worldwide grid **programmatically** from population
data instead of hand-listed regions or a uniform world grid. See
`config/tiling_config_global.yaml`.

**How it works** (`src/extract/global_tiling.py`):

- **Data source:** GeoNames *cities1000* — every settlement on Earth with
  population ≥ 1000 (144,563 points) — bundled offline inside the
  `reverse_geocoder` package (no download). City points are a proxy for both
  *inhabited land* (a grid cell with no cities is ocean/desert/wilderness and is
  skipped — empirically ~86% of 1° cells) and *population density* (the count of
  cities in an area).
- **Adaptive cell size:** each populated 1° "supercell" is counted and mapped
  through `density_bands` to a Mapillary-safe cell size — denser → finer. The
  defaults (validated live): ≥100 cities → 0.02°, ≥30 → 0.03°, ≥8 → 0.05°,
  ≥1 → 0.08°. So Paris/London/NYC tile at 0.02° while rural land tiles at 0.08°.
  Sizes stay under Mapillary's 0.01 sq-degree bbox-area limit, and a cell the API
  still rejects (too dense, or too large) is auto-subdivided and retried.
- **Ordering:** `spread` (default) emits one tile per supercell per pass, densest
  first, so a bounded run samples many places worldwide rather than exhausting a
  single metro. `density` concentrates on the biggest cities first. Both are fully
  deterministic.

**Running it incrementally.** A real global build is far too large for one run, so
bound each session and repeat — the checkpoint store skips `completed` tiles, so
successive sessions resume and never re-query finished work:

```bash
export GALLERY_HOME=/path/to/gallery-pipeline
export AIRFLOW_HOME=~/airflow-gallery
export GALLERY_CONFIG=config/tiling_config_global.yaml   # set bounds: null for worldwide

# Run this repeatedly (e.g. from cron or a loop). Each invocation advances the
# global frontier by at most max_tiles_per_run tiles / max_runtime_seconds seconds.
while true; do
  airflow dags test gallery_etl "$(date +%Y-%m-%d)"
  # ...stop when satisfied, or sleep between sessions to respect API limits
done
```

Bound each session two ways (either or both, in the YAML):

- `tiling.max_tiles_per_run` — hard tile budget per invocation.
- `extract.max_runtime_seconds` — wall-clock budget; extraction stops cleanly
  after the current tile and leaves the rest pending (`extract` reports
  `stopped_early: true`). `0` disables the limit.

Each session runs the full `extract → … → quality_check` chain, so it also
publishes a new gallery version incorporating everything gathered so far. Inspect
the global frontier at any time:

```bash
sqlite3 data/global-test/catalog.db \
  "select status, count(*) from tiles group by status;"
```

## Gallery versions and quality reports

Each successful load reserves a new directory:

```text
data/gallery/v1/
  gallery.faiss
  metadata.parquet
  manifest.json
  quality_report.json
  quality_report.txt
```

`metadata.parquet` maps each zero-based FAISS position to `image_id`, coordinates, tile, coarse region, and raw path. The manifest records the vector dimension/model, UTC build time, total images, tiles, and region counts. A later run creates `v2`, even when `v1` failed quality checks; old versions are never overwritten.

Critical failures—unreadable artifacts, an empty/undersized build, row-count mismatch, duplicate IDs, inconsistent dimensions, or non-finite vectors—write reports and then fail the Airflow task/DAG. Expected-region shortfalls are warnings because development builds intentionally cover only a few places. For a global production crawl, populate `quality_check.expected_regions` in the YAML.

## Credential-free verification

The test suite substitutes a deterministic API transport and embedding generator but exercises the real checkpoint store, image validation, Parquet writer, FAISS builder, version allocator, and gallery validator:

```bash
pytest
python -m compileall -q dags src tests
```

To prove task execution inside Airflow without a token or network API dependency, the extractor also has an explicit fixture-only smoke mode. It is never selected by default:

```bash
export GALLERY_EXTRACT_MODE=fixture
export GALLERY_CONFIG=config/airflow_smoke_config.yaml
airflow tasks test gallery_etl extract 2026-01-01
unset GALLERY_EXTRACT_MODE GALLERY_CONFIG
```

That command runs the real decorated Airflow task, tiling logic, image writes, SQLite catalog, and checkpoint transition. Fixture rows are labeled `mapillary_fixture`, use the isolated `data/airflow-smoke/` directory, and cannot be mistaken for live API data.
