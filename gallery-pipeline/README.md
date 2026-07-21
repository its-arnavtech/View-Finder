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

- **Deliberate coverage:** `src/extract/tiling.py` creates stable latitude/longitude cells. The checked-in development config uses denser grids around three example regions. Set `tiling.mode: world` for a global 1-degree grid. `max_tiles_per_run` bounds each run, so global collection advances incrementally instead of issuing tens of thousands of requests at once.
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
