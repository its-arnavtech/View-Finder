"""Airflow TaskFlow DAG for the View Finder gallery ETL pipeline."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:  # Airflow 3 public authoring API; fallback keeps the DAG parseable on 2.10.
    from airflow.sdk import dag, task
except ImportError:  # pragma: no cover - exercised only on Airflow 2.x
    from airflow.decorators import dag, task
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.config import ensure_data_directories, load_config  # noqa: E402
from src.extract.runner import run_extract  # noqa: E402
from src.load.index_builder import run_load  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.quality_check.gallery_validator import run_quality_check  # noqa: E402
from src.transform.runner import run_transform  # noqa: E402
from src.validate.runner import run_validate  # noqa: E402


LOGGER = logging.getLogger(__name__)
DEFAULT_ARGS = {
    "owner": "view-finder-data-engineering",
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=15),
}


def _runtime_config() -> dict:
    configure_logging()
    config = load_config()
    ensure_data_directories(config)
    return config


@dag(
    dag_id="gallery_etl",
    description="Build a balanced, searchable geotagged-image gallery from Mapillary",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["view-finder", "etl", "mapillary", "faiss"],
)
def gallery_etl():
    @task(task_id="extract")
    def extract_task() -> dict:
        result = run_extract(_runtime_config())
        LOGGER.info("Extract metrics: %s", result)
        return result

    @task(task_id="validate")
    def validate_task(extraction: dict) -> dict:
        result = run_validate(_runtime_config(), extraction.get("tile_ids"))
        LOGGER.info("Validate metrics: %s", result)
        return result

    @task(task_id="transform")
    def transform_task(validation: dict) -> dict:
        result = run_transform(_runtime_config(), validation.get("tile_ids"))
        LOGGER.info("Transform metrics: %s", result)
        return result

    @task(task_id="load")
    def load_task(transformation: dict) -> dict:
        LOGGER.info("Loading %d retained images", transformation.get("retained", 0))
        result = run_load(_runtime_config())
        LOGGER.info("Load metrics: %s", result)
        return result

    @task(task_id="quality_check")
    def quality_check_task(build: dict) -> dict:
        result = run_quality_check(_runtime_config(), build["version_dir"])
        LOGGER.info("Quality-check metrics: passed=%s warnings=%s", result["passed"], result["warnings"])
        return {"version": build["version"], "passed": result["passed"]}

    extracted = extract_task()
    validated = validate_task(extracted)
    transformed = transform_task(validated)
    built = load_task(transformed)
    quality_check_task(built)


gallery_etl_dag = gallery_etl()
