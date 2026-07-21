"""Parse the DAG with Airflow and assert its operational contract."""

from __future__ import annotations

from pathlib import Path

from airflow.models import DagBag


project_root = Path(__file__).resolve().parents[1]
try:
    bag = DagBag(
        dag_folder=str(project_root / "dags"), include_examples=False, safe_mode=False
    )
except TypeError:  # Airflow 3 removed the include_examples argument from DagBag.
    bag = DagBag(dag_folder=str(project_root / "dags"), safe_mode=False)
if bag.import_errors:
    raise RuntimeError(f"DAG import errors: {bag.import_errors}")

# Access the parsed in-memory object directly; this check intentionally does not require
# an initialized Airflow metadata database.
dag = bag.dags.get("gallery_etl")
if dag is None:
    raise RuntimeError("gallery_etl was not discovered")

expected_order = ["extract", "validate", "transform", "load", "quality_check"]
if set(dag.task_ids) != set(expected_order):
    raise RuntimeError(f"Unexpected task IDs: {dag.task_ids}")
for upstream, downstream in zip(expected_order, expected_order[1:]):
    if downstream not in dag.get_task(upstream).downstream_task_ids:
        raise RuntimeError(f"Missing dependency {upstream} -> {downstream}")
for task_id in expected_order:
    task = dag.get_task(task_id)
    if task.retries != 3 or not task.retry_exponential_backoff:
        raise RuntimeError(f"Retry policy is incorrect for {task_id}")

print("DAG import: OK")
print("Task order: " + " -> ".join(expected_order))
print("Retry policy: 3 retries with exponential backoff on every task")
