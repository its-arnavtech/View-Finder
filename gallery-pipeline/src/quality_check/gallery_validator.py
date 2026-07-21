"""Validate a newly built FAISS gallery and emit machine/human-readable reports."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow.parquet as pq


LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"Gallery quality report: v{report['version']}",
        f"Status: {'PASS' if report['passed'] else 'FAIL'}",
        f"Generated: {report['checked_at']}",
        "",
        "Checks:",
    ]
    for name, result in report["checks"].items():
        lines.append(f"- {name}: {result['status'].upper()} - {result['message']}")
    if report["warnings"]:
        lines.extend(["", "Warnings:", *[f"- {warning}" for warning in report["warnings"]]])
    if report["critical_failures"]:
        lines.extend(["", "Critical failures:", *[f"- {failure}" for failure in report["critical_failures"]]])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_quality_check(config: dict[str, Any], version_dir: str | Path) -> dict[str, Any]:
    version_path = Path(version_dir)
    match_version = version_path.name.removeprefix("v")
    version = int(match_version) if match_version.isdigit() else -1
    checks: dict[str, dict[str, str]] = {}
    warnings: list[str] = []
    critical: list[str] = []
    manifest: dict[str, Any] = {}

    try:
        manifest = json.loads((version_path / "manifest.json").read_text(encoding="utf-8"))
        index = faiss.read_index(str(version_path / "gallery.faiss"))
        metadata = pq.read_table(version_path / "metadata.parquet")
    except Exception as exc:
        critical.append(f"Gallery artifacts could not be read: {exc}")
        checks["artifact_integrity"] = {"status": "fail", "message": str(exc)}
        index = None
        metadata = None
    else:
        checks["artifact_integrity"] = {"status": "pass", "message": "Manifest, FAISS index, and metadata are readable"}

    if index is not None and metadata is not None:
        row_count = metadata.num_rows
        minimum = int(config["quality_check"].get("minimum_total_images", 1))
        expected_count = int(manifest.get("total_image_count", -1))
        if row_count < minimum or index.ntotal != row_count or expected_count != row_count:
            message = f"metadata={row_count}, index={index.ntotal}, manifest={expected_count}, minimum={minimum}"
            critical.append(f"Row-count consistency failed: {message}")
            checks["row_count"] = {"status": "fail", "message": message}
        else:
            checks["row_count"] = {"status": "pass", "message": f"All artifacts contain {row_count} images"}

        image_ids = metadata["image_id"].to_pylist()
        duplicate_count = len(image_ids) - len(set(image_ids))
        if duplicate_count:
            critical.append(f"Final metadata contains {duplicate_count} duplicate image_id values")
            checks["duplicate_image_ids"] = {"status": "fail", "message": f"Found {duplicate_count} duplicates"}
        else:
            checks["duplicate_image_ids"] = {"status": "pass", "message": "All image_id values are unique"}

        manifest_dimension = int(manifest.get("embedding_dimension", -1))
        try:
            reconstructed = index.reconstruct_n(0, index.ntotal)
            consistent_dimension = reconstructed.ndim == 2 and reconstructed.shape[1] == index.d == manifest_dimension
            finite = bool(np.isfinite(reconstructed).all())
        except Exception as exc:
            consistent_dimension = False
            finite = False
            warnings.append(f"Index reconstruction failed: {exc}")
        if not consistent_dimension or not finite:
            message = f"index_dimension={index.d}, manifest_dimension={manifest_dimension}, finite={finite}"
            critical.append(f"Embedding consistency failed: {message}")
            checks["embedding_consistency"] = {"status": "fail", "message": message}
        else:
            checks["embedding_consistency"] = {"status": "pass", "message": f"All vectors have dimension {index.d} and finite values"}

        region_counts = Counter(metadata["region"].to_pylist())
        expected_regions = config["quality_check"].get("expected_regions", [])
        minimum_fraction = float(config["quality_check"].get("minimum_region_fraction", 0.01))
        underrepresented = []
        for region in expected_regions:
            fraction = region_counts.get(region, 0) / max(row_count, 1)
            if fraction < minimum_fraction:
                underrepresented.append(f"{region} ({fraction:.2%})")
        if underrepresented:
            warning = "Underrepresented expected regions: " + ", ".join(underrepresented)
            warnings.append(warning)
            checks["geographic_distribution"] = {"status": "warning", "message": warning}
        else:
            checks["geographic_distribution"] = {"status": "pass", "message": f"Coverage: {dict(sorted(region_counts.items()))}"}

    report = {
        "version": version,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "passed": not critical,
        "checks": checks,
        "warnings": warnings,
        "critical_failures": critical,
    }
    version_path.mkdir(parents=True, exist_ok=True)
    _write_json(version_path / "quality_report.json", report)
    _write_summary(version_path / "quality_report.txt", report)

    if manifest:
        manifest["status"] = "quality_passed" if report["passed"] else "quality_failed"
        manifest["quality_checked_at"] = report["checked_at"]
        _write_json(version_path / "manifest.json", manifest)
    if critical:
        LOGGER.error("Gallery v%d failed quality checks: %s", version, critical)
        raise ValueError("Critical gallery quality checks failed: " + "; ".join(critical))
    LOGGER.info("Gallery v%d passed quality checks with warnings=%s", version, warnings)
    return report

