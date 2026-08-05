"""Build View Finder's fully local, no-key geographic evidence datasets.

Every long stage is resumable. Run ``status`` at any time to inspect progress.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_root in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from geonames_builder import build_geonames_database, download_geonames
from landmark_builder import (
    build_landmark_index,
    catalog_stats,
    discover_commons_assets,
    download_commons_assets,
    plan_targets,
)
from osm_builder import download_osm_regions, ingest_osm_region


LOGGER = logging.getLogger("local-data-builder")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build global GeoNames, local OCR models, a bounded Wikimedia Commons "
            "landmark gallery, and optional regional OSM POIs without API keys."
        )
    )
    parser.add_argument(
        "command",
        choices=(
            "geonames-download",
            "geonames-build",
            "geonames-all",
            "ocr-cache",
            "landmarks-plan",
            "landmarks-discover",
            "landmarks-download",
            "landmarks-index",
            "landmarks-all",
            "osm-download",
            "osm-ingest",
            "osm-all",
            "all",
            "status",
        ),
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "config.default.json"),
        help="JSON build configuration",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="optional target/image limit for smoke testing a long stage",
    )
    parser.add_argument(
        "--region",
        action="append",
        help="OSM region name from config; repeatable (default: every configured region)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(payload["data_root"]).expanduser()
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    payload["data_root"] = str(root)
    return payload


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["data_root"])
    return {
        "root": root,
        "geonames_raw": root / "geonames" / "raw",
        "gazetteer": root / "geonames" / "geonames.sqlite3",
        "ocr_models": root / "ocr-models",
        "landmark_catalog": root / "landmarks" / "catalog.sqlite3",
        "landmark_images": root / "landmarks" / "images",
        "landmark_gallery": root / "landmarks" / "gallery",
        "osm_raw": root / "osm" / "raw",
    }


def _check_storage(paths: dict[str, Path], budget_gb: float) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(paths["root"])
    required = int((budget_gb + 8.0) * 1024**3)
    LOGGER.info(
        "Storage free=%.1f GiB; configured landmark budget=%.1f GiB",
        usage.free / 1024**3,
        budget_gb,
    )
    if usage.free < required:
        raise RuntimeError(
            f"Only {usage.free / 1024**3:.1f} GiB is free, but the configured "
            f"{budget_gb:.1f} GiB landmark corpus plus indexes needs at least "
            f"{required / 1024**3:.1f} GiB. Change data_root or reduce the budget."
        )


def _geonames_download(config: dict[str, Any], paths: dict[str, Path]) -> None:
    download_geonames(paths["geonames_raw"], user_agent=config["user_agent"])


def _geonames_build(paths: dict[str, Path]) -> None:
    build_geonames_database(paths["geonames_raw"], paths["gazetteer"])


def _ocr_cache(paths: dict[str, Path]) -> None:
    from viewfinder.local_ocr import PaddleSceneText

    LOGGER.info("Loading PaddleOCR once to populate %s", paths["ocr_models"])
    PaddleSceneText(paths["ocr_models"])
    LOGGER.info("PaddleOCR model cache is ready")


def _landmarks_plan(config: dict[str, Any], paths: dict[str, Path]) -> None:
    settings = config["landmarks"]
    plan_targets(
        paths["gazetteer"],
        paths["landmark_catalog"],
        us_limit=int(settings["us_target_limit"]),
        world_limit=int(settings["world_target_limit"]),
        seed_geoname_ids=settings.get("seed_geoname_ids", ()),
        seed_targets=settings.get("seed_targets", ()),
    )


def _landmarks_discover(
    config: dict[str, Any], paths: dict[str, Path], limit: int | None
) -> None:
    result = discover_commons_assets(
        paths["landmark_catalog"],
        config["landmarks"],
        user_agent=config["user_agent"],
        limit=limit,
    )
    LOGGER.info("Commons discovery result: %s", result)


def _landmarks_download(
    config: dict[str, Any], paths: dict[str, Path], limit: int | None
) -> None:
    _check_storage(paths, float(config["landmarks"]["storage_budget_gb"]))
    result = download_commons_assets(
        paths["landmark_catalog"],
        paths["landmark_images"],
        config["landmarks"],
        user_agent=config["user_agent"],
        limit=limit,
    )
    LOGGER.info("Commons download result: %s", result)


def _landmarks_index(config: dict[str, Any], paths: dict[str, Path]) -> None:
    output = build_landmark_index(
        paths["landmark_catalog"],
        paths["landmark_gallery"],
        config["landmarks"],
    )
    LOGGER.info("Landmark gallery ready: %s", output)


def _selected_regions(config: dict[str, Any], requested: list[str] | None) -> dict[str, str]:
    available = dict(config.get("osm_regions") or {})
    if not requested:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown OSM regions {unknown}; available: {sorted(available)}")
    return {name: available[name] for name in requested}


def _osm_download(
    config: dict[str, Any],
    paths: dict[str, Path],
    requested: list[str] | None,
) -> dict[str, Path]:
    return download_osm_regions(
        paths["osm_raw"],
        _selected_regions(config, requested),
        user_agent=config["user_agent"],
    )


def _osm_ingest(
    config: dict[str, Any],
    paths: dict[str, Path],
    requested: list[str] | None,
) -> None:
    regions = _selected_regions(config, requested)
    for name in regions:
        ingest_osm_region(
            paths["osm_raw"] / f"{name}.osm.pbf",
            paths["gazetteer"],
            region_name=name,
        )
    if os.name == "nt":
        # Avoid pyosmium's native interpreter-shutdown wait after all handlers,
        # SQLite commits, and metadata writes have completed. OSM is deliberately
        # the final stage in every command that invokes this helper.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _status(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "data_root": str(paths["root"]),
        "geonames": {
            "database_ready": paths["gazetteer"].is_file(),
            "database_bytes": (
                paths["gazetteer"].stat().st_size if paths["gazetteer"].is_file() else 0
            ),
            "sources": {
                path.name: path.stat().st_size
                for path in paths["geonames_raw"].glob("*")
                if path.is_file()
            },
        },
        "ocr": {
            "cache_ready": paths["ocr_models"].is_dir()
            and any(paths["ocr_models"].iterdir())
        },
        "landmarks": {
            "catalog_ready": paths["landmark_catalog"].is_file(),
            "gallery_versions": sorted(
                path.name
                for path in paths["landmark_gallery"].glob("v*")
                if path.is_dir()
            ),
        },
        "osm": {
            "downloads": {
                path.stem: path.stat().st_size
                for path in paths["osm_raw"].glob("*.pbf")
            }
        },
    }
    if paths["landmark_catalog"].is_file():
        result["landmarks"].update(catalog_stats(paths["landmark_catalog"]))
    if paths["gazetteer"].is_file():
        connection = sqlite3.connect(paths["gazetteer"])
        try:
            result["geonames"]["place_count"] = connection.execute(
                "SELECT COUNT(*) FROM places"
            ).fetchone()[0]
            result["geonames"]["name_count"] = connection.execute(
                "SELECT COUNT(*) FROM names"
            ).fetchone()[0]
            result["osm"]["name_count"] = connection.execute(
                "SELECT COUNT(*) FROM osm_names"
            ).fetchone()[0]
        finally:
            connection.close()
    return result


def run(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = _load_config(args.config)
    paths = _paths(config)

    if args.command == "status":
        print(json.dumps(_status(config, paths), indent=2))
        return 0
    if args.command in {"geonames-download", "geonames-all", "all"}:
        _geonames_download(config, paths)
    if args.command in {"geonames-build", "geonames-all", "all"}:
        _geonames_build(paths)
    if args.command in {"ocr-cache", "all"}:
        _ocr_cache(paths)
    if args.command in {"landmarks-plan", "landmarks-all", "all"}:
        _landmarks_plan(config, paths)
    if args.command in {"landmarks-discover", "landmarks-all", "all"}:
        _landmarks_discover(config, paths, args.limit)
    if args.command in {"landmarks-download", "landmarks-all", "all"}:
        _landmarks_download(config, paths, args.limit)
    if args.command in {"landmarks-index", "landmarks-all", "all"}:
        _landmarks_index(config, paths)
    if args.command in {"osm-download", "osm-all", "all"}:
        _osm_download(config, paths, args.region)
    if args.command in {"osm-ingest", "osm-all", "all"}:
        _osm_ingest(config, paths, args.region)
    return 0


if __name__ == "__main__":
    exit_code = run()
    # pyosmium 4.2 can leave its final native cleanup wait parked indefinitely
    # on Windows after apply_file has returned and SQLite has committed. OSM is
    # always the last stage in compound commands, so a flushed process exit is
    # safe and keeps unattended builds from hanging after successful ingestion.
    if os.name == "nt" and sys.argv[1:2] and sys.argv[1] in {
        "osm-ingest",
        "osm-all",
        "all",
    }:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    raise SystemExit(exit_code)
