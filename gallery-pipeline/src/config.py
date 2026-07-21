"""Configuration loading and project-relative path resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and make runtime paths independent of the current directory."""
    configured = path or os.getenv("GALLERY_CONFIG", "config/tiling_config.yaml")
    config_path = Path(configured)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    data_dir = Path(config.get("paths", {}).get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    config.setdefault("paths", {})["data_dir"] = str(data_dir.resolve())
    config["paths"]["catalog_path"] = str((data_dir / "catalog.db").resolve())
    config["paths"]["raw_dir"] = str((data_dir / "raw").resolve())
    config["paths"]["transformed_dir"] = str((data_dir / "transformed").resolve())
    config["paths"]["gallery_dir"] = str((data_dir / "gallery").resolve())
    return config


def ensure_data_directories(config: dict[str, Any]) -> None:
    for key in ("data_dir", "raw_dir", "transformed_dir", "gallery_dir"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)

