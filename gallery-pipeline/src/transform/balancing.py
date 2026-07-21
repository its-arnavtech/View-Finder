"""Deterministic per-tile geographic balancing."""

from __future__ import annotations

from typing import Any


def cap_tile(records: list[dict[str, Any]], maximum: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if maximum < 1:
        raise ValueError("maximum images per tile must be positive")
    # Prefer sharper images; image_id is a stable tiebreaker for repeatable builds.
    ordered = sorted(records, key=lambda item: (-float(item.get("blur_score", 0.0)), item["image_id"]))
    return ordered[:maximum], ordered[maximum:]

