"""Atomic allocation of immutable gallery version directories."""

from __future__ import annotations

import re
from pathlib import Path


VERSION_PATTERN = re.compile(r"^v(\d+)$")


def allocate_version(gallery_root: str | Path) -> tuple[int, Path]:
    """Reserve the next version using mkdir's atomic create semantics."""
    root = Path(gallery_root)
    root.mkdir(parents=True, exist_ok=True)
    versions = [
        int(match.group(1))
        for child in root.iterdir()
        if child.is_dir() and (match := VERSION_PATTERN.match(child.name))
    ]
    candidate = max(versions, default=0) + 1
    while True:
        path = root / f"v{candidate}"
        try:
            path.mkdir()
            return candidate, path
        except FileExistsError:
            candidate += 1

