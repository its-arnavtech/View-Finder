"""Pillow-based image integrity checks."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError


def validate_image(path: str | Path) -> tuple[bool, str | None]:
    """Fully decode an image; ``verify`` alone does not force pixel decompression."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            if image.width < 32 or image.height < 32:
                return False, "image_too_small"
        return True, None
    except (OSError, ValueError, UnidentifiedImageError):
        return False, "corrupt_or_unreadable"

