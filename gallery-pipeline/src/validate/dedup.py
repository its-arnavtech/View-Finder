"""Perceptual hash generation and near-duplicate matching."""

from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image


def perceptual_hash(path: str | Path) -> imagehash.ImageHash:
    with Image.open(path) as image:
        return imagehash.phash(image.convert("RGB"))


def find_near_duplicate(
    candidate: imagehash.ImageHash,
    accepted: list[tuple[str, imagehash.ImageHash]],
    maximum_distance: int,
) -> str | None:
    for image_id, existing in accepted:
        if candidate - existing <= maximum_distance:
            return image_id
    return None

