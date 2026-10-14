"""Decode, orient and size the source photo.

Phone photos are almost always stored sideways with an EXIF orientation tag,
arrive at 12-48 MP, and may come from an untrusted upload. This stage turns any
of that into an upright RGB array at the profile's working resolution, and
reads the one piece of metadata the camera model wants before resizing throws
the EXIF away.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Hard ceiling on decoded pixels. 100 MP covers every phone camera on sale
# (48 MP sensors included) while refusing decompression bombs.
MAX_SOURCE_PIXELS = 100_000_000
MIN_SOURCE_SIDE = 32

_HIGH_BIT_MODES = {"I;16", "I;16B", "I;16L", "I;16N", "I", "F"}

_EXIF_IFD = 0x8769
_FOCAL_35MM = 0xA405

ImageSource = Union[str, os.PathLike, bytes, bytearray, Image.Image]


class ImageInputError(ValueError):
    """The input is not a usable still image."""


@dataclass(frozen=True)
class SourceImage:
    rgb: np.ndarray  # HxWx3 uint8 at working resolution
    source_width: int
    source_height: int
    focal_35mm: float | None
    name: str


def _focal_35mm(image: Image.Image) -> float | None:
    try:
        exif = image.getexif()
    except Exception:
        return None
    value = exif.get(_FOCAL_35MM)
    if value is None:
        try:
            value = exif.get_ifd(_EXIF_IFD).get(_FOCAL_35MM)
        except Exception:
            value = None
    try:
        focal = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    # Zero means "unknown" in EXIF; anything outside this band is corrupt.
    if focal is None or not 8.0 <= focal <= 1200.0:
        return None
    return focal


def _open(source: ImageSource) -> tuple[Image.Image, str]:
    if isinstance(source, Image.Image):
        return source, "image"
    if isinstance(source, (bytes, bytearray)):
        stream: io.BytesIO | str = io.BytesIO(bytes(source))
        name = "upload"
    else:
        path = Path(source)
        if not path.is_file():
            raise ImageInputError(f"Input file not found: {path}")
        stream = str(path)
        name = path.stem
    try:
        image = Image.open(stream)
        width, height = image.size
        if width * height > MAX_SOURCE_PIXELS:
            raise ImageInputError(
                f"Image is {width}x{height}; the limit is {MAX_SOURCE_PIXELS // 1_000_000} MP"
            )
        image.load()
    except ImageInputError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as exc:
        raise ImageInputError(f"That file isn't a readable image ({exc.__class__.__name__})") from exc
    return image, name


def load_source(source: ImageSource, working_long_side: int) -> SourceImage:
    """Return the upright RGB photo, downsized to the working pixel budget.

    The budget is the area of a 4:3 frame whose long side is
    ``working_long_side``; images are never upscaled.
    """
    image, name = _open(source)
    focal = _focal_35mm(image)
    if getattr(image, "n_frames", 1) > 1:
        image.seek(0)
    image = ImageOps.exif_transpose(image)
    if image.mode in _HIGH_BIT_MODES:
        # Pillow's I;16/I/F -> RGB conversion clips at 255 instead of scaling,
        # which turns an ordinary 16-bit PNG nearly white. Stretch to 8 bits.
        values = np.asarray(image, dtype=np.float32)
        low, high = np.percentile(values, [0.5, 99.5])
        scaled = np.clip((values - low) / max(high - low, 1e-6) * 255.0, 0, 255)
        image = Image.fromarray(scaled.astype(np.uint8), mode="L")
    if image.mode != "RGB":
        # Composite transparency over white rather than letting it go black.
        if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
    source_width, source_height = image.size
    if min(source_width, source_height) < MIN_SOURCE_SIDE:
        raise ImageInputError(
            f"Image is {source_width}x{source_height}; at least {MIN_SOURCE_SIDE}px per side is needed"
        )
    # Budget by *area* (that of a 4:3 frame at the working long side), so a
    # panorama keeps a usable height instead of collapsing to a sliver, and
    # cap the long side so textures stay within phone GPU limits.
    budget = working_long_side * working_long_side * 0.75
    scale = min(
        (budget / (source_width * source_height)) ** 0.5,
        2.0 * working_long_side / max(source_width, source_height),
    )
    if scale < 1.0:
        size = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)
    return SourceImage(
        rgb=np.asarray(image, dtype=np.uint8).copy(),
        source_width=source_width,
        source_height=source_height,
        focal_35mm=focal,
        name=name,
    )
