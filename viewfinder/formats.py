"""Canonical input-format definitions shared by the CLI and the pipeline modules.

This is the single source of truth for which files View Finder accepts. It exists
because the definition was previously duplicated in three places and drifted: the
CLI accepted only ``.jpg/.jpeg/.png`` and rejected files the image loader could
happily open (it already handled seven formats), so valid photos were refused at
the argument-validation step before anything tried to read them.

Image extensions are derived from **Pillow's own registry** at import time rather
than hardcoded, so the accepted set automatically tracks what the installed
Pillow can actually decode. Installing an optional plugin (e.g. ``pillow-heif``
for iPhone ``.heic`` photos) widens support with no change here.

The module intentionally imports nothing heavier than Pillow, so the CLI can
validate arguments without paying for torch/OpenCV, and it degrades to a static
list if Pillow is unavailable so argument parsing still produces a clean error.
"""

from __future__ import annotations

# Formats Pillow can technically open but that must NOT be treated as stills.
#
# * Video/animation containers: decoding these as a single image would silently
#   analyse one frame of a clip. They belong to the video path instead.
# * Scientific/raster data and vector/document formats: not photographs, and
#   several need external tooling (Ghostscript) that we do not require.
_NON_PHOTO_EXTENSIONS = {
    # video / animation containers
    ".mpeg", ".mpg", ".fli", ".flc",
    # scientific data
    ".grib", ".h5", ".hdf", ".fits", ".fit", ".bufr",
    # vector / document (need Ghostscript or platform metafile support)
    ".eps", ".ps", ".wmf", ".emf",
}

# Used only if Pillow cannot be imported, so the CLI still reports something
# sensible instead of crashing during argument validation.
_FALLBACK_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".dib", ".gif",
     ".webp", ".tif", ".tiff", ".avif", ".heic", ".heif", ".ico", ".tga",
     ".ppm", ".pgm", ".pbm", ".pnm", ".jp2", ".j2k", ".psd", ".qoi"}
)

# Containers OpenCV is commonly built to decode. Kept explicit (rather than
# probed) because OpenCV exposes no reliable registry of supported containers.
VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg",
     ".wmv", ".flv", ".3gp", ".m2ts", ".mts", ".ts", ".ogv"}
)


def _discover_image_extensions() -> frozenset[str]:
    """Return every extension the installed Pillow can open, minus non-photos."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - depends on environment
        return _FALLBACK_IMAGE_EXTENSIONS

    try:
        Image.init()  # populate Image.EXTENSION with all registered plugins
        discovered = {
            ext.lower()
            for ext, fmt in Image.EXTENSION.items()
            if fmt in Image.OPEN
        }
    except Exception:  # pragma: no cover - defensive: never break arg parsing
        return _FALLBACK_IMAGE_EXTENSIONS

    usable = (discovered | _FALLBACK_IMAGE_EXTENSIONS) - _NON_PHOTO_EXTENSIONS
    return frozenset(usable)


IMAGE_EXTENSIONS = _discover_image_extensions()


def is_image(path: str) -> bool:
    """True if ``path`` has an extension the image loader can decode."""
    import os

    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def is_video(path: str) -> bool:
    """True if ``path`` has a recognised video container extension."""
    import os

    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def describe_supported() -> str:
    """Human-readable summary of accepted formats, for CLI help and errors."""
    images = ", ".join(sorted(IMAGE_EXTENSIONS))
    videos = ", ".join(sorted(VIDEO_EXTENSIONS))
    return f"images: {images}\nvideos: {videos}"
