"""Frame extraction and quality filtering for video inputs.

Videos are sampled at a fixed wall-clock rate (default 1 fps) rather than
frame-by-frame: consecutive frames of a video are almost always redundant for
geolocation, and GeoCLIP inference is the expensive part of the pipeline.

Two cheap filters are applied to the sampled frames before they ever reach the
model:

* **Blur rejection** -- the variance of the Laplacian of the grayscale frame is
  a standard, dependency-free sharpness proxy. A frame that is out of focus or
  motion-blurred has little high-frequency content, so the second derivative is
  near zero everywhere and its variance collapses. Frames below
  ``blur_threshold`` are dropped.
* **Near-duplicate rejection** -- each kept frame is reduced to a 32x32
  grayscale "signature". If the mean absolute difference between a new frame's
  signature and the last kept frame's signature is below ``dedup_threshold``
  (in 0-255 intensity units), the new frame is considered the same view and is
  dropped. This matters for static/tripod shots, where 1 fps sampling would
  otherwise feed the model dozens of copies of one image and produce a falsely
  confident cross-frame agreement score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Extensions routed to this module by the CLI.
# Re-exported from formats.py, the single source of truth for accepted inputs.
from .formats import VIDEO_EXTENSIONS  # noqa: F401

# Defaults. These are deliberately permissive: it is better to hand the model a
# mediocre frame than to hand it nothing at all.
DEFAULT_BLUR_THRESHOLD = 60.0
DEFAULT_DEDUP_THRESHOLD = 6.0

# Size of the downscaled grayscale signature used for near-duplicate detection.
_SIGNATURE_SIZE = (32, 32)

# If every sampled frame fails the blur test, fall back to the sharpest frames
# seen rather than aborting. This many are retained.
_BLUR_FALLBACK_COUNT = 5


@dataclass
class Frame:
    """A single sampled video frame that survived filtering."""

    index: int
    """Zero-based frame index within the source video."""

    timestamp: float
    """Position of the frame in the video, in seconds."""

    image: Image.Image = field(repr=False)
    """The frame as an RGB PIL image, ready for the model's preprocessor."""

    blur_score: float
    """Variance of the Laplacian. Higher is sharper."""


@dataclass
class ExtractionStats:
    """Bookkeeping about what the sampler saw and why frames were discarded."""

    video_fps: float = 0.0
    duration_seconds: float = 0.0
    total_frames: int = 0
    sampled: int = 0
    kept: int = 0
    skipped_blurry: int = 0
    skipped_duplicate: int = 0
    blur_fallback_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_fps": round(self.video_fps, 3),
            "duration_seconds": round(self.duration_seconds, 2),
            "total_frames": self.total_frames,
            "frames_sampled": self.sampled,
            "frames_kept": self.kept,
            "frames_skipped_blurry": self.skipped_blurry,
            "frames_skipped_duplicate": self.skipped_duplicate,
            "blur_fallback_used": self.blur_fallback_used,
        }


def is_video(path: str) -> bool:
    """Return True if ``path`` has a recognised video extension."""
    import os

    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def blur_score(gray: np.ndarray) -> float:
    """Return the variance of the Laplacian of a grayscale frame.

    Higher values mean more high-frequency detail, i.e. a sharper frame.
    The absolute scale depends on resolution and content, so the threshold is
    a tunable rather than a universal constant.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _signature(gray: np.ndarray) -> np.ndarray:
    """Reduce a grayscale frame to a small float array for duplicate checks."""
    return cv2.resize(gray, _SIGNATURE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def _to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR frame to an RGB PIL image."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def extract_frames(
    video_path: str,
    fps: float = 1.0,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    max_frames: int | None = None,
) -> tuple[list[Frame], ExtractionStats]:
    """Sample, filter and return frames from a video.

    Args:
        video_path: Path to the video file.
        fps: Target sampling rate in frames per second of video time. Values
            above the video's native rate are clamped to every frame.
        blur_threshold: Minimum Laplacian variance for a frame to be kept.
            Pass ``0`` to disable blur filtering.
        dedup_threshold: Minimum mean absolute signature difference (0-255)
            between consecutive kept frames. Pass ``0`` to disable dedup.
        max_frames: Optional cap on the number of frames returned. Sampling
            stops once the cap is reached.

    Returns:
        A ``(frames, stats)`` tuple. ``frames`` is ordered by timestamp.

    Raises:
        RuntimeError: If the video cannot be opened or contains no decodable
            frames.
    """
    capture = cv2.VideoCapture(video_path)
    # Ask OpenCV to honour rotation metadata where the backend supports it;
    # phone videos are frequently stored rotated, and feeding the model a
    # sideways frame measurably degrades predictions.
    try:
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except cv2.error:  # pragma: no cover - backend dependent
        pass

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        return _sample(capture, fps, blur_threshold, dedup_threshold, max_frames)
    finally:
        capture.release()


def _sample(
    capture: cv2.VideoCapture,
    fps: float,
    blur_threshold: float,
    dedup_threshold: float,
    max_frames: int | None,
) -> tuple[list[Frame], ExtractionStats]:
    """Drive the decode loop. Split out so the caller can own capture cleanup."""
    native_fps = capture.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0 or np.isnan(native_fps):
        # Some containers report no frame rate; 30 is a safe assumption and only
        # affects the sampling stride and reported timestamps.
        logger.warning("Video reports no frame rate, assuming 30 fps for sampling")
        native_fps = 30.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stats = ExtractionStats(
        video_fps=native_fps,
        total_frames=total_frames,
        duration_seconds=total_frames / native_fps if total_frames else 0.0,
    )

    # One frame is retained every `stride` decoded frames.
    stride = max(1, int(round(native_fps / fps))) if fps > 0 else 1

    frames: list[Frame] = []
    # Sharpest rejected frames, kept only in case *everything* fails the blur
    # test. Bounded to _BLUR_FALLBACK_COUNT entries so memory stays flat.
    fallback: list[Frame] = []
    last_signature: np.ndarray | None = None
    index = -1

    while True:
        # grab() decodes without colour-converting, so skipped frames are cheap.
        if not capture.grab():
            break
        index += 1
        if index % stride:
            continue

        ok, bgr = capture.retrieve()
        if not ok or bgr is None:
            continue

        stats.sampled += 1
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sharpness = blur_score(gray)
        timestamp = index / native_fps

        if blur_threshold > 0 and sharpness < blur_threshold:
            stats.skipped_blurry += 1
            _remember_fallback(fallback, Frame(index, timestamp, _to_pil(bgr), sharpness))
            continue

        signature = _signature(gray)
        if (
            dedup_threshold > 0
            and last_signature is not None
            and float(np.mean(np.abs(signature - last_signature))) < dedup_threshold
        ):
            stats.skipped_duplicate += 1
            continue

        last_signature = signature
        frames.append(Frame(index, timestamp, _to_pil(bgr), sharpness))
        if max_frames is not None and len(frames) >= max_frames:
            break

    if stats.sampled == 0:
        raise RuntimeError("No decodable frames found in video")

    if not frames and fallback:
        # Every sampled frame was blurry. A low-confidence answer from soft
        # frames beats no answer, so use the sharpest ones and let the
        # confidence score reflect the poor input.
        logger.warning(
            "All %d sampled frames fell below the blur threshold (%.1f); "
            "falling back to the %d sharpest",
            stats.sampled,
            blur_threshold,
            len(fallback),
        )
        frames = sorted(fallback, key=lambda f: f.index)
        stats.skipped_blurry -= len(frames)
        stats.blur_fallback_used = True

    stats.kept = len(frames)
    if stats.duration_seconds == 0.0 and frames:
        stats.duration_seconds = frames[-1].timestamp

    return frames, stats


def _remember_fallback(fallback: list[Frame], frame: Frame) -> None:
    """Keep the ``_BLUR_FALLBACK_COUNT`` sharpest rejected frames seen so far."""
    fallback.append(frame)
    if len(fallback) > _BLUR_FALLBACK_COUNT:
        fallback.sort(key=lambda f: f.blur_score, reverse=True)
        fallback.pop()


def load_image(image_path: str) -> Image.Image:
    """Load a still image as RGB, applying any EXIF orientation tag.

    Phone photos are commonly stored unrotated with an orientation tag; ignoring
    it hands the model a sideways image.
    """
    from PIL import ImageOps

    with Image.open(image_path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")
