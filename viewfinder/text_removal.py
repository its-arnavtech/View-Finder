"""Detect and inpaint text regions in an image with minimal visual alteration.

Motivation
----------
Overlaid text -- watermarks, stock-photo captions, camera timestamps, social-media
stickers -- is *not* part of the scene and can bias a geolocation prediction
toward wherever that text style is common. Removing it before inference makes the
model judge the scenery itself.

IMPORTANT CAVEAT -- read before enabling this
---------------------------------------------
**Scene text is one of the strongest geolocation signals that exists.** Street
signs, shop fronts, licence plates and the *script* a language is written in are
exactly how a human GeoGuessr player narrows down a location, and CLIP-based
models exploit the same cue. Removing genuine scene text can therefore make
predictions **worse**, not better.

This module is consequently **opt-in** (CLI flag ``--remove-text``; never on by
default). Use it when an image carries a watermark or burned-in caption; do not
expect an accuracy gain on clean photographs -- on an image with no text it does
nothing at all.

Why EAST and not a hand-rolled heuristic
----------------------------------------
An earlier implementation here used MSER blob detection plus geometric and
photometric heuristics. Measured on a real landscape photo it failed both ways:
it flagged ~48% of a grass hillside as "text" (natural texture produces endless
glyph-shaped, baseline-aligned blobs), while simultaneously **failing to detect
an actual rendered watermark**, because MSER fragments text drawn over busy
foliage into 4-9 pixel pieces rather than clean glyphs. No threshold combination
fixed both -- a sweep of 36 settings found zero that detected real text without
also firing on the clean photo.

This version uses **EAST** (Zhou et al., CVPR 2017), a purpose-built text
detector run through ``cv2.dnn``. Its weights (~96 MB) download once and it then
runs fully offline, matching how the project already handles CLIP weights.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

EAST_URL = (
    "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/"
    "frozen_east_text_detection.pb"
)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "frozen_east_text_detection.pb",
)

# --- Tunables -----------------------------------------------------------------
# Minimum EAST confidence for a region to count as text. 0.5 is the value used in
# the OpenCV reference sample; raising it trades recall for precision.
TEXT_CONFIDENCE = 0.5
# IoU threshold for non-maximum suppression over overlapping detections.
NMS_THRESHOLD = 0.4
# EAST requires input dimensions that are multiples of 32. Larger is more
# sensitive to small text but slower; 640 handles typical photo resolutions.
EAST_INPUT_SIZE = 640
# Detected boxes are grown slightly so anti-aliased glyph edges are covered.
BOX_PADDING_RATIO = 0.06
MASK_DILATION_PX = 3
INPAINT_RADIUS_PX = 3
# Safety valve: if the proposed mask covers more of the image than this, the
# detection is almost certainly wrong, so the edit is abandoned rather than
# mangling the photograph.
MAX_MASK_FRACTION = 0.25

_net_cache: dict[str, Any] = {}


@dataclass
class TextRemovalResult:
    """Outcome of a text-removal pass."""

    image: Image.Image = field(repr=False)
    """The cleaned image (the original object when nothing was removed)."""

    regions_removed: int = 0
    """Number of text regions inpainted."""

    pixels_modified: int = 0
    mask_fraction: float = 0.0
    """Fraction of the image's pixels that were altered (0.0-1.0)."""

    aborted_reason: str | None = None
    """Set when detection ran but the edit was deliberately skipped."""

    @property
    def modified(self) -> bool:
        return self.regions_removed > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "regions_removed": self.regions_removed,
            "pixels_modified": self.pixels_modified,
            "mask_fraction": round(self.mask_fraction, 6),
            "aborted_reason": self.aborted_reason,
        }


def ensure_model(path: str = DEFAULT_MODEL_PATH, download: bool = True) -> str:
    """Return the EAST weights path, downloading them once if necessary.

    Raises:
        FileNotFoundError: If the weights are absent and ``download`` is False.
    """
    if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
        return path
    if not download:
        raise FileNotFoundError(
            f"EAST text-detection weights not found at {path}. "
            f"Download once from {EAST_URL}"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info("Downloading EAST text-detection weights (~96 MB, one time)...")
    urllib.request.urlretrieve(EAST_URL, path)
    logger.info("EAST weights saved to %s", path)
    return path


def _load_net(model_path: str):
    """Load (and process-cache) the EAST network."""
    net = _net_cache.get(model_path)
    if net is None:
        net = cv2.dnn.readNet(model_path)
        _net_cache[model_path] = net
    return net


def detect_text_boxes(
    bgr: np.ndarray,
    model_path: str = DEFAULT_MODEL_PATH,
    confidence: float = TEXT_CONFIDENCE,
) -> list[tuple[int, int, int, int]]:
    """Return axis-aligned bounding boxes of detected text, in image coordinates."""
    height, width = bgr.shape[:2]
    # EAST needs both dimensions to be multiples of 32.
    new_w = max(32, int(round(EAST_INPUT_SIZE / 32)) * 32)
    new_h = new_w
    ratio_w, ratio_h = width / float(new_w), height / float(new_h)

    blob = cv2.dnn.blobFromImage(
        bgr, 1.0, (new_w, new_h), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net = _load_net(model_path)
    net.setInput(blob)
    scores, geometry = net.forward(
        ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    )

    rects: list[tuple[int, int, int, int]] = []
    confidences: list[float] = []
    rows, cols = scores.shape[2:4]
    for y in range(rows):
        scores_row = scores[0, 0, y]
        x0, x1, x2, x3 = (geometry[0, i, y] for i in range(4))
        angles = geometry[0, 4, y]
        for x in range(cols):
            score = float(scores_row[x])
            if score < confidence:
                continue
            # EAST outputs are at 1/4 the network input resolution.
            offset_x, offset_y = x * 4.0, y * 4.0
            angle = float(angles[x])
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            box_h = float(x0[x] + x2[x])
            box_w = float(x1[x] + x3[x])
            end_x = int(offset_x + cos_a * float(x1[x]) + sin_a * float(x2[x]))
            end_y = int(offset_y - sin_a * float(x1[x]) + cos_a * float(x2[x]))
            start_x, start_y = int(end_x - box_w), int(end_y - box_h)
            rects.append((start_x, start_y, int(box_w), int(box_h)))
            confidences.append(score)

    if not rects:
        return []

    keep = cv2.dnn.NMSBoxes(rects, confidences, confidence, NMS_THRESHOLD)
    if keep is None or len(keep) == 0:
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for i in np.array(keep).flatten():
        x, y, w, h = rects[int(i)]
        # Scale back to the original image and pad slightly.
        x = int(x * ratio_w)
        y = int(y * ratio_h)
        w = int(w * ratio_w)
        h = int(h * ratio_h)
        pad_x = int(w * BOX_PADDING_RATIO) + 2
        pad_y = int(h * BOX_PADDING_RATIO) + 2
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(width, x + w + pad_x)
        y1 = min(height, y + h + pad_y)
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


def build_text_mask(
    bgr: np.ndarray, model_path: str = DEFAULT_MODEL_PATH
) -> tuple[np.ndarray, int]:
    """Return (mask, region_count) marking pixels believed to be text."""
    boxes = detect_text_boxes(bgr, model_path)
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    for x, y, w, h in boxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
    if boxes and MASK_DILATION_PX > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (MASK_DILATION_PX * 2 + 1, MASK_DILATION_PX * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)
    return mask, len(boxes)


def remove_text(
    image: Image.Image,
    enabled: bool = True,
    model_path: str = DEFAULT_MODEL_PATH,
    download: bool = True,
) -> TextRemovalResult:
    """Inpaint text regions, leaving everything else bit-identical.

    Args:
        image: RGB PIL image.
        enabled: When False, returns the image unchanged (keeps callers simple).
        model_path: EAST weights location.
        download: Fetch the weights if missing.

    Returns:
        A :class:`TextRemovalResult`. When no text is found the original image
        object is returned and ``regions_removed`` is 0.
    """
    if not enabled:
        return TextRemovalResult(image=image)

    try:
        resolved = ensure_model(model_path, download=download)
    except Exception as exc:  # noqa: BLE001
        logger.warning("text removal unavailable: %s", exc)
        return TextRemovalResult(image=image, aborted_reason=f"model_unavailable:{exc}")

    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    mask, region_count = build_text_mask(bgr, resolved)
    modified_pixels = int(np.count_nonzero(mask))
    total_pixels = mask.size
    fraction = modified_pixels / float(total_pixels) if total_pixels else 0.0

    if region_count == 0 or modified_pixels == 0:
        logger.debug("text removal: no text detected")
        return TextRemovalResult(image=image)

    if fraction > MAX_MASK_FRACTION:
        logger.warning(
            "text removal: proposed mask covers %.1f%% of the image (limit %.0f%%); "
            "skipping as a likely false positive",
            fraction * 100, MAX_MASK_FRACTION * 100,
        )
        return TextRemovalResult(
            image=image, mask_fraction=fraction,
            aborted_reason=f"mask_too_large({fraction:.3f})",
        )

    cleaned_bgr = cv2.inpaint(bgr, mask, INPAINT_RADIUS_PX, cv2.INPAINT_TELEA)
    cleaned = Image.fromarray(cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB))

    logger.info(
        "text removal: inpainted %d text region(s), %.3f%% of pixels modified",
        region_count, fraction * 100,
    )
    return TextRemovalResult(
        image=cleaned,
        regions_removed=region_count,
        pixels_modified=modified_pixels,
        mask_fraction=fraction,
    )
