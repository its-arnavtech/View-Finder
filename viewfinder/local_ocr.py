"""Local PaddleOCR adapter for scene text used as geographic evidence."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_CACHE = Path("local-data/ocr-models")
MIN_OCR_SCORE = 0.55


class LocalOcrError(RuntimeError):
    """Raised when the local OCR runtime or its cached models are unavailable."""


@dataclass(frozen=True)
class OcrLine:
    text: str
    score: float
    polygon: list[list[float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "polygon": self.polygon,
        }


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _polygon(value: Any) -> list[list[float]]:
    try:
        return [[round(float(x), 2), round(float(y), 2)] for x, y in value]
    except (TypeError, ValueError):
        return []


def _new_result_lines(result: Any) -> list[OcrLine]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        try:
            payload = dict(payload)
        except (TypeError, ValueError):
            return []
    body = payload.get("res", payload)
    texts = body.get("rec_texts") or body.get("texts") or []
    scores = body.get("rec_scores") or body.get("scores") or []
    polygons = body.get("rec_polys") or body.get("dt_polys") or []
    lines: list[OcrLine] = []
    for index, text in enumerate(texts):
        cleaned = _clean_text(str(text))
        score = float(scores[index]) if index < len(scores) else 0.0
        poly = _polygon(polygons[index]) if index < len(polygons) else []
        if cleaned and score >= MIN_OCR_SCORE:
            lines.append(OcrLine(cleaned, score, poly))
    return lines


def _old_result_lines(results: Any) -> list[OcrLine]:
    lines: list[OcrLine] = []
    if not isinstance(results, (list, tuple)):
        return lines
    flattened: Iterable[Any] = results
    if len(results) == 1 and isinstance(results[0], list):
        flattened = results[0]
    for item in flattened:
        try:
            box, recognition = item
            text, score = recognition
            cleaned = _clean_text(str(text))
            numeric_score = float(score)
        except (TypeError, ValueError):
            continue
        if cleaned and numeric_score >= MIN_OCR_SCORE:
            lines.append(OcrLine(cleaned, numeric_score, _polygon(box)))
    return lines


class PaddleSceneText:
    """Run PP-OCR locally; model downloads are confined to the configured cache."""

    def __init__(
        self,
        model_cache_dir: str | Path = DEFAULT_MODEL_CACHE,
        *,
        device: str | None = None,
        engine: Any | None = None,
    ) -> None:
        self.model_cache_dir = Path(model_cache_dir).expanduser().resolve()
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(self.model_cache_dir))
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        # Paddle 3.3.1's Windows oneDNN path fails on PP-OCRv6 array
        # attributes (ConvertPirAttribute2RuntimeAttribute). The plain Paddle
        # CPU backend is slower but correct and portable.
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
        if engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise LocalOcrError(
                    "Local OCR requires paddleocr and paddlepaddle. Install "
                    "viewfinder/requirements-local-evidence.txt."
                ) from exc
            selected_device = device or "cpu"
            kwargs = {
                "device": selected_device,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
            }
            try:
                engine = PaddleOCR(**kwargs)
            except TypeError:
                # Compatibility with the pre-3.x constructor.
                engine = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
            except Exception as exc:
                raise LocalOcrError(
                    f"Could not load cached PaddleOCR models from "
                    f"{self.model_cache_dir}: {exc}"
                ) from exc
        self.engine = engine

    def analyze(self, image: Image.Image) -> list[OcrLine]:
        pixels = np.asarray(image.convert("RGB"))
        try:
            if hasattr(self.engine, "predict"):
                raw = self.engine.predict(pixels)
                lines: list[OcrLine] = []
                for result in raw:
                    lines.extend(_new_result_lines(result))
            else:
                raw = self.engine.ocr(pixels, cls=True)
                lines = _old_result_lines(raw)
        except Exception as exc:
            raise LocalOcrError(f"PaddleOCR inference failed: {exc}") from exc
        unique: dict[str, OcrLine] = {}
        for line in lines:
            key = line.text.casefold()
            if key not in unique or line.score > unique[key].score:
                unique[key] = line
        return sorted(unique.values(), key=lambda line: line.score, reverse=True)
