"""Batched CLIP embeddings with GPU acceleration and CPU fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate normalized OpenCLIP image vectors and an indoor probability.

    OpenCLIP was chosen over Transformers because its preprocessing, model registry,
    and text/image encoders are exposed through a compact purpose-built API.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        batch_size_gpu: int = 64,
        batch_size_cpu: int = 16,
        indoor_probability_threshold: float = 0.65,
    ):
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError("Embedding generation requires torch and open-clip-torch") from exc
        self.torch = torch
        self.open_clip = open_clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size_gpu if self.device == "cuda" else batch_size_cpu
        self.model_name = model_name
        self.pretrained = pretrained
        self.indoor_probability_threshold = indoor_probability_threshold
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        labels = tokenizer([
            "an indoor photograph of a room or building interior",
            "an outdoor street-level or landscape photograph",
        ]).to(self.device)
        with torch.inference_mode():
            text_features = self.model.encode_text(labels)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        LOGGER.info("Loaded OpenCLIP model=%s pretrained=%s device=%s batch_size=%d", model_name, pretrained, self.device, self.batch_size)

    @property
    def full_model_name(self) -> str:
        return f"open_clip:{self.model_name}:{self.pretrained}"

    def generate(self, image_paths: Sequence[str | Path]) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized image embeddings and indoor probabilities in input order."""
        if not image_paths:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
        all_embeddings: list[np.ndarray] = []
        all_indoor: list[np.ndarray] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    tensors.append(self.preprocess(image.convert("RGB")))
            batch = self.torch.stack(tensors).to(self.device)
            with self.torch.inference_mode():
                context = self.torch.autocast(device_type="cuda") if self.device == "cuda" else _NullContext()
                with context:
                    image_features = self.model.encode_image(batch)
                image_features = image_features.float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                probabilities = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)[:, 0]
            all_embeddings.append(image_features.cpu().numpy().astype(np.float32))
            all_indoor.append(probabilities.cpu().numpy().astype(np.float32))
            LOGGER.info("Embedded batch start=%d size=%d", start, len(batch_paths))
        return np.concatenate(all_embeddings), np.concatenate(all_indoor)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

