"""Runtime landmark retrieval against the local Wikimedia Commons gallery."""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .openclip_support import load_model, parse_model_spec, resolve_device


LOGGER = logging.getLogger(__name__)
DEFAULT_GALLERY_ROOT = Path("local-data/landmarks/gallery")
DEFAULT_SEARCH_K = 50


class LandmarkGalleryError(RuntimeError):
    """Raised when the local Commons landmark gallery is unavailable."""


@dataclass(frozen=True)
class LandmarkMatch:
    page_id: int
    geoname_id: int
    landmark_name: str
    lat: float
    lon: float
    country_code: str
    admin1_code: str
    title: str
    page_url: str
    license_short: str
    artist: str
    similarity: float
    phash_distance: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "page_id": self.page_id,
            "geoname_id": self.geoname_id,
            "landmark_name": self.landmark_name,
            "latitude": round(self.lat, 6),
            "longitude": round(self.lon, 6),
            "country_code": self.country_code,
            "admin1_code": self.admin1_code,
            "title": self.title,
            "page_url": self.page_url,
            "license": self.license_short,
            "artist": self.artist,
            "similarity": round(self.similarity, 6),
        }
        if self.phash_distance is not None:
            payload["phash_distance"] = self.phash_distance
        return payload


def latest_gallery(root: str | Path = DEFAULT_GALLERY_ROOT) -> Path:
    directory = Path(root).expanduser().resolve()
    if (directory / "manifest.json").is_file():
        return directory
    versions = [
        path
        for path in directory.glob("v*")
        if path.is_dir() and re.fullmatch(r"v\d+", path.name)
    ]
    if not versions:
        raise LandmarkGalleryError(
            f"No local landmark gallery found under {directory}. Run "
            "'python local-data-pipeline/build_local_data.py landmarks-all'."
        )
    return max(versions, key=lambda path: int(path.name[1:]))


class LandmarkRetriever:
    """Search the Commons gallery and expose match provenance for evidence fusion."""

    def __init__(
        self,
        gallery_path: str | Path = DEFAULT_GALLERY_ROOT,
        *,
        device: str | None = None,
    ) -> None:
        path = latest_gallery(gallery_path)
        try:
            import faiss
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise LandmarkGalleryError(
                "Landmark retrieval requires faiss-cpu and pyarrow"
            ) from exc
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            self.index = faiss.read_index(str(path / "gallery.faiss"))
            table = pq.read_table(path / "metadata.parquet")
        except Exception as exc:
            raise LandmarkGalleryError(f"Could not load landmark gallery {path}: {exc}") from exc
        model_name, pretrained = parse_model_spec(manifest["embedding_model"])
        self.device = resolve_device(device)
        self.model, self.preprocess = load_model(model_name, pretrained, self.device)
        self.path = path
        self.manifest = manifest
        self.rows = table.to_pylist()
        if self.index.ntotal != len(self.rows):
            raise LandmarkGalleryError(
                f"Index has {self.index.ntotal} vectors but metadata has {len(self.rows)} rows"
            )
        self._phashes = [
            int(row["phash"], 16) if row.get("phash") else None for row in self.rows
        ]

    def _encode(self, image: Image.Image) -> np.ndarray:
        import torch

        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            vector = self.model.encode_image(tensor).float()
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector.cpu().numpy().astype(np.float32)

    def _match(self, position: int, similarity: float, distance: int | None = None) -> LandmarkMatch:
        row = self.rows[position]
        return LandmarkMatch(
            page_id=int(row["page_id"]),
            geoname_id=int(row["geoname_id"]),
            landmark_name=str(row["landmark_name"]),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            country_code=str(row.get("country_code") or ""),
            admin1_code=str(row.get("admin1_code") or ""),
            title=str(row.get("title") or ""),
            page_url=str(row.get("page_url") or ""),
            license_short=str(row.get("license_short") or ""),
            artist=str(row.get("artist") or ""),
            similarity=float(similarity),
            phash_distance=distance,
        )

    def perceptual_matches(
        self,
        image: Image.Image,
        *,
        max_distance: int = 6,
        limit: int = 10,
    ) -> list[LandmarkMatch]:
        try:
            import imagehash
        except ImportError:
            return []
        query = int(str(imagehash.phash(image.convert("RGB"))), 16)
        hits: list[tuple[int, int]] = []
        for position, reference in enumerate(self._phashes):
            if reference is None:
                continue
            distance = (query ^ reference).bit_count()
            if distance <= max_distance:
                hits.append((distance, position))
        hits.sort()
        return [
            self._match(position, similarity=1.0, distance=distance)
            for distance, position in hits[:limit]
        ]

    def search(
        self,
        image: Image.Image,
        *,
        top_k: int = DEFAULT_SEARCH_K,
    ) -> list[LandmarkMatch]:
        k = min(max(1, int(top_k)), int(self.index.ntotal))
        similarities, positions = self.index.search(self._encode(image), k)
        return [
            self._match(int(position), float(similarity))
            for similarity, position in zip(similarities[0], positions[0])
            if position >= 0
        ]

    def analyze(self, image: Image.Image, *, top_k: int = DEFAULT_SEARCH_K) -> dict[str, Any]:
        perceptual = self.perceptual_matches(image)
        visual = self.search(image, top_k=top_k)
        groups: dict[int, list[LandmarkMatch]] = {}
        for match in visual:
            groups.setdefault(match.geoname_id, []).append(match)
        ranked_groups = sorted(
            groups.values(),
            key=lambda group: (
                max(item.similarity for item in group),
                len({item.page_id for item in group}),
                sum(item.similarity for item in group),
            ),
            reverse=True,
        )
        return {
            "perceptual_matches": perceptual,
            "visual_matches": visual,
            "groups": ranked_groups,
            "gallery_path": str(self.path),
            "gallery_size": int(self.index.ntotal),
        }

