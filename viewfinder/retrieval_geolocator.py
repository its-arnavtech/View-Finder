"""Image-to-image retrieval geolocation against our own scraped photo gallery.

This is an independent alternative to :mod:`geolocator`, not a wrapper around
it. Nothing here touches GeoCLIP. The two systems answer the same question by
fundamentally different means:

* **GeoCLIP** compares a query image's embedding against *learned location
  embeddings* produced from raw GPS coordinates. No reference photograph is
  involved at any point -- the model has internalised what places look like, and
  it can only ever answer with a coordinate present in its GPS gallery.
* **This module** compares a query image's embedding against the embeddings of
  *other real photographs* whose coordinates we know, collected by our own
  gallery-pipeline from Mapillary. A prediction here means "this looks like that
  specific photo, which was taken here".

The retrieval path is therefore end-to-end ours: our scraped photos, our
embeddings, our index.

Honest limitation -- read before trusting a result
--------------------------------------------------
A retrieval system can only ever propose a location it holds a reference photo
for. With a gallery of a few hundred images covering a handful of regions, a
query from anywhere else cannot possibly be answered correctly: the nearest
neighbour will be whichever gallery photo happens to look most similar, and its
coordinates will be confidently wrong. This is a structural property of
retrieval, not a bug and not something the confidence score can detect -- the
score measures agreement among the returned neighbours, and neighbours drawn
from a tiny gallery can agree strongly while all being wrong.

Treat results as experimental until the gallery covers the areas being queried
at a meaningful density. :data:`EXPERIMENTAL_GALLERY_THRESHOLD` controls the
warning emitted for small galleries.

Embedding-space compatibility
-----------------------------
Retrieval is only meaningful if the query is embedded by exactly the model that
embedded the gallery. Rather than hardcoding a model name that could silently
drift from the pipeline's, the model and checkpoint are read from the gallery's
own ``manifest.json`` (written by gallery-pipeline's load stage as
``embedding_model``). A gallery built with a different encoder therefore selects
that encoder automatically, and an unreadable or non-OpenCLIP spec is a hard
error rather than a silently meaningless search.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from .aggregator import Candidate, FramePrediction

logger = logging.getLogger(__name__)

# Galleries smaller than this cannot cover enough of the world for a retrieval
# answer to be trustworthy outside the regions they were scraped from.
EXPERIMENTAL_GALLERY_THRESHOLD = 1000

# Default neighbours retrieved per query image.
DEFAULT_TOP_K = 10

# Query images encoded per forward pass on CPU.
DEFAULT_BATCH_SIZE = 8

# Files written by gallery-pipeline's load stage.
INDEX_FILENAME = "gallery.faiss"
METADATA_FILENAME = "metadata.parquet"
MANIFEST_FILENAME = "manifest.json"

# Prefix used by the pipeline's `EmbeddingGenerator.full_model_name`.
_OPEN_CLIP_PREFIX = "open_clip"


class GalleryError(RuntimeError):
    """Raised when a gallery directory is missing, malformed, or inconsistent."""


@dataclass
class GalleryInfo:
    """Summary of a loaded gallery, for reporting and for warning the user."""

    path: str
    version: int | None
    image_count: int
    dimension: int
    model_name: str
    pretrained: str
    tiles_covered: int | None = None
    regions: dict[str, int] = field(default_factory=dict)

    @property
    def model_spec(self) -> str:
        return f"{_OPEN_CLIP_PREFIX}:{self.model_name}:{self.pretrained}"

    @property
    def is_experimental(self) -> bool:
        """True when the gallery is too small for results to be relied upon."""
        return self.image_count < EXPERIMENTAL_GALLERY_THRESHOLD

    def warning(self) -> str | None:
        """Return a user-facing caveat, or ``None`` if the gallery is large enough."""
        if not self.is_experimental:
            return None
        regions = ", ".join(sorted(self.regions)) if self.regions else "a few areas"
        return (
            f"EXPERIMENTAL: the retrieval gallery holds only {self.image_count} reference "
            f"photos (below {EXPERIMENTAL_GALLERY_THRESHOLD}), covering {regions}. "
            "This predictor can only propose locations it has reference photos for, so a "
            "query from anywhere else will be confidently wrong. Treat these results as "
            "experimental until the gallery is scaled up."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "version": self.version,
            "image_count": self.image_count,
            "embedding_dimension": self.dimension,
            "embedding_model": self.model_spec,
            "tiles_covered": self.tiles_covered,
            "geographic_coverage": self.regions,
            "experimental": self.is_experimental,
        }


@dataclass
class _LoadedGallery:
    """A FAISS index plus the coordinate metadata for each of its positions."""

    info: GalleryInfo
    index: Any
    latitudes: np.ndarray
    longitudes: np.ndarray
    image_ids: list[str]


# Process-wide caches. Loading a FAISS index and an OpenCLIP checkpoint are both
# expensive, and a video run would otherwise repeat them per call. Same
# double-checked-locking discipline as reverse_geocode.py.
_gallery_cache: dict[str, _LoadedGallery] = {}
_gallery_lock = threading.Lock()

_model_cache: dict[tuple[str, str, str], tuple[Any, Any]] = {}
_model_lock = threading.Lock()


def _parse_model_spec(spec: str) -> tuple[str, str]:
    """Split ``open_clip:<model>:<checkpoint>`` into its parts.

    Raises:
        GalleryError: If the spec is empty, malformed, or not an OpenCLIP model.
    """
    parts = str(spec).split(":")
    if len(parts) != 3 or not all(parts):
        raise GalleryError(
            f"Gallery manifest has an unparseable embedding_model {spec!r}; "
            f"expected '{_OPEN_CLIP_PREFIX}:<model>:<checkpoint>'"
        )
    framework, model_name, pretrained = parts
    if framework != _OPEN_CLIP_PREFIX:
        raise GalleryError(
            f"Gallery was built with {framework!r}, but this predictor can only encode "
            f"queries with OpenCLIP. Retrieval against a different embedding space would "
            f"return meaningless matches."
        )
    return model_name, pretrained


def load_gallery(gallery_path: str | Path) -> _LoadedGallery:
    """Load (and cache) a gallery version directory's index and metadata.

    Args:
        gallery_path: A gallery version directory, e.g. ``.../gallery/v1``.

    Returns:
        The cached :class:`_LoadedGallery`.

    Raises:
        GalleryError: If required files are missing or the index, metadata and
            manifest disagree about the gallery's size or dimensionality.
    """
    resolved = str(Path(gallery_path).expanduser().resolve())

    cached = _gallery_cache.get(resolved)
    if cached is not None:
        return cached

    with _gallery_lock:
        cached = _gallery_cache.get(resolved)
        if cached is not None:
            return cached

        directory = Path(resolved)
        if not directory.is_dir():
            raise GalleryError(f"Gallery path is not a directory: {directory}")

        index_path = directory / INDEX_FILENAME
        metadata_path = directory / METADATA_FILENAME
        manifest_path = directory / MANIFEST_FILENAME
        for required in (index_path, metadata_path, manifest_path):
            if not required.is_file():
                raise GalleryError(
                    f"Gallery is missing {required.name}. Expected a gallery version "
                    f"directory produced by gallery-pipeline's load stage, e.g. "
                    f".../data/<run>/gallery/v1"
                )

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GalleryError(f"Could not read {manifest_path}: {exc}") from exc

        model_name, pretrained = _parse_model_spec(manifest.get("embedding_model", ""))

        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise GalleryError(
                "Retrieval requires faiss-cpu (pip install faiss-cpu)"
            ) from exc
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise GalleryError(
                "Reading gallery metadata requires pyarrow (pip install pyarrow)"
            ) from exc

        index = faiss.read_index(str(index_path))
        table = pq.read_table(metadata_path)

        columns = {name.lower(): name for name in table.column_names}
        for required_column in ("lat", "lon", "image_id"):
            if required_column not in columns:
                raise GalleryError(
                    f"Gallery metadata lacks a {required_column!r} column; found "
                    f"{table.column_names}"
                )

        # Metadata rows are aligned to index positions by index_position when the
        # column exists, rather than trusting file order.
        if "index_position" in columns:
            order = np.argsort(np.asarray(table[columns["index_position"]].to_pylist()))
        else:
            order = np.arange(table.num_rows)

        latitudes = np.asarray(table[columns["lat"]].to_pylist(), dtype=np.float64)[order]
        longitudes = np.asarray(table[columns["lon"]].to_pylist(), dtype=np.float64)[order]
        image_ids = [str(table[columns["image_id"]].to_pylist()[i]) for i in order]

        if index.ntotal != len(latitudes):
            raise GalleryError(
                f"Gallery is inconsistent: FAISS index holds {index.ntotal} vectors but "
                f"metadata has {len(latitudes)} rows"
            )
        manifest_count = int(manifest.get("total_image_count", index.ntotal))
        if manifest_count != index.ntotal:
            raise GalleryError(
                f"Gallery is inconsistent: manifest claims {manifest_count} images but the "
                f"index holds {index.ntotal}"
            )

        info = GalleryInfo(
            path=resolved,
            version=manifest.get("version"),
            image_count=int(index.ntotal),
            dimension=int(index.d),
            model_name=model_name,
            pretrained=pretrained,
            tiles_covered=manifest.get("tiles_covered"),
            regions=dict(manifest.get("geographic_coverage") or {}),
        )

        loaded = _LoadedGallery(
            info=info,
            index=index,
            latitudes=latitudes,
            longitudes=longitudes,
            image_ids=image_ids,
        )
        _gallery_cache[resolved] = loaded
        logger.info(
            "Loaded retrieval gallery v%s: %d photos, dim=%d, model=%s",
            info.version, info.image_count, info.dimension, info.model_spec,
        )
        return loaded


def _load_model(model_name: str, pretrained: str, device: str) -> tuple[Any, Any]:
    """Load (and cache) the OpenCLIP model and preprocessing transform."""
    key = (model_name, pretrained, device)
    cached = _model_cache.get(key)
    if cached is not None:
        return cached

    with _model_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            return cached

        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise GalleryError(
                "Retrieval requires open-clip-torch (pip install open-clip-torch)"
            ) from exc

        logger.info("Loading OpenCLIP %s/%s on %s", model_name, pretrained, device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        _model_cache[key] = (model, preprocess)
        return model, preprocess


def resolve_device(requested: str | None = None) -> str:
    """Pick the torch device to run on."""
    import torch

    if requested and requested.lower() != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RetrievalGeolocator:
    """Geolocate by finding the most visually similar photos of known location."""

    def __init__(self, gallery_path: str | Path, device: str | None = None) -> None:
        """Load the gallery and the matching OpenCLIP encoder.

        Both are cached process-wide, so constructing several instances against
        the same gallery is cheap.

        Args:
            gallery_path: A gallery version directory, e.g. ``.../gallery/v1``.
            device: Torch device, or ``None``/``"auto"`` to detect one.

        Raises:
            GalleryError: If the gallery cannot be loaded or its encoder is not
                available.
        """
        self.gallery = load_gallery(gallery_path)
        self.info = self.gallery.info
        self.device = resolve_device(device)
        self.model, self.preprocess = _load_model(
            self.info.model_name, self.info.pretrained, self.device
        )

        warning = self.info.warning()
        if warning:
            logger.warning("%s", warning)

    # -- encoding ---------------------------------------------------------

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Embed query images into the gallery's vector space.

        Vectors are L2-normalised so that the index's inner product is exactly
        cosine similarity -- matching how gallery-pipeline normalised the
        gallery vectors before adding them to its ``IndexFlatIP``.

        Args:
            images: RGB PIL images.

        Returns:
            Array of shape ``(len(images), dimension)``, dtype float32.
        """
        import torch

        if not images:
            return np.empty((0, self.info.dimension), dtype=np.float32)

        tensors = [self.preprocess(image.convert("RGB")) for image in images]
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(batch).float()
            features = features / features.norm(dim=-1, keepdim=True)

        vectors = features.cpu().numpy().astype(np.float32)
        if vectors.shape[1] != self.info.dimension:
            raise GalleryError(
                f"Query embeddings have dimension {vectors.shape[1]} but the gallery index "
                f"expects {self.info.dimension}"
            )
        return vectors

    # -- retrieval --------------------------------------------------------

    def search(self, vectors: np.ndarray, top_k: int = DEFAULT_TOP_K) -> list[list[Candidate]]:
        """Run one batched nearest-neighbour search for many query vectors.

        FAISS searches a whole query matrix in a single call, so a video's
        frames are retrieved together rather than one call per frame.

        Args:
            vectors: Query matrix of shape ``(n, dimension)``, L2-normalised.
            top_k: Neighbours to return per query. Clamped to the gallery size.

        Returns:
            One ranked candidate list per query row.
        """
        if len(vectors) == 0:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # Asking for more neighbours than the gallery holds makes FAISS pad the
        # result with position -1, which must not be treated as a match.
        k = min(top_k, self.info.image_count)
        similarities, positions = self.gallery.index.search(
            np.ascontiguousarray(vectors, dtype=np.float32), k
        )

        results: list[list[Candidate]] = []
        for row_similarities, row_positions in zip(similarities, positions):
            candidates: list[Candidate] = []
            for similarity, position in zip(row_similarities, row_positions):
                if position < 0:
                    continue
                candidates.append(
                    Candidate(
                        lat=float(self.gallery.latitudes[position]),
                        lon=float(self.gallery.longitudes[position]),
                        score=float(similarity),
                        source_image_id=self.gallery.image_ids[position],
                    )
                )
            results.append(candidates)
        return results

    def predict_batch(
        self, images: Sequence[Image.Image], top_k: int = DEFAULT_TOP_K
    ) -> list[list[Candidate]]:
        """Encode and retrieve for a batch of images."""
        return self.search(self.encode(images), top_k=top_k)

    def predict_image(self, image: Image.Image, top_k: int = DEFAULT_TOP_K) -> list[Candidate]:
        """Encode and retrieve for a single image."""
        return self.predict_batch([image], top_k=top_k)[0]

    def predict_frames(
        self,
        frames: Sequence[Any],
        top_k: int = DEFAULT_TOP_K,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress: bool = False,
    ) -> list[FramePrediction]:
        """Encode video frames in batches and retrieve for all of them at once.

        Encoding is chunked to bound peak memory, but the retrieval itself runs
        as a single batched FAISS search over every frame's vector.

        Args:
            frames: Objects exposing ``image``, ``index``, ``timestamp`` and
                ``blur_score`` -- i.e. :class:`video_processor.Frame`.
            top_k: Neighbours to retain per frame.
            batch_size: Frames encoded per forward pass.
            progress: Log progress after each encoded batch.

        Returns:
            One :class:`FramePrediction` per frame, in input order.
        """
        if not frames:
            return []

        chunks: list[np.ndarray] = []
        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            chunks.append(self.encode([frame.image for frame in chunk]))
            if progress:
                logger.info(
                    "Encoded %d/%d frames", min(start + batch_size, len(frames)), len(frames)
                )

        vectors = np.vstack(chunks)
        retrieved = self.search(vectors, top_k=top_k)

        return [
            FramePrediction(
                candidates=candidates,
                frame_index=frame.index,
                timestamp=frame.timestamp,
                blur_score=frame.blur_score,
            )
            for frame, candidates in zip(frames, retrieved)
        ]


def matched_image_ids(candidates: Iterable[Candidate]) -> list[str]:
    """Return the distinct gallery photos a set of candidates matched against."""
    seen: dict[str, None] = {}
    for candidate in candidates:
        if candidate.source_image_id is not None:
            seen.setdefault(candidate.source_image_id, None)
    return list(seen)
