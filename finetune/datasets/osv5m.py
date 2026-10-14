"""OpenStreetView-5M (OSV-5M): 5.1M street-view images, 225 countries.

Status: **CLEAR.** Verified 2026-08-09 from primary sources.

Licence
-------
* The Hugging Face dataset card for ``osv5m/osv5m`` declares
  **CC-BY-SA-4.0**, confirmed structurally through the Hub API rather than read
  off a rendered page: both ``cardData.license`` and the repo's tag list return
  ``cc-by-sa-4.0``. Attribution plus share-alike, with **no NonCommercial and
  no NoDerivatives term** -- so unlike YFCC100M's mixed licences or GSV-Cities'
  CC BY-NC-ND, a checkpoint trained on it can be redistributed.
* The repo is **not gated** (``gated: False``, ``private: False``): no terms
  acceptance, no account, no manual approval. This is the difference from MSLS,
  whose edition terms could not even be read, and from YFCC100M, which needs an
  AWS or Yahoo account.
* Attribution/share-alike obligations still apply to anything redistributed.
  Cite Astruc et al., "OpenStreetView-5M: The Many Roads to Global Visual
  Geolocation", CVPR 2024.

Intended use covers training
----------------------------
Checked explicitly, because some geolocation benchmarks are evaluation-only.
The paper presents OSV-5M as a resource for *both* training and evaluation: it
benchmarks "various state-of-the-art image encoders, spatial representations,
and training strategies", and enforces a strict train/test separation
specifically so that learned geographic features can be evaluated "beyond mere
memorization". Training is the primary intended use, not a tolerated one.

Why this is the cleanest source tried so far
---------------------------------------------
* **Genuine per-image GPS.** Camera positions, not subject centroids -- the flaw
  that disqualified Google Landmarks v2 and that the Commons landmark set
  carries (its ``camera_lat`` equals ``landmark_lat`` for every row).
* **Real global spread.** 225 countries, against the Commons batch's 67%
  Americas concentration.
* **Street-level imagery**, matching the independent benchmark's own domain
  (which is entirely street-view), removing a distribution gap that confounded
  the commons01 comparison.

Layout and the batch unit
-------------------------
98 train shards of ~50,000 images each, ~2.35 GB per shard (229.5 GB total), so
**one shard is the natural batch unit**: a single self-contained download that
yields tens of thousands of images. Coordinates live in a separate 2.78 GB
``train.csv``, and shard membership is *not* contiguous within it -- 60 MB of
CSV (105,547 rows) matched only 2.2% of shard 00's ids. :mod:`osv5m_index`
therefore streams that CSV once into a compact local parquet, after which every
batch joins locally and transfers no metadata at all.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from .base import Availability, LicenceStatus, validate_frame

REPO_ID = "osv5m/osv5m"
TRAIN_SHARDS = 98
SHARD_TEMPLATE = "images/train/{shard:02d}.zip"
DEFAULT_INDEX = os.path.join("finetune", "work", "osv5m", "osv5m_index.parquet")
DEFAULT_RAW_DIR = os.path.join("finetune", "work", "osv5m", "raw")


class OSV5MAdapter:
    """Ingest one OSV-5M shard, joined against the cached coordinate index."""

    name = "osv5m"

    def __init__(
        self,
        shard: int | str = 0,
        index_path: str | None = None,
        raw_dir: str | None = None,
        repo_root: str | None = None,
        download: bool | str = True,
    ) -> None:
        root = repo_root or str(Path(__file__).resolve().parents[2])
        self.shard = int(shard)
        self.index_path = index_path or os.path.join(root, DEFAULT_INDEX)
        self.raw_dir = raw_dir or os.path.join(root, DEFAULT_RAW_DIR)
        self.download = download if isinstance(download, bool) else download != "false"
        self.extracted_dir: str | None = None
        self._shard_zip: str | None = None

    # -- gating -----------------------------------------------------------

    def availability(self) -> Availability:
        index_present = os.path.isfile(self.index_path)
        return Availability(
            licence=LicenceStatus.CLEAR,
            licence_note=(
                "CC-BY-SA-4.0, confirmed via the Hugging Face Hub API "
                "(cardData.license and repo tags); repo is not gated. No "
                "NonCommercial or NoDerivatives term. Paper presents the dataset "
                "for training as well as evaluation. Attribution + share-alike "
                "apply to redistribution (Astruc et al., CVPR 2024)."
            ),
            access_ok=index_present or self.download,
            access_note=(
                f"coordinate index at {self.index_path}"
                if index_present
                else (
                    f"no coordinate index yet; build it once with "
                    f"python datasets/osv5m_index.py {self.index_path}"
                )
            ),
        )

    # -- ingest -----------------------------------------------------------

    def _ensure_shard(self) -> str:
        """Return a local path to this shard's zip, downloading if needed."""
        filename = SHARD_TEMPLATE.format(shard=self.shard)
        local = os.path.join(self.raw_dir, filename.replace("/", os.sep))
        if os.path.isfile(local):
            return local
        if not self.download:
            raise FileNotFoundError(f"{local} missing and download disabled")

        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=self.raw_dir,
        )

    def load(self, limit: int | None = None, seed: int = 1337) -> pd.DataFrame:
        availability = self.availability()
        if not availability.usable:
            raise PermissionError(
                f"{self.name} cannot be ingested: {availability.blocked_reason()}"
            )
        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(
                f"Coordinate index missing: {self.index_path}. Build it once with "
                f"python datasets/osv5m_index.py {self.index_path}"
            )

        self._shard_zip = self._ensure_shard()
        extract_root = os.path.join(self.raw_dir, f"extracted_{self.shard:02d}")

        with zipfile.ZipFile(self._shard_zip) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith(".jpg")]
            ids = pd.Index([os.path.splitext(os.path.basename(n))[0] for n in members])

            index = pd.read_parquet(self.index_path)
            index["id"] = index["id"].astype(str)
            matched = index[index["id"].isin(set(ids))].copy()
            if matched.empty:
                raise RuntimeError(
                    f"No shard {self.shard:02d} ids found in {self.index_path}; "
                    "the index may be incomplete."
                )

            # Subsample BEFORE extracting: no point writing 50,000 files to disk
            # when the batch only needs a few thousand of them.
            if limit is not None and len(matched) > limit:
                matched = matched.sample(n=limit, random_state=seed)

            wanted = set(matched["id"])
            os.makedirs(extract_root, exist_ok=True)
            member_by_id = {
                os.path.splitext(os.path.basename(n))[0]: n for n in members
            }
            paths: dict[str, str] = {}
            for image_id in wanted:
                member = member_by_id.get(image_id)
                if member is None:
                    continue
                target = os.path.join(extract_root, f"{image_id}.jpg")
                if not os.path.isfile(target):
                    with archive.open(member) as source, open(target, "wb") as sink:
                        shutil.copyfileobj(source, sink)
                paths[image_id] = target

        self.extracted_dir = extract_root
        self.shard_image_count = len(members)
        self.matched_in_index = len(matched)

        frame = matched.rename(columns={"latitude": "lat", "longitude": "lon"}).copy()
        frame["image_path"] = frame["id"].map(paths)
        frame["image_id"] = "osv5m-" + frame["id"].astype(str)
        # Mapillary sequences are the leak unit: consecutive frames of one
        # sequence are metres apart along the same road.
        frame["tile_id"] = "seq-" + frame["sequence"].astype(str)
        frame["region"] = frame["country"].astype(str)
        frame["dataset"] = self.name

        return validate_frame(frame, self.name)

    # -- discard ----------------------------------------------------------

    def discard(self, frame: pd.DataFrame) -> int:
        """Delete this batch's extracted images and the shard zip.

        Keeps only checkpoints and metrics, per the ingest -> train ->
        checkpoint -> discard pattern, so disk stays flat across shards. The
        cached coordinate index is deliberately kept: it is small and every
        future shard needs it.
        """
        removed = 0
        for path in frame["image_path"]:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
        if self.extracted_dir and os.path.isdir(self.extracted_dir):
            shutil.rmtree(self.extracted_dir, ignore_errors=True)
        if self._shard_zip and os.path.isfile(self._shard_zip):
            try:
                os.remove(self._shard_zip)
            except OSError:
                pass
        return removed


def _parse_shards(shards: str | Iterable[int | str]) -> list[int]:
    if isinstance(shards, str):
        values = [part.strip() for part in shards.split(",") if part.strip()]
    else:
        values = list(shards)
    parsed = [int(value) for value in values]
    if not parsed:
        raise ValueError("At least one OSV-5M shard is required")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Duplicate shards are not allowed: {parsed}")
    invalid = [value for value in parsed if not 0 <= value < TRAIN_SHARDS]
    if invalid:
        raise ValueError(f"OSV-5M shards out of range 0-{TRAIN_SHARDS - 1}: {invalid}")
    return parsed


class _OSV5MEmbeddingCacheDataset:
    """Pickle-safe image dataset used by Windows DataLoader workers."""

    def __init__(self, rows: pd.DataFrame):
        from clip_transforms import clip_train_transform, clip_val_transform

        self.rows = rows[
            ["source_id", "raw_path", "embedding_index", "cache_role"]
        ].to_dict("records")
        self.train_transform = clip_train_transform()
        self.val_transform = clip_val_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        row = self.rows[index]
        image = Image.open(row["raw_path"]).convert("RGB")
        transform = (
            self.train_transform if row["cache_role"] == "train" else self.val_transform
        )
        return transform(image), int(row["embedding_index"])


class OSV5MCombinedAdapter:
    """Build one exactly thinned training pool from several OSV-5M shards.

    Cross-shard thinning and low disk use pull in opposite directions: exact
    greedy thinning needs all coordinates before it can decide which images to
    retain, while keeping every 2.35 GB shard zip would accumulate raw data.
    This adapter therefore uses two low-disk passes:

    1. Download one zip, record its member IDs, and immediately remove the zip.
    2. Thin the combined metadata once with the same exact latitude-banded
       implementation used by single-shard runs.
    3. After the train/validation split is known, re-fetch one zip at a time,
       extract only globally retained members, encode each image once, append
       its frozen CLIP feature to a compact cache, and remove the raw files and
       zip before advancing to the next shard.

    The extra transfer avoids both per-shard thinning (which is methodologically
    wrong here) and simultaneous storage of all unthinned raw shards.
    """

    name = "osv5m-combined"

    def __init__(
        self,
        shards: str | Iterable[int | str],
        index_path: str | None = None,
        raw_dir: str | None = None,
        pool_dir: str | None = None,
        repo_root: str | None = None,
        download: bool | str = True,
    ) -> None:
        root = repo_root or str(Path(__file__).resolve().parents[2])
        self.shards = _parse_shards(shards)
        self.index_path = index_path or os.path.join(root, DEFAULT_INDEX)
        self.raw_dir = raw_dir or os.path.join(root, DEFAULT_RAW_DIR)
        self.pool_dir = pool_dir or os.path.join(
            root, "finetune", "work", "osv5m", "combined_pool"
        )
        self.download = download if isinstance(download, bool) else download != "false"
        self.stats: dict[str, object] = {}
        self._active_zips: set[str] = set()
        self._selected_metadata: pd.DataFrame | None = None

    def availability(self) -> Availability:
        return OSV5MAdapter(
            shard=self.shards[0],
            index_path=self.index_path,
            raw_dir=self.raw_dir,
            download=self.download,
        ).availability()

    def _ensure_shard(self, shard: int) -> str:
        adapter = OSV5MAdapter(
            shard=shard,
            index_path=self.index_path,
            raw_dir=self.raw_dir,
            download=self.download,
        )
        path = adapter._ensure_shard()
        self._active_zips.add(path)
        return path

    def _discard_zip(self, path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        finally:
            self._active_zips.discard(path)

    @staticmethod
    def _member_ids(archive: zipfile.ZipFile) -> tuple[list[str], dict[str, str]]:
        members = [name for name in archive.namelist() if name.lower().endswith(".jpg")]
        member_by_id = {
            os.path.splitext(os.path.basename(name))[0]: name for name in members
        }
        return list(member_by_id), member_by_id

    def load_thinned(
        self,
        *,
        min_spacing_m: float,
        seed: int = 1337,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Return exactly cross-shard-thinned metadata, before image caching."""
        if limit is not None:
            raise ValueError("--limit is not supported for combined OSV-5M pools")
        availability = self.availability()
        if not availability.usable:
            raise PermissionError(
                f"{self.name} cannot be ingested: {availability.blocked_reason()}"
            )
        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(f"Coordinate index missing: {self.index_path}")

        from location_sampling import thin_by_spacing

        coordinate_index = pd.read_parquet(self.index_path).copy()
        coordinate_index["id"] = coordinate_index["id"].astype(str)
        coordinate_index = coordinate_index.drop_duplicates(subset=["id"]).set_index("id")

        metadata_parts: list[pd.DataFrame] = []
        archive_counts: dict[str, int] = {}
        matched_counts: dict[str, int] = {}
        print("\ncombined OSV-5M pass 1/2: catalogue shard membership (one zip at a time)")
        for shard in self.shards:
            shard_zip = self._ensure_shard(shard)
            try:
                with zipfile.ZipFile(shard_zip) as archive:
                    ids, _ = self._member_ids(archive)
                matched = coordinate_index.reindex(pd.Index(ids, name="id")).dropna(
                    subset=["latitude", "longitude"]
                ).reset_index()
                if matched.empty:
                    raise RuntimeError(
                        f"No shard {shard:02d} ids found in {self.index_path}; "
                        "the coordinate index may be incomplete."
                    )
                matched["source_shard"] = shard
                metadata_parts.append(matched)
                archive_counts[f"{shard:02d}"] = len(ids)
                matched_counts[f"{shard:02d}"] = len(matched)
                print(f"  shard {shard:02d}: {len(ids)} images, {len(matched)} indexed")
            finally:
                self._discard_zip(shard_zip)

        metadata = pd.concat(metadata_parts, ignore_index=True)
        metadata = metadata.drop_duplicates(subset=["id"]).reset_index(drop=True)
        metadata = metadata.rename(columns={"latitude": "lat", "longitude": "lon"})
        metadata["image_path"] = "pending"
        metadata["image_id"] = "osv5m-" + metadata["id"].astype(str)
        metadata["tile_id"] = "seq-" + metadata["sequence"].astype(str)
        metadata["region"] = metadata["country"].astype(str)
        metadata["dataset"] = self.name

        print(
            f"\nexact combined thinning across {len(metadata)} indexed images "
            f"from shards {','.join(f'{s:02d}' for s in self.shards)}"
        )
        thinned = thin_by_spacing(metadata, min_spacing_m=min_spacing_m, seed=seed)
        print(
            f"combined thinning to >={min_spacing_m:.0f} m: {len(metadata)} -> "
            f"{len(thinned)} ({len(thinned) / len(metadata):.1%} retained)"
        )

        selected_counts = {
            f"{shard:02d}": int(thinned["source_shard"].eq(shard).sum())
            for shard in self.shards
        }
        self._selected_metadata = thinned.copy()
        result = thinned[list((
            "image_id", "lat", "lon", "tile_id", "region", "dataset", "image_path"
        ))].reset_index(drop=True)

        country_counts = {
            str(country): int(count)
            for country, count in thinned["region"].value_counts().items()
        }
        self.stats = {
            "shards": self.shards,
            "archive_images_by_shard": archive_counts,
            "indexed_images_by_shard": matched_counts,
            "selected_locations_by_shard": selected_counts,
            "archive_images": int(sum(archive_counts.values())),
            "images_ingested": int(len(metadata)),
            "locations_after_thinning": int(len(result)),
            "retention": float(len(result) / len(metadata)),
            "country_distribution": country_counts,
            "pool_dir": self.pool_dir,
            "ingest_design": (
                "two-pass exact combined metadata thinning and embedding cache; "
                "one raw zip/extraction active"
            ),
        }
        return result

    def cache_embeddings(
        self,
        train_frame: pd.DataFrame,
        val_frame: pd.DataFrame,
        *,
        model,
        device: str,
        batch_size: int,
        amp: bool,
        num_workers: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Encode selected images once, deleting each raw shard immediately."""
        if self._selected_metadata is None:
            raise RuntimeError("load_thinned must run before cache_embeddings")

        import json
        import time

        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        ordered = pd.concat(
            [
                train_frame.assign(cache_role="train"),
                val_frame.assign(cache_role="val"),
            ],
            ignore_index=True,
        )
        ordered["embedding_index"] = np.arange(len(ordered), dtype=np.int64)
        source = self._selected_metadata[["image_id", "id", "source_shard"]].rename(
            columns={"id": "source_id"}
        )
        ordered = ordered.merge(source, on="image_id", how="left", validate="one_to_one")
        if ordered[["source_id", "source_shard"]].isna().any().any():
            raise RuntimeError("Could not map every split row back to its source shard")

        os.makedirs(self.pool_dir, exist_ok=True)
        cache_path = os.path.join(self.pool_dir, "image_embeddings.f16")
        metadata_path = os.path.join(self.pool_dir, "image_embeddings.json")
        cache = None
        embedding_dim = None
        raw_images_discarded = 0
        raw_bytes_extracted = 0
        encode_seconds = 0.0
        started = time.perf_counter()
        by_shard: dict[str, dict[str, float | int]] = {}
        model.image_encoder.eval()
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        print("\ncombined OSV-5M pass 2/2: cache frozen CLIP embeddings shard-by-shard")
        for shard in self.shards:
            shard_started = time.perf_counter()
            rows = ordered[ordered["source_shard"].eq(shard)].copy()
            wanted = set(rows["source_id"].astype(str))
            temp_dir = os.path.join(self.pool_dir, f"raw_{shard:02d}")
            os.makedirs(temp_dir, exist_ok=True)
            shard_zip = self._ensure_shard(shard)
            try:
                with zipfile.ZipFile(shard_zip) as archive:
                    _, member_by_id = self._member_ids(archive)
                    missing = wanted - set(member_by_id)
                    if missing:
                        raise RuntimeError(
                            f"Shard {shard:02d} is missing {len(missing)} selected image(s)"
                        )
                    paths: dict[str, str] = {}
                    for image_id in wanted:
                        target = os.path.join(temp_dir, f"{image_id}.jpg")
                        with archive.open(member_by_id[image_id]) as source_file, open(
                            target, "wb"
                        ) as sink:
                            shutil.copyfileobj(source_file, sink)
                        paths[image_id] = target
                        raw_bytes_extracted += os.path.getsize(target)

                rows["raw_path"] = rows["source_id"].astype(str).map(paths)
                loader = DataLoader(
                    _OSV5MEmbeddingCacheDataset(rows),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=False,
                )
                shard_encode_started = time.perf_counter()
                with torch.no_grad():
                    for images, indices in loader:
                        images = images.to(device, non_blocking=True)
                        if amp:
                            with torch.autocast("cuda", dtype=torch.float16):
                                features = model.image_encoder(images)
                        else:
                            features = model.image_encoder(images)
                        features = features.detach().to(dtype=torch.float16).cpu().numpy()
                        if cache is None:
                            embedding_dim = int(features.shape[1])
                            cache = np.memmap(
                                cache_path,
                                mode="w+",
                                dtype=np.float16,
                                shape=(len(ordered), embedding_dim),
                            )
                        cache[indices.numpy()] = features
                shard_encode_seconds = time.perf_counter() - shard_encode_started
                encode_seconds += shard_encode_seconds
                raw_images_discarded += len(rows)
                by_shard[f"{shard:02d}"] = {
                    "images": int(len(rows)),
                    "encode_seconds": shard_encode_seconds,
                    "wall_seconds": time.perf_counter() - shard_started,
                }
                print(
                    f"  shard {shard:02d}: cached {len(rows)} embeddings in "
                    f"{shard_encode_seconds:.1f}s; raw images discarded"
                )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
                self._discard_zip(shard_zip)

        if cache is None or embedding_dim is None:
            raise RuntimeError("No image embeddings were cached")
        cache.flush()
        cache_bytes = os.path.getsize(cache_path)
        wall_seconds = time.perf_counter() - started
        peak_gb = (
            torch.cuda.max_memory_reserved() / 1024 ** 3
            if str(device).startswith("cuda") else 0.0
        )

        lookup = ordered.set_index("image_id")["embedding_index"]
        def attach(frame: pd.DataFrame) -> pd.DataFrame:
            result = frame.copy()
            result["embedding_path"] = cache_path
            result["embedding_index"] = result["image_id"].map(lookup).astype(np.int64)
            result["embedding_count"] = len(ordered)
            result["embedding_dim"] = embedding_dim
            result["image_path"] = "cached-embedding"
            return result

        train_cached, val_cached = attach(train_frame), attach(val_frame)
        cache_stats = {
            "path": cache_path,
            "dtype": "float16",
            "shape": [len(ordered), embedding_dim],
            "bytes": cache_bytes,
            "gib": cache_bytes / 1024 ** 3,
            "wall_seconds": wall_seconds,
            "encode_seconds": encode_seconds,
            "images_per_second": len(ordered) / encode_seconds,
            "peak_reserved_gb": peak_gb,
            "raw_images_discarded": raw_images_discarded,
            "raw_bytes_extracted": raw_bytes_extracted,
            "raw_to_cache_size_ratio": raw_bytes_extracted / cache_bytes,
            "by_shard": by_shard,
            "train_transform": "one fixed stochastic CLIP-normalized augmentation per image",
            "val_transform": "deterministic CLIP inference preprocessing",
        }
        self.stats["embedding_cache"] = cache_stats
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(cache_stats, handle, indent=2)
        return train_cached, val_cached

    def load(self, limit: int | None = None, seed: int = 1337) -> pd.DataFrame:
        raise RuntimeError(
            "OSV5MCombinedAdapter requires load_thinned(min_spacing_m=..., seed=...)"
        )

    def discard(self, frame: pd.DataFrame) -> int:
        for path in list(self._active_zips):
            self._discard_zip(path)
        for shard in self.shards:
            shutil.rmtree(os.path.join(self.pool_dir, f"raw_{shard:02d}"), ignore_errors=True)
        cache_stats = self.stats.get("embedding_cache", {})
        return int(cache_stats.get("raw_images_discarded", 0))
