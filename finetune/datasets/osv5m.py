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
