"""Build a compact, reusable local index of OSV-5M's metadata.

Why this exists
---------------
OSV-5M ships coordinates in ``train.csv``, which is 2.78 GB of 32 columns for
5.1M rows. The images ship as 98 zip shards of ~50,000 images each, and shard
membership is **not** contiguous in the CSV -- a check over the first 60 MB
(105,547 rows) matched only 1,111 of shard 00's 50,000 ids, i.e. 2.2%. So there
is no way to resolve one shard's coordinates from a slice of the CSV; the whole
thing has to be read.

Reading it once and keeping only the four columns training needs turns a 2.78 GB
repeated cost into a ~100 MB one-time one. Every later batch joins against the
cached parquet and transfers no metadata at all.

Columns kept:
    id        image id, matches ``<id>.jpg`` inside the shard zips
    latitude  per-image GPS, genuine camera position (not a subject centroid)
    longitude
    sequence  Mapillary capture sequence -- the natural leak unit, since frames
              of one sequence are the same road seconds apart
    country   for reporting geographic spread
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request

import pandas as pd

CSV_URL = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main/train.csv"
KEEP_COLUMNS = ["id", "latitude", "longitude", "sequence", "country"]
CHUNK_BYTES = 8 * 1024 * 1024


def _remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def build_index(
    destination: str,
    url: str = CSV_URL,
    progress_every: int = 25,
    max_retries: int = 8,
) -> str:
    """Stream the remote CSV in resumable ranges and write a compact parquet.

    Ranged rather than one long read: a single 2.78 GB stream is fragile over a
    home connection, and the first attempt died at ~0.4 GB with
    ``ConnectionResetError``. Each chunk is an independent HTTP Range request,
    so a dropped connection costs one chunk and retries from the same offset
    instead of restarting the whole download.

    Streamed rather than saved so the 2.78 GB never lands on disk; only the
    reduced frame is persisted.
    """
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)

    size = _remote_size(url)
    started = time.perf_counter()
    frames: list[pd.DataFrame] = []
    offset = 0
    leftover = b""
    header: list[str] | None = None
    chunks_seen = 0
    retries_used = 0

    while offset < size:
        end = min(offset + CHUNK_BYTES, size) - 1
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}"})

        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    block = response.read()
                break
            except Exception as exc:  # noqa: BLE001 - any transport failure retries
                retries_used += 1
                if attempt == max_retries - 1:
                    raise
                wait = min(2**attempt, 30)
                print(f"  chunk at {offset / 1024**3:.2f} GB failed ({exc}); "
                      f"retry {attempt + 1}/{max_retries} in {wait}s", flush=True)
                time.sleep(wait)

        offset += len(block)
        chunks_seen += 1
        data = leftover + block
        # Keep the final partial line for the next chunk.
        cut = data.rfind(b"\n")
        if cut == -1:
            leftover = data
            continue
        body, leftover = data[: cut + 1], data[cut + 1 :]

        text = body.decode("utf-8", "replace")
        lines = text.splitlines()
        if header is None:
            header = lines[0].split(",")
            lines = lines[1:]
        if not lines:
            continue

        wanted = {name: header.index(name) for name in KEEP_COLUMNS}
        rows = [line.split(",") for line in lines if line]
        # Rows can carry stray commas inside URLs, so take columns positionally
        # against the known header and drop anything too short to index.
        width = max(wanted.values()) + 1
        rows = [r for r in rows if len(r) >= width]
        if not rows:
            continue
        frames.append(pd.DataFrame(
            {name: [r[position] for r in rows] for name, position in wanted.items()}
        ))

        if chunks_seen % progress_every == 0:
            elapsed = time.perf_counter() - started
            rate = offset / max(elapsed, 1e-9) / 1024**2
            row_count = sum(len(f) for f in frames)
            print(f"  {offset / size * 100:5.1f}%  {offset / 1024**3:5.2f} GB  "
                  f"{row_count:>9,} rows  {rate:5.1f} MB/s  "
                  f"{elapsed / 60:5.1f} min  retries={retries_used}", flush=True)

    combined = pd.concat(frames, ignore_index=True)
    combined["latitude"] = pd.to_numeric(combined["latitude"], errors="coerce")
    combined["longitude"] = pd.to_numeric(combined["longitude"], errors="coerce")
    combined = combined.dropna(subset=["latitude", "longitude"])
    combined["id"] = combined["id"].astype(str).str.strip()
    combined["latitude"] = combined["latitude"].astype("float32")
    combined["longitude"] = combined["longitude"].astype("float32")
    combined["sequence"] = combined["sequence"].astype("category")
    combined["country"] = combined["country"].astype("category")

    combined.to_parquet(destination, index=False)
    elapsed = time.perf_counter() - started
    print(f"\nindexed {len(combined):,} rows in {elapsed / 60:.1f} min "
          f"-> {destination} ({os.path.getsize(destination) / 1024**2:.1f} MB)")
    return destination


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "osv5m_index.parquet"
    build_index(target)
