"""Merge the separately-built gallery-pipeline galleries into one benchmark gallery.

The retrieval predictor can only search a single gallery at a time, so
benchmarking it against whichever small gallery happened to be loaded would
understate it. This concatenates every source gallery's FAISS vectors and
metadata into one combined, versioned gallery in the exact format the pipeline's
load stage produces (gallery.faiss + metadata.parquet + manifest.json), so
RetrievalGeolocator can consume it unmodified.

Requirements: all source galleries must share one embedding model and dimension
(they do -- open_clip:ViT-B-32:laion2b_s34b_b79k, 512-d). This is verified rather
than assumed, because mixing embedding spaces would silently produce meaningless
nearest neighbours.

Duplicate image_ids (the same photo present in two galleries) are dropped,
keeping the first occurrence, so the FAISS index and metadata stay 1:1.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("merge_galleries")

# v2 of target-regions supersedes v1 (same regions, 1305 vs 299 images), so the
# default set takes v2 for the widest possible coverage.
DEFAULT_SOURCES = [
    "gallery-pipeline/data/live-smoke/gallery/v1",
    "gallery-pipeline/data/target-regions/gallery/v2",
    "gallery-pipeline/data/global-test/gallery/v1",
]


def load_gallery(path: str) -> tuple[np.ndarray, "pa.Table", dict]:
    """Return (vectors, metadata_table, manifest) for one gallery directory."""
    index = faiss.read_index(os.path.join(path, "gallery.faiss"))
    vectors = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    table = pq.read_table(os.path.join(path, "metadata.parquet"))
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    if index.ntotal != table.num_rows:
        raise ValueError(f"{path}: index has {index.ntotal} vectors but metadata has {table.num_rows} rows")
    return vectors, table, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge galleries into one benchmark gallery.")
    parser.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--sources", nargs="*", default=None, help="gallery version dirs (repo-relative)")
    parser.add_argument("--out", default="gallery-pipeline/data/benchmark-gallery/gallery/v1")
    args = parser.parse_args()

    sources = args.sources or DEFAULT_SOURCES
    out_dir = os.path.join(args.repo_root, args.out)
    os.makedirs(out_dir, exist_ok=True)

    all_vectors: list[np.ndarray] = []
    all_tables: list[pa.Table] = []
    model_specs: set[str] = set()
    dimensions: set[int] = set()
    provenance: list[dict] = []

    for rel in sources:
        path = os.path.join(args.repo_root, rel)
        if not os.path.isdir(path):
            logger.warning("source gallery missing, skipping: %s", path)
            continue
        vectors, table, manifest = load_gallery(path)
        model_specs.add(str(manifest.get("embedding_model")))
        dimensions.add(int(manifest.get("embedding_dimension", vectors.shape[1])))
        all_vectors.append(vectors)
        all_tables.append(table)
        provenance.append({"source": rel, "images": int(vectors.shape[0]), "version": manifest.get("version")})
        logger.info("loaded %-55s %5d vectors", rel, vectors.shape[0])

    if not all_vectors:
        raise SystemExit("No source galleries found.")
    if len(model_specs) != 1:
        raise SystemExit(f"Refusing to merge different embedding models: {model_specs}")
    if len(dimensions) != 1:
        raise SystemExit(f"Refusing to merge different dimensions: {dimensions}")

    vectors = np.vstack(all_vectors)
    # promote_options keeps schemas compatible if a column set differs slightly.
    table = pa.concat_tables(all_tables, promote_options="default")

    # Drop duplicate image_ids, keeping first occurrence.
    image_ids = [str(v) for v in table["image_id"].to_pylist()]
    keep: list[int] = []
    seen: set[str] = set()
    for i, iid in enumerate(image_ids):
        if iid not in seen:
            seen.add(iid)
            keep.append(i)
    dropped = len(image_ids) - len(keep)
    if dropped:
        logger.info("dropping %d duplicate image_id row(s)", dropped)
    keep_arr = np.array(keep, dtype=np.int64)
    vectors = vectors[keep_arr]
    table = table.take(pa.array(keep_arr))

    # Vectors come from IndexFlatIP galleries and are already L2-normalised, but
    # normalise again so the merged index is cosine-correct regardless of source.
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Reassign index_position so it matches the merged ordering.
    if "index_position" in table.column_names:
        table = table.drop(["index_position"])
    table = table.add_column(0, "index_position", pa.array(range(table.num_rows), pa.int64()))

    faiss.write_index(index, os.path.join(out_dir, "gallery.faiss"))
    pq.write_table(table, os.path.join(out_dir, "metadata.parquet"), compression="zstd")

    regions: dict[str, int] = {}
    if "region" in table.column_names:
        for r in table["region"].to_pylist():
            regions[str(r)] = regions.get(str(r), 0) + 1

    manifest = {
        "version": 1,
        "status": "merged_for_benchmark",
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_image_count": int(index.ntotal),
        "embedding_dimension": int(vectors.shape[1]),
        "embedding_model": next(iter(model_specs)),
        "geographic_coverage": dict(sorted(regions.items())),
        "merged_from": provenance,
        "duplicates_dropped": dropped,
        "files": {"index": "gallery.faiss", "metadata": "metadata.parquet"},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logger.info("merged gallery -> %s", out_dir)
    logger.info("  total images: %d (dim %d)", index.ntotal, vectors.shape[1])
    for p in provenance:
        logger.info("    from %-55s %5d", p["source"], p["images"])


if __name__ == "__main__":
    main()
