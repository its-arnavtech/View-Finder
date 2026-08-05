"""Recover the per-image confidence scores the calibration runs never captured.

The calibration CSVs kept only each predictor's top-1 candidate, so they carry
retrieval's raw cosine `top_similarity` but not the 0-100 confidence that
`aggregator.aggregate()` derives -- the number the CLI actually shows a user.
That confidence is the subject of a specific finding (it measures neighbour
agreement, so it can read high while the underlying match is weak), so the
export needs both columns side by side to make that chartable.

Recomputing requires top-k candidates, hence this re-run rather than a reshape
of the existing files. Predictions themselves are unchanged; only the confidence
column is new.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from viewfinder.aggregator import DEFAULT_WEAK_SIMILARITY_THRESHOLD, FramePrediction, aggregate

SAMPLE = os.path.join(REPO, "finetune", "work", "external_benchmark_sample.csv")
GALLERY_V2 = os.path.join(REPO, "gallery-pipeline", "data", "benchmark-gallery", "gallery", "v2")
OUT = os.path.join(HERE, "_confidences.csv")
BATCH = 32
TOP_K = 5


def confidence_for(candidates, weak_threshold=None) -> float:
    """Aggregator confidence (0-100) for one image's candidate list."""
    if not candidates:
        return float("nan")
    frame = FramePrediction(candidates=list(candidates), frame_index=0, timestamp=0.0)
    result = aggregate([frame], top_k=TOP_K, weak_similarity_threshold=weak_threshold)
    return float(result.primary.confidence)


def main() -> None:
    from viewfinder.geolocator import GeoLocator
    from viewfinder.retrieval_geolocator import RetrievalGeolocator

    frame = pd.read_csv(SAMPLE)
    frame = frame[frame["image_path"].map(os.path.exists)].reset_index(drop=True)
    print(f"images: {len(frame)}", flush=True)

    print("loading predictors...", flush=True)
    geoclip = GeoLocator(offline=True)
    retrieval = RetrievalGeolocator(GALLERY_V2)

    rows = []
    paths = frame["image_path"].tolist()
    for start in range(0, len(paths), BATCH):
        chunk = paths[start : start + BATCH]
        images = [Image.open(p).convert("RGB") for p in chunk]

        geo_batch = geoclip.predict_batch(images, top_k=TOP_K)
        ret_batch = retrieval.predict_batch(images, top_k=TOP_K)

        for path, geo_c, ret_c in zip(chunk, geo_batch, ret_batch):
            rows.append({
                # Keyed by image_id, not the absolute path: this file is
                # committed, and a machine-specific path would be both a
                # needless disclosure and useless on any other checkout.
                "image_id": os.path.splitext(os.path.basename(path))[0],
                # GeoCLIP scores are softmax probabilities with no similarity
                # scale, so the weak-similarity re-rank does not apply to it.
                "geoclip_confidence": confidence_for(geo_c, None),
                "retrieval_confidence": confidence_for(ret_c, DEFAULT_WEAK_SIMILARITY_THRESHOLD),
            })

        done = min(start + BATCH, len(paths))
        if done % (BATCH * 8) == 0 or done == len(paths):
            print(f"  {done}/{len(paths)}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(out)} rows)")
    print(out[["geoclip_confidence", "retrieval_confidence"]].describe().to_string())


if __name__ == "__main__":
    main()
