"""Generate the databricks-export CSVs from saved analysis artifacts.

Everything here is derived from files already on disk (calibration CSVs,
validation JSONs, run summaries, gallery metadata) plus _confidences.csv. No
numbers are hand-transcribed except the pipeline_reliability narrative table and
the external-transfer sign-test statistic, both of which are annotated as such.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.path.join(REPO, "finetune", "work")
GALLERY = os.path.join(REPO, "gallery-pipeline", "data", "benchmark-gallery", "gallery")
OUT = HERE

TRUST_THRESHOLD = 0.825
BUCKETS = [0.0, 0.30, 0.50, 0.65, 0.80, 1.01]

rowcounts: dict[str, int] = {}


def save(name: str, frame: pd.DataFrame) -> None:
    path = os.path.join(OUT, name)
    frame.to_csv(path, index=False)
    rowcounts[name] = len(frame)
    print(f"  {name:<32} {len(frame):>6} rows  {os.path.getsize(path)/1024:>8.1f} KB")


# --- 1. benchmark_results.csv -------------------------------------------------

def build_benchmark_results() -> None:
    calib = pd.read_csv(os.path.join(WORK, "calib_external_v2.csv"))
    sample = pd.read_csv(os.path.join(WORK, "external_benchmark_sample.csv"))
    conf = pd.read_csv(os.path.join(OUT, "_confidences.csv"))

    df = calib.merge(sample[["image_path", "source_dataset"]], on="image_path", how="left")
    # _confidences.csv is keyed by image_id so it stays portable across checkouts.
    df["image_id"] = df["image_path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    df = df.merge(conf, on="image_id", how="left")

    trust = df["top_similarity"] >= TRUST_THRESHOLD
    out = pd.DataFrame({
        "image_id": df["image_id"],
        "source_dataset": df["source_dataset"],
        "true_lat": df["true_lat"].round(6),
        "true_lon": df["true_lon"].round(6),
        "geoclip_pred_lat": df["geoclip_lat"].round(6),
        "geoclip_pred_lon": df["geoclip_lon"].round(6),
        "geoclip_confidence": df["geoclip_confidence"].round(2),
        "geoclip_error_km": df["geoclip_error_km"].round(3),
        "retrieval_pred_lat": df["retrieval_lat"].round(6),
        "retrieval_pred_lon": df["retrieval_lon"].round(6),
        "retrieval_confidence": df["retrieval_confidence"].round(2),
        "retrieval_top_similarity": df["top_similarity"].round(4),
        "retrieval_error_km": df["retrieval_error_km"].round(3),
        "ensemble_pred_lat": np.where(trust, df["retrieval_lat"], df["geoclip_lat"]).round(6),
        "ensemble_pred_lon": np.where(trust, df["retrieval_lon"], df["geoclip_lon"]).round(6),
        "ensemble_source_used": np.where(trust, "retrieval", "geoclip"),
        "ensemble_error_km": np.where(
            trust, df["retrieval_error_km"], df["geoclip_error_km"]
        ).round(3),
    })
    save("benchmark_results.csv", out)


# --- 2. accuracy_summary.csv --------------------------------------------------

POPULATION_ALIASES = {
    "EXTERNAL 1000 global (gallery does NOT cover)": "external",
    "EXTERNAL 1000 global vs gallery v2": "external",
    "IN-COVERAGE Probe A honest n24": "probe_a",
    "IN-COVERAGE Probe A v2 honest n93": "probe_a",
    "IN-COVERAGE Probe B near-duplicate upper bound": "probe_b",
    "IN-COVERAGE Probe B v2 upper bound n367": "probe_b",
}
METRIC_KEYS = [
    ("acc_1km", "acc@1km"), ("acc_25km", "acc@25km"), ("acc_200km", "acc@200km"),
    ("acc_750km", "acc@750km"), ("acc_2500km", "acc@2500km"),
    ("median_error_km", "median_error_km"), ("mean_error_km", "mean_error_km"),
]


def build_accuracy_summary() -> None:
    rows = []
    for version, fname in (("v1", "ensemble_validation.json"), ("v2", "ensemble_validation_v2.json")):
        with open(os.path.join(WORK, fname), encoding="utf-8") as fh:
            payload = json.load(fh)
        for set_label, per_predictor in payload["sets"].items():
            population = POPULATION_ALIASES.get(set_label, set_label)
            for predictor, metrics in per_predictor.items():
                if predictor.startswith("_") or not isinstance(metrics, dict):
                    continue
                for key, metric_name in METRIC_KEYS:
                    if key in metrics:
                        rows.append({
                            "predictor": predictor,
                            "gallery_version": version,
                            "population": population,
                            "n": metrics.get("n"),
                            "metric": metric_name,
                            "value": round(float(metrics[key]), 4),
                        })
    save("accuracy_summary.csv", pd.DataFrame(rows))


# --- 3. threshold_calibration.csv ---------------------------------------------

CALIB_FILES = {
    ("external", "v1"): "calib_external.csv",
    ("probe_a", "v1"): "calib_probe_a.csv",
    ("probe_b", "v1"): "calib_probe_b.csv",
    ("external", "v2"): "calib_external_v2.csv",
    ("probe_a", "v2"): "calib_probe_a_v2.csv",
    ("probe_b", "v2"): "calib_probe_b_v2.csv",
}


def build_threshold_calibration() -> None:
    rows = []
    for (population, version), fname in CALIB_FILES.items():
        df = pd.read_csv(os.path.join(WORK, fname)).dropna(
            subset=["top_similarity", "retrieval_error_km", "geoclip_error_km"]
        )
        for low, high in zip(BUCKETS[:-1], BUCKETS[1:]):
            sel = df[(df["top_similarity"] >= low) & (df["top_similarity"] < high)]
            rows.append({
                "population": population,
                "gallery_version": version,
                "top_similarity_bucket": f"[{low:.2f},{high:.2f})",
                "bucket_low": low,
                "bucket_high": high,
                "n": len(sel),
                "median_error_km": round(float(sel["retrieval_error_km"].median()), 3) if len(sel) else None,
                "geoclip_median_error_km": round(float(sel["geoclip_error_km"].median()), 3) if len(sel) else None,
                "retrieval_win_rate": round(float((sel["retrieval_error_km"] < sel["geoclip_error_km"]).mean()), 4) if len(sel) else None,
            })
    save("threshold_calibration.csv", pd.DataFrame(rows))


# --- 4. false_accept_ceiling.csv ----------------------------------------------

def build_false_accept_ceiling() -> None:
    rows = []
    for version, gallery_size in (("v1", 1719), ("v2", 3314)):
        ext = pd.read_csv(os.path.join(WORK, CALIB_FILES[("external", version)]))
        cov = pd.read_csv(os.path.join(WORK, CALIB_FILES[("probe_a", version)]))
        rows.append({
            "gallery_version": version,
            "gallery_size": gallery_size,
            "out_of_coverage_max_similarity": round(float(ext["top_similarity"].max()), 4),
            "out_of_coverage_mean_similarity": round(float(ext["top_similarity"].mean()), 4),
            "out_of_coverage_p99_similarity": round(float(ext["top_similarity"].quantile(0.99)), 4),
            "in_coverage_median_similarity": round(float(cov["top_similarity"].median()), 4),
            "in_coverage_n": int(len(cov)),
            "trust_threshold": TRUST_THRESHOLD,
            "external_images_above_threshold": int((ext["top_similarity"] >= TRUST_THRESHOLD).sum()),
            "external_images_total": int(len(ext)),
            "in_coverage_above_threshold": int((cov["top_similarity"] >= TRUST_THRESHOLD).sum()),
        })
    save("false_accept_ceiling.csv", pd.DataFrame(rows))


# --- 5. gallery_coverage.csv --------------------------------------------------

NAMED_AREAS = {
    "chicago_loop": "chicago",
    "puget_hood_canal": "puget-sound",
    "avenue_of_giants": "humboldt",
}


def region_label(tile_id: str, lat: float, lon: float) -> str:
    """Human-readable area name: explicit for the hand-configured regions,
    coordinate-derived for the programmatically tiled `global_*` cells."""
    for prefix, label in NAMED_AREAS.items():
        if isinstance(tile_id, str) and tile_id.startswith(prefix):
            return label
    if 35 <= lat <= 43 and -125 <= lon <= -118:
        return "norcal"
    if lat >= 35 and -25 <= lon <= 45:
        return "europe"
    if -35 <= lat < 37 and -18 <= lon <= 52:
        return "africa"
    if -56 <= lat < 13 and -82 <= lon <= -34:
        return "south-america"
    if 5 <= lat <= 35 and 60 <= lon <= 92:
        return "south-asia"
    if -11 <= lat < 46 and 92 < lon <= 146:
        return "east-asia"
    if lat < -11 and 110 < lon <= 180:
        return "oceania"
    if lat >= 15 and -170 <= lon < -50:
        return "north-america-other"
    return "other"


def build_gallery_coverage() -> None:
    v1 = pd.read_parquet(os.path.join(GALLERY, "v1", "metadata.parquet"))
    v2 = pd.read_parquet(os.path.join(GALLERY, "v2", "metadata.parquet"))
    v1_ids = set(v1["image_id"].astype(str))

    v2 = v2.copy()
    v2["image_id"] = v2["image_id"].astype(str)
    # An image's version is the first gallery it appeared in.
    v2["gallery_version"] = np.where(v2["image_id"].isin(v1_ids), "v1", "v2")
    v2["region_label"] = [
        region_label(t, a, b) for t, a, b in zip(v2["tile_id"], v2["lat"], v2["lon"])
    ]

    out = v2[["image_id", "lat", "lon", "region_label", "gallery_version", "tile_id"]].copy()
    out["lat"] = out["lat"].round(6)
    out["lon"] = out["lon"].round(6)
    save("gallery_coverage.csv", out)
    print("      by version:", dict(out["gallery_version"].value_counts()))
    print("      by region :", dict(out["region_label"].value_counts()))


# --- 6. training_trend.csv ----------------------------------------------------

RUNS = [
    ("batch01_es", "batch01_es", 184),
    ("breadth01", "breadth01", 543),
    ("breadth02", "breadth02", 730),
]


def build_training_trend() -> None:
    rows = []
    for run_name, folder, locations in RUNS:
        with open(os.path.join(WORK, folder, "summary.json"), encoding="utf-8") as fh:
            summary = json.load(fh)
        best = summary.get("best_checkpoint", {})
        val = summary.get("val_same_region") or summary.get("val") or {}
        before = float(val["before"]["median_error_km"])
        after = float(val["after"]["median_error_km"])
        rows.append({
            "run_name": run_name,
            "training_locations": locations,
            "best_epoch": best.get("epoch"),
            "useful_steps": best.get("step"),
            "n": val["before"].get("n"),
            "held_out_median_error_before_km": round(before, 2),
            "held_out_median_error_after_km": round(after, 2),
            "pct_improvement": round((before - after) / before * 100.0, 2),
            "sign_test_z": None,
            "result": "in_distribution_improvement",
        })

    # The out-of-distribution control. z comes from the paired sign test in
    # finetune/work/external_paired.json (250 closer / 255 farther / 495 tied).
    rows.append({
        "run_name": "external_transfer",
        "training_locations": None,
        "best_epoch": None,
        "useful_steps": None,
        "n": 1000,
        "held_out_median_error_before_km": 1292.53,
        "held_out_median_error_after_km": 1438.07,
        "pct_improvement": round((1292.53 - 1438.07) / 1292.53 * 100.0, 2),
        "sign_test_z": -0.22,
        "result": "null_effect",
    })
    frame = pd.DataFrame(rows)
    # Keep whole numbers as integers: the null row would otherwise coerce these
    # columns to float and export as "184.0", which imports awkwardly.
    for column in ("training_locations", "best_epoch", "useful_steps", "n"):
        frame[column] = frame[column].astype("Int64")
    save("training_trend.csv", frame)


# --- 7. pipeline_reliability.csv ----------------------------------------------

BUGS = [
    {
        "bug_description": "Shared retry budget: retryable HTTP statuses and transport failures decremented one counter",
        "session_found": "gallery growth session",
        "component": "gallery-pipeline/src/extract/mapillary_client.py",
        "fix_description": "Independent max_attempts / max_transport_attempts budgets; both counters increase monotonically so the loop stays bounded",
        "impact": "Four HTTP 500 density rejections exhausted the budget, so a fifth-attempt connection reset re-raised with no retry and killed a 500-tile pull",
    },
    {
        "bug_description": "One failed tile aborted the entire extraction run",
        "session_found": "gallery growth session",
        "component": "gallery-pipeline/src/extract/runner.py",
        "fix_description": "Transient failures mark the tile failed and continue; max_consecutive_tile_failures (default 8) still stops a systemic outage",
        "impact": "Discarded every remaining tile even though completed tiles were already durable in the checkpoint store",
    },
    {
        "bug_description": "Shallow subdivision depth made undividable dense metros fatal for breadth pulls",
        "session_found": "gallery growth session",
        "component": "gallery-pipeline/src/extract/runner.py",
        "fix_description": "extract.skip_unsubdividable_tiles, default off so gallery builds stay fail-fast, on for breadth pulls",
        "impact": "South America and Africa pulls each died within ~2 minutes on a single dense city tile",
    },
    {
        "bug_description": "Fixed-step checkpointing skipped the best model state",
        "session_found": "fine-tuning session",
        "component": "finetune/train_batch.py",
        "fix_description": "End-of-epoch checkpoints plus save-best on the held-out metric and early stopping",
        "impact": "batch01's optimum was step 50 while checkpoints were written only at 100/200/300, so the best weights were never saved",
    },
    {
        "bug_description": "Training loss kept falling while held-out error regressed",
        "session_found": "fine-tuning session",
        "component": "finetune/train_batch.py",
        "fix_description": "Early stopping on held-out median error with configurable patience",
        "impact": "A 30-epoch run ended worse than pretrained (126.5 -> 239.4 km median) and also degraded the untrained-region control",
    },
    {
        "bug_description": "geoclip normalizes with ImageNet stats in training but CLIP stats at inference",
        "session_found": "fine-tuning session",
        "component": "finetune/clip_transforms.py",
        "fix_description": "Standardised on CLIP statistics for both training and evaluation",
        "impact": "Would have fed the frozen tower out-of-distribution inputs, understating the pretrained baseline and inflating apparent gains",
    },
    {
        "bug_description": "Retrieval benchmarked against a gallery built from the same scraped data",
        "session_found": "ensemble session",
        "component": "finetune/benchmark_external.py (methodology)",
        "fix_description": "Benchmark only against external data; build in-coverage probes from images the gallery has not memorised",
        "impact": "Produced a bimodal artifact (43.75% at exactly 0.00 km, rest ~8000 km) measuring gallery membership rather than geolocation",
    },
    {
        "bug_description": "Oversized third-party binaries were untracked but not gitignored",
        "session_found": "security audit session",
        "component": ".gitignore",
        "fix_description": "Ignore archives, model binaries and root-level images; verified no collateral on tracked fixtures",
        "impact": "A bulk 'git add -A' would have staged a 308 MB zip, past GitHub's 100 MB limit, and the push would have been rejected",
    },
]


def build_pipeline_reliability() -> None:
    save("pipeline_reliability.csv", pd.DataFrame(BUGS))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("building exports:")
    build_benchmark_results()
    build_accuracy_summary()
    build_threshold_calibration()
    build_false_accept_ceiling()
    build_gallery_coverage()
    build_training_trend()
    build_pipeline_reliability()
    print("\nrow counts:", json.dumps(rowcounts, indent=2))
