"""Run View Finder against collected sample images and score vs ground truth.

For each manifest entry, runs the requested predictors, reads the JSON result,
computes haversine error vs the true GPS from the manifest, and writes a
summary CSV + JSON.
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def run_predictor(img, predictor, gallery, out_json, env):
    cmd = [sys.executable, "-m", "viewfinder.main",
           "--input", str(img), "--predictor", predictor,
           "--output", str(out_json), "--quiet"]
    if gallery:
        cmd += ["--gallery-path", str(gallery)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if r.returncode not in (0, 3):
        return None, f"exit {r.returncode}: {r.stderr.strip()[:200]}"
    try:
        return json.loads(Path(out_json).read_text()), None
    except Exception as e:
        return None, f"read json failed: {e}"


def top_pred(result):
    if not result:
        return None
    if "results" in result:  # 'both' shape
        return None
    p = result.get("prediction")
    if p and p.get("latitude") is not None:
        return p
    preds = result.get("predictions") or []
    return preds[0] if preds else None


def main(manifest_path, gallery, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text())
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
    rows = []
    summary = []
    predictors = ["hybrid", "geoclip"]
    if gallery:
        predictors.append("retrieval")
    for entry in manifest:
        img = Path(entry["file"])
        true_lat, true_lon = entry["true_lat"], entry["true_lon"]
        per_entry = {"label": entry["label"], "file": str(img), "true": [true_lat, true_lon], "results": {}}
        for pred in predictors:
            out_json = out_dir / f"{img.stem}_{pred}.json"
            res, err = run_predictor(img, pred, gallery if pred == "retrieval" else None, out_json, env)
            if err:
                rows.append([entry["label"], img.name, pred, "", "", "", "", err])
                per_entry["results"][pred] = {"error": err}
                continue
            p = top_pred(res)
            if p and p.get("latitude") is not None:
                lat, lon = p["latitude"], p["longitude"]
                err_km = haversine_km(true_lat, true_lon, lat, lon)
                conf = p.get("confidence_percent")
                rows.append([entry["label"], img.name, pred, f"{lat:.5f}", f"{lon:.5f}",
                             f"{err_km:.1f}", f"{conf}", ""])
                per_entry["results"][pred] = {"lat": lat, "lon": lon, "error_km": round(err_km, 1),
                                              "confidence": conf, "city": p.get("city"),
                                              "country": p.get("country")}
            else:
                abstain = res.get("prediction") is None
                rows.append([entry["label"], img.name, pred, "", "", "", "",
                             "abstain" if abstain else "no-pred"])
                per_entry["results"][pred] = {"abstain": abstain}
            print(f"  {entry['label']:20s} {pred:10s} -> {per_entry['results'][pred]}", flush=True)
        summary.append(per_entry)
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "file", "predictor", "lat", "lon", "error_km", "confidence", "note"])
        w.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path} and {out_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    manifest = sys.argv[1]
    gallery = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("sample-data/results")
    main(manifest, gallery, out)
