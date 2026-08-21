"""Run only the retrieval predictor for entries in the manifest."""
from __future__ import annotations
import csv, json, math, os, subprocess, sys
from pathlib import Path

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def main(manifest_path, gallery, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text())
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
    rows = []
    summary = []
    for entry in manifest:
        img = Path(entry["file"])
        true_lat, true_lon = entry["true_lat"], entry["true_lon"]
        out_json = out_dir / f"{img.stem}_retrieval.json"
        cmd = [sys.executable, "-m", "viewfinder.main", "--input", str(img),
               "--predictor", "retrieval", "--gallery-path", str(gallery),
               "--output", str(out_json), "--quiet"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        if r.returncode not in (0, 3):
            rows.append([entry["label"], img.name, "retrieval", "", "", "", "", f"exit {r.returncode}: {r.stderr.strip()[:200]}"])
            summary.append({"label": entry["label"], "file": str(img), "result": {"error": f"exit {r.returncode}"}})
            print(f"  {entry['label']:20s} retrieval -> FAIL exit {r.returncode}", flush=True)
            continue
        try:
            res = json.loads(Path(out_json).read_text())
        except Exception as e:
            rows.append([entry["label"], img.name, "retrieval", "", "", "", "", f"read json: {e}"])
            print(f"  {entry['label']:20s} retrieval -> json err {e}", flush=True)
            continue
        p = res.get("prediction")
        if p and p.get("latitude") is not None:
            lat, lon = p["latitude"], p["longitude"]
            err = haversine_km(true_lat, true_lon, lat, lon)
            conf = p.get("confidence_percent")
            rows.append([entry["label"], img.name, "retrieval", f"{lat:.5f}", f"{lon:.5f}", f"{err:.1f}", f"{conf}", ""])
            summary.append({"label": entry["label"], "file": str(img), "lat": lat, "lon": lon,
                            "error_km": round(err,1), "confidence": conf, "city": p.get("city"), "country": p.get("country")})
            print(f"  {entry['label']:20s} retrieval -> {err:.1f} km  {p.get('city')}/{p.get('country')}  conf={conf}", flush=True)
        else:
            rows.append([entry["label"], img.name, "retrieval", "", "", "", "", "abstain"])
            print(f"  {entry['label']:20s} retrieval -> abstain", flush=True)
    csv_path = out_dir / "retrieval_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["label","file","predictor","lat","lon","error_km","confidence","note"]); w.writerows(rows)
    (out_dir / "retrieval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]) if len(sys.argv) > 3 else Path("sample-data/results"))
