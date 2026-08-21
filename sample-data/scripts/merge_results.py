"""Merge hybrid/geoclip results.csv with retrieval_results.csv -> combined.csv + summary."""
import csv, json, statistics
from pathlib import Path

OUT = Path("sample-data/results")
hybrid_geoclip = list(csv.DictReader((OUT/"results.csv").open()))
retrieval = list(csv.DictReader((OUT/"retrieval_results.csv").open()))
retrieval_summary = json.loads((OUT/"retrieval_summary.json").read_text())
hybrid_summary = json.loads((OUT/"summary.json").read_text())

# Index retrieval by (label, file)
retr_by_key = {(r["label"], r["file"]): r for r in retrieval}
retr_sum_by_key = {(r["label"], r["file"]): r for r in retrieval_summary}

combined_rows = []
for r in hybrid_geoclip:
    key = (r["label"], r["file"])
    if r["predictor"] == "retrieval":
        # replace with the actually-completed retrieval row
        rr = retr_by_key.get(key)
        if rr:
            combined_rows.append(rr)
        else:
            combined_rows.append(r)
    else:
        combined_rows.append(r)

with (OUT/"combined_results.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["label","file","predictor","lat","lon","error_km","confidence","note"])
    w.writeheader(); w.writerows(combined_rows)

# Build per-image summary with all 3 predictors
by_img = {}
for r in combined_rows:
    key = (r["label"], r["file"])
    by_img.setdefault(key, {})[r["predictor"]] = {
        "lat": float(r["lat"]) if r["lat"] not in ("", None) else None,
        "lon": float(r["lon"]) if r["lon"] not in ("", None) else None,
        "error_km": float(r["error_km"]) if r["error_km"] not in ("", None) else None,
        "confidence": float(r["confidence"]) if r["confidence"] not in ("", None, "None") else None,
        "note": r["note"],
    }

summary = []
for (label, file), preds in by_img.items():
    summary.append({"label": label, "file": file, "predictors": preds})

(OUT/"combined_summary.json").write_text(json.dumps(summary, indent=2))

# Aggregate metrics
def stats(errors):
    errors = [e for e in errors if e is not None]
    if not errors: return None
    return {
        "n": len(errors),
        "median_km": round(statistics.median(errors), 1),
        "mean_km": round(statistics.mean(errors), 1),
        "within_25_km": sum(1 for e in errors if e <= 25),
        "within_200_km": sum(1 for e in errors if e <= 200),
        "within_750_km": sum(1 for e in errors if e <= 750),
   }

agg = {}
for pred in ["hybrid", "geoclip", "retrieval"]:
    errs = [p[pred]["error_km"] for p in by_img.values() if pred in p and p[pred]["error_km"] is not None]
    agg[pred] = stats(errs)
(OUT/"aggregate_metrics.json").write_text(json.dumps(agg, indent=2))
print(json.dumps(agg, indent=2))
print(f"\nWrote combined_results.csv, combined_summary.json, aggregate_metrics.json")
