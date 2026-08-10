"""Re-score a finished run's saved checkpoints on both metrics.

Retraining is not needed to answer "which checkpoint was actually best?". Only
the location encoder trains, so for a fixed image set the image features are
constant -- encode the held-out slice and the independent benchmark **once**,
then swapping in each saved location encoder and re-scoring costs a forward
pass through a 9.5M-parameter MLP plus a gallery lookup. Every epoch of a
finished run can therefore be scored in about the time one training epoch took.

This exists because ``commons01`` was probed externally only every 2 epochs
while checkpoints were written every epoch, so half its external trajectory was
never measured. It is also the tool for re-deriving a best checkpoint under the
corrected selection rule without discarding a run.

Uses the same fixed external sample as training (``load_external_sample``'s
default seed), so numbers are directly comparable to the run's own log.

    python rescore_checkpoints.py --run-name commons01
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geoclip.model.misc import file_dir as geoclip_file_dir
from geoclip.model.misc import load_gps_data

import train_batch as tb
from checkpoint_selection import (
    CheckpointSelector,
    divergence_summary,
    format_dual_trajectory,
)
from evaluate import encode_val_images, evaluate
from external_probe import ExternalProbe, interpret, load_external_sample

EPOCH_PATTERN = re.compile(r"location_encoder_epoch(\d+)_step(\d+)\.pth$")


def discover_checkpoints(ckpt_dir: str) -> list[dict]:
    """Return every scoreable checkpoint in epoch order, pretrained first."""
    found: list[dict] = []

    pretrained = os.path.join(ckpt_dir, "location_encoder_pretrained.pth")
    if os.path.isfile(pretrained):
        found.append({"label": "pretrained", "epoch": -1, "step": 0, "path": pretrained})

    for path in sorted(glob.glob(os.path.join(ckpt_dir, "location_encoder_epoch*_step*.pth"))):
        match = EPOCH_PATTERN.search(os.path.basename(path))
        if match:
            found.append({
                "label": f"epoch{int(match.group(1))}",
                "epoch": int(match.group(1)),
                "step": int(match.group(2)),
                "path": path,
            })

    final = sorted(glob.glob(os.path.join(ckpt_dir, "location_encoder_final*.pth")))
    if final:
        found.append({"label": "final", "epoch": 99, "step": -1, "path": final[-1]})

    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--external-sample", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--write-best", action="store_true", default=True,
                        help="write location_encoder_best_{indist,external}.pth")
    parser.add_argument("--no-write-best", dest="write_best", action="store_false")
    args = parser.parse_args()

    ckpt_dir = os.path.join(tb.CKPT_ROOT, args.run_name)
    work_dir = os.path.join(tb.WORK_ROOT, args.run_name)
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"No checkpoint directory: {ckpt_dir}")

    checkpoints = discover_checkpoints(ckpt_dir)
    if not checkpoints:
        raise SystemExit(f"No scoreable checkpoints in {ckpt_dir}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    device = "cuda"
    amp = not args.no_amp

    print("=" * 78)
    print(f"Re-scoring {len(checkpoints)} checkpoints from run '{args.run_name}'")
    print("=" * 78)

    from model_setup import build_model, freeze_image_side
    model = build_model(device=device)
    freeze_image_side(model)
    gps_gallery = load_gps_data(
        os.path.join(geoclip_file_dir, "gps_gallery", "coordinates_100K.csv")
    )

    # --- encode both fixed image sets once ---
    val_path = os.path.join(work_dir, "split_val.csv")
    val_features = val_coords = None
    if os.path.isfile(val_path):
        val_frame = pd.read_csv(val_path)
        print(f"\nencoding {len(val_frame)} held-out images (once)...")
        val_features, val_coords = encode_val_images(
            model, val_frame, args.eval_batch_size, device, amp=amp
        )
    else:
        print(f"\nno {val_path}; in-distribution column will be blank")

    external_frame = load_external_sample(sample_size=args.external_sample)
    print(f"encoding {len(external_frame)} independent benchmark images (once)...")
    probe = ExternalProbe(model, external_frame, gps_gallery, device,
                          batch_size=args.eval_batch_size, amp=amp)

    # --- score every checkpoint ---
    print(f"\nscoring {len(checkpoints)} checkpoints...\n")
    trajectory: list[dict] = []
    baseline_external: dict | None = None
    baseline_indist: dict | None = None

    for entry in checkpoints:
        state = torch.load(entry["path"], map_location=device)
        model.location_encoder.load_state_dict(state["location_encoder"])

        external = probe.measure()
        indist = (
            evaluate(model, val_features, val_coords, gps_gallery, device)
            if val_features is not None else None
        )

        if entry["label"] == "pretrained":
            baseline_external = external
            baseline_indist = indist
            probe.baseline = external

        point = {
            "label": entry["label"],
            "epoch": entry["epoch"],
            "step": entry["step"],
            "path": entry["path"],
            "mean_loss": float(state.get("loss", float("nan"))),
            "external_median_error_km": external["median_error_km"],
            "external_mean_error_km": external["mean_error_km"],
            "external_acc_200km": external["acc_200km"],
        }
        if indist is not None:
            point["median_error_km"] = indist["median_error_km"]
            point["acc_25km"] = indist["acc_25km"]

        # Paired sign test against the pretrained baseline, once it exists.
        if baseline_external is not None:
            base = baseline_external["_errors"]
            errors = external["_errors"]
            closer = int(np.sum(errors < base - 1e-9))
            farther = int(np.sum(errors > base + 1e-9))
            decided = closer + farther
            point["external_sign_z"] = float(
                (closer - farther) / np.sqrt(decided) if decided else 0.0
            )
            point["closer"] = closer
            point["farther"] = farther
            point["identical"] = int(len(errors) - closer - farther)
            point["external_delta_median_km"] = (
                external["median_error_km"] - baseline_external["median_error_km"]
            )

        trajectory.append(point)
        print(f"  {entry['label']:<12} external {external['median_error_km']:>9.2f} km"
              + (f" | indist {indist['median_error_km']:>8.2f} km" if indist else "")
              + (f" | z {point.get('external_sign_z', 0.0):+.2f}" if "external_sign_z" in point else ""))

    # --- report ---
    trained = [p for p in trajectory if p["label"] not in ("pretrained",)]
    print("\n" + "=" * 78)
    print("  FULL trajectory, both metrics (pretrained = baseline row)")
    print("=" * 78)
    print(format_dual_trajectory(
        trajectory,
        external_baseline=baseline_external["median_error_km"] if baseline_external else None,
        indist_baseline=baseline_indist["median_error_km"] if baseline_indist else None,
    ))
    print()
    print(divergence_summary(trained))

    # --- re-derive both bests through the shared selection logic ---
    selectors = [
        CheckpointSelector(name="indist", metric_key="median_error_km",
                           description="in-distribution held-out median error (km)"),
        CheckpointSelector(name="external", metric_key="external_median_error_km",
                           drives_early_stop=True,
                           description="independent benchmark median error (km)"),
    ]
    chosen_source: dict[str, str] = {}
    for point in trained:
        for selector in selectors:
            def copy_from(filename: str, source=point["path"]) -> str:
                target = os.path.join(ckpt_dir, filename)
                if args.write_best:
                    shutil.copyfile(source, target)
                return target
            if selector.offer(point, epoch=point["epoch"], step=point["step"], save=copy_from):
                chosen_source[selector.name] = point["path"]

    print("\n" + "=" * 78)
    print("  Re-derived best checkpoints")
    print("=" * 78)
    for selector in selectors:
        if selector.epoch is None:
            continue
        point = next(p for p in trained if p["epoch"] == selector.epoch)
        print(f"\nbest_{selector.name}: epoch {selector.epoch} "
              f"({os.path.basename(chosen_source[selector.name])})")
        print(f"  in-distribution : {point.get('median_error_km', float('nan')):.2f} km median")
        print(f"  external        : {point['external_median_error_km']:.2f} km median "
              f"({point.get('external_delta_median_km', 0.0):+.2f} vs pretrained), "
              f"acc@200km {point['external_acc_200km']:.4f}")
        if "external_sign_z" in point:
            verdict = interpret({
                "sign_test_z": point["external_sign_z"],
                "closer": point["closer"],
                "farther": point["farther"],
                "identical": point["identical"],
                "delta_median_km": point.get("external_delta_median_km", 0.0),
            })
            print(f"  verdict         : {verdict}")
        if args.write_best:
            print(f"  written to      : {selector.filename}")

    out = os.path.join(work_dir, "rescore.json")
    os.makedirs(work_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({
            "run_name": args.run_name,
            "external_sample": args.external_sample,
            "baseline_external": (
                {k: v for k, v in baseline_external.items() if k != "_errors"}
                if baseline_external else None
            ),
            "baseline_indist": tb._clean(baseline_indist) if baseline_indist else None,
            "trajectory": trajectory,
            "selectors": [selector.as_dict() for selector in selectors],
            "divergence": divergence_summary(trained),
        }, handle, indent=2, default=str)
    print(f"\nrescore written to {out}")


if __name__ == "__main__":
    main()
