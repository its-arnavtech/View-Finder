"""Fine-tune on the worldwide breadth batch, split by location.

Differs from train_batch.py in how the data is prepared, not how it trains:

* Source is the extract-stage catalog (no gallery artifacts are built).
* The batch is thinned to a minimum spacing so effective location diversity
  matches the raw image count, instead of collapsing 1365 images into 184 places.
* The held-out slice is separated *by distance*, and every training row within
  the buffer of a held-out row is dropped, then the separation is verified.

Everything downstream -- the freeze contract, the frozen-CLIP contrastive step,
best-checkpoint selection, early stopping -- is reused from train_batch.py.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from geoclip.model.misc import file_dir as geoclip_file_dir
from geoclip.model.misc import load_gps_data

import train_batch as tb
from dataset_index import build_index, build_index_from_catalog
from evaluate import encode_val_images, evaluate, format_comparison
from location_sampling import (
    location_split,
    measure_redundancy,
    thin_by_spacing,
    verify_split_separation,
)
from model_setup import assert_freeze_contract, build_model, freeze_image_side, trainable_parameter_report

BREADTH_CATALOG = r"C:\View-Finder\gallery-pipeline\data\finetune-breadth\catalog.db"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=str, default=BREADTH_CATALOG)
    parser.add_argument("--min-spacing-m", type=float, default=100.0,
                        help="minimum spacing between retained locations")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--include-existing", action="store_true",
                        help="also fold in the existing gallery-pipeline pool")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--queue-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    if args.queue_size % args.batch_size != 0:
        raise SystemExit(f"queue_size {args.queue_size} must be divisible by batch_size {args.batch_size}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    run_name = args.run_name or f"breadth_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = os.path.join(tb.CKPT_ROOT, run_name)
    work_dir = os.path.join(tb.WORK_ROOT, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    print("=" * 70)
    print(f"GeoCLIP breadth-batch fine-tune | run: {run_name}")
    print("=" * 70)

    # ---- 1. Assemble the batch ----
    frame, stats = build_index_from_catalog(args.catalog, dataset_name="finetune-breadth")
    print(f"breadth pull : {stats.present_on_disk} images / {stats.distinct_tiles} tiles")

    if args.include_existing:
        existing, ex_stats = build_index()
        print(f"existing pool: {ex_stats.present_on_disk} images / {ex_stats.distinct_tiles} tiles")
        frame = pd.concat([frame, existing], ignore_index=True)
        frame = frame.drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    print(f"combined     : {len(frame)} images\n")

    before = measure_redundancy(frame)
    print(before.format("breadth batch, unthinned"))
    print()

    # ---- 2. Thin to real location diversity ----
    thinned = thin_by_spacing(frame, min_spacing_m=args.min_spacing_m, seed=args.seed)
    print(f"thinning to >={args.min_spacing_m:.0f} m: {len(frame)} -> {len(thinned)} images "
          f"({len(thinned) / len(frame):.1%} retained)\n")
    after = measure_redundancy(thinned)
    print(after.format(f"after >={args.min_spacing_m:.0f} m thinning"))
    print()

    # ---- 3. Leak-proof split, verified ----
    split = location_split(thinned, val_fraction=args.val_fraction,
                           min_gap_m=args.min_spacing_m, seed=args.seed)
    train_frame, val_frame = split["train"], split["val"]
    check = verify_split_separation(train_frame, val_frame, min_gap_m=args.min_spacing_m)
    print(f"split: train={len(train_frame)} val={len(val_frame)} "
          f"(dropped {split['dropped_for_proximity']} training rows inside the buffer)")
    print(f"separation verified: passed={check['passed']} | closest train/val pair "
          f"{check['min_gap_m_observed']:.1f} m | violations {check['violations']}")
    if not check["passed"]:
        raise SystemExit("Split separation check failed -- refusing to train on a leaking split.")
    print()

    for name, f in (("train", train_frame), ("val", val_frame)):
        f.to_csv(os.path.join(work_dir, f"split_{name}.csv"), index=False)

    if len(train_frame) < args.batch_size:
        raise SystemExit(
            f"Only {len(train_frame)} training locations after thinning, "
            f"below batch_size={args.batch_size}. Pull more tiles before training."
        )

    # ---- 4. Model + freeze contract ----
    model = build_model(device=device, queue_size=args.queue_size)
    freeze_image_side(model)
    assert_freeze_contract(model)
    print(trainable_parameter_report(model))

    gps_gallery = load_gps_data(os.path.join(geoclip_file_dir, "gps_gallery", "coordinates_100K.csv"))
    amp = not args.no_amp

    # ---- 5. Baseline ----
    print("\nencoding held-out images (frozen image side, computed once)...")
    val_features, val_coords = encode_val_images(model, val_frame, args.eval_batch_size, device, amp=amp)
    print("evaluating PRETRAINED baseline...")
    before_val = evaluate(model, val_features, val_coords, gps_gallery, device)

    torch.save({"location_encoder": model.location_encoder.state_dict(), "step": 0,
                "note": "untouched pretrained baseline"},
               os.path.join(ckpt_dir, "location_encoder_pretrained.pth"))

    # ---- 6. Train ----
    def eval_hook(_epoch, _step):
        return tb._clean(evaluate(model, val_features, val_coords, gps_gallery, device))

    torch.cuda.reset_peak_memory_stats()
    history, trajectory, best = tb.train_one_batch(
        model,
        train_frame,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=amp,
        num_workers=args.num_workers,
        queue_size=args.queue_size,
        ckpt_dir=ckpt_dir,
        ckpt_every_steps=0,
        log_path=os.path.join(work_dir, "train_log.jsonl"),
        eval_hook=eval_hook,
        eval_every_epochs=args.eval_every_epochs,
        early_stop_patience=args.early_stop_patience,
    )

    final_path = tb.save_checkpoint(model, ckpt_dir, history[-1]["step"], history[-1]["loss"], tag="final")

    # ---- 7. Report against the best checkpoint ----
    after_val = evaluate(model, val_features, val_coords, gps_gallery, device)
    selected = "final"
    if best.get("path") and best.get("step") != history[-1]["step"]:
        state = torch.load(best["path"], map_location=device)
        model.location_encoder.load_state_dict(state["location_encoder"])
        after_val = evaluate(model, val_features, val_coords, gps_gallery, device)
        selected = f"best@epoch{best['epoch']}"

    print(format_comparison(f"HELD-OUT: >={args.min_spacing_m:.0f} m separated locations [{selected}]",
                            before_val, after_val))

    if best.get("path"):
        print(f"\nBEST held-out state: epoch {best['epoch']} (step {best['step']}), "
              f"{best['value']:.2f} km median -> {best['path']}")
        if len(trajectory) < args.epochs:
            print(f"early stopping triggered: ran {len(trajectory)} of {args.epochs} requested epochs")

    print("\n--- held-out trajectory ---")
    print(f"{'epoch':>6} {'step':>6} {'train_loss':>11} {'median_km':>11} {'acc@25km':>10}")
    for point in trajectory:
        print(f"{point['epoch']:>6} {point['step']:>6} {point['mean_loss']:>11.5f} "
              f"{point['median_error_km']:>11.2f} {point['acc_25km']:>10.4f}")

    summary = {
        "run_name": run_name,
        "config": vars(args),
        "redundancy_before_thinning": vars(before),
        "redundancy_after_thinning": vars(after),
        "images_before_thinning": int(len(frame)),
        "locations_after_thinning": int(len(thinned)),
        "train_locations": int(len(train_frame)),
        "val_locations": int(len(val_frame)),
        "separation_check": check,
        "val": {"before": tb._clean(before_val), "after": tb._clean(after_val)},
        "trajectory": trajectory,
        "selected_checkpoint": selected,
        "best_checkpoint": {k: v for k, v in best.items() if k != "metrics"},
        "final_checkpoint": final_path,
        "peak_reserved_gb": float(max(h["peak_reserved_gb"] for h in history)),
    }
    with open(os.path.join(work_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nrun summary: {os.path.join(work_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
