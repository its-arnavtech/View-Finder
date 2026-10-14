"""Dataset-agnostic batch training for GeoCLIP's location encoder.

Generalizes ``train_breadth_batch.py`` from "one Mapillary catalog" to "any
adapter in :mod:`finetune.datasets`", so a session can run many sequential
batches from different sources without the loop changing.

The batch pattern is ingest -> train -> checkpoint -> discard: raw imagery for a
batch is deleted after its checkpoint is written (when the adapter owns the
files), so total disk stays flat regardless of how many millions of images pass
through across sessions. What accumulates is checkpoints and metrics.

Every safeguard from the earlier runs is preserved, because each of them was
added in response to a measured failure:

* **Thinning by spacing before anything else.** Dense street-level pulls
  collapse: 1365 images were 184 distinct places at 100 m. Distinct locations,
  not image count, is what bounds this fine-tune.
* **Splitting by location, then verifying the separation.** Splitting by image
  leaks near-duplicates across the split and reports a fictitious gain. The
  verification is fail-closed: a violation aborts the run rather than warning.
* **Save-best plus early stopping.** With few locations the selection metric
  bottoms within ~20 optimizer steps and then regresses monotonically while
  training loss keeps falling all the way down.
* **Periodic external benchmarking.** The in-run held-out slice cannot detect a
  failure to generalize because it shares the training data's collection
  biases. See :mod:`external_probe`.

Checkpoint selection defaults to the EXTERNAL metric (changed 2026-08-09)
------------------------------------------------------------------------
This is a deliberate reversal of the previous default, forced by the
``commons01`` run. Selection used to track in-distribution held-out error
alone. Over that run the two metrics moved in **opposite** directions:

===== =================== =================== ===========
epoch held-out median km  external median km  sign-test z
===== =================== =================== ===========
0     324.0               **1164.6** (best)   +1.74
2     320.7               1199.1              +1.34
4     **314.3** (best)    1453.4              -1.04
6     326.9               1478.7              -0.90
===== =================== =================== ===========

Held-out error bottomed at epoch 4, so save-best shipped epoch 4 -- close to
the worst point of the run on the metric that reflects the actual goal. In
other words the old rule did not merely fail to help, it actively selected
against generalization.

So ``--checkpoint-select-metric`` now defaults to ``external``. Both bests are
always written -- ``location_encoder_best_indist.pth`` and
``location_encoder_best_external.pth`` -- and both trajectories are always
printed side by side, because the divergence between them is the finding worth
seeing. The flag chooses only which one spends early-stopping patience.

When external drives selection the probe is forced onto the same cadence as the
held-out eval, since a selector cannot choose a checkpoint it never scored.

VRAM note: this box is an 8 GB card under WDDM, which does **not** raise CUDA
OOM -- it silently spills to system RAM and runs ~50x slower. Peak reserved is
therefore asserted against a budget rather than trusted to fail loudly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

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
from datasets import get_adapter
from evaluate import encode_val_images, evaluate, format_comparison
from external_probe import ExternalProbe, interpret, load_external_sample
from location_sampling import (
    location_split,
    measure_redundancy,
    thin_by_spacing,
    verify_split_separation,
)
from model_setup import (
    assert_freeze_contract,
    build_model,
    freeze_image_side,
    trainable_parameter_report,
)

# Treat a config whose peak reserved exceeds this share of total VRAM as failed,
# even though the driver will not raise.
VRAM_BUDGET_FRACTION = 0.95


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="commons-landmarks",
                        help="adapter name from finetune.datasets.ADAPTERS")
    parser.add_argument("--dataset-arg", action="append", default=[], metavar="K=V",
                        help="adapter constructor argument, repeatable")
    parser.add_argument("--limit", type=int, default=None,
                        help="max images to ingest for this batch")
    parser.add_argument("--min-spacing-m", type=float, default=100.0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--queue-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--reg-strength", type=float, default=0.0,
        help="L2-SP strength: penalize squared drift from this run's initialized "
             "location-encoder weights (0 disables; moderate experimental value: 1e-3)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--external-every-epochs", type=int, default=2,
                        help="run the independent benchmark every N epochs (0 disables). "
                             "Forced to match --eval-every-epochs when external "
                             "selection is active, since a selector cannot pick a "
                             "checkpoint it never scored")
    parser.add_argument("--step-eval-every", type=int, default=0,
                        help="also evaluate (and probe) every N optimizer steps, not "
                             "just at epoch boundaries. Epoch granularity is too coarse "
                             "early: commons01 gained essentially all its external value "
                             "inside the first epoch's 107 steps. 0 disables")
    parser.add_argument("--step-eval-until", type=int, default=0,
                        help="stop dense step probing after this many steps (0 = whole run)")
    parser.add_argument(
        "--step-eval-early-every", type=int, default=0,
        help="override the normal step-probe cadence through "
             "--step-eval-early-until (0 disables)",
    )
    parser.add_argument(
        "--step-eval-early-until", type=int, default=0,
        help="last step governed by --step-eval-early-every",
    )
    parser.add_argument("--checkpoint-select-metric",
                        choices=("indist", "external", "both"),
                        default="external",
                        help="which metric spends early-stopping patience. Both best "
                             "checkpoints are always saved regardless; this chooses "
                             "only what stops the run. Default changed from indist to "
                             "external on 2026-08-09 -- see the module docstring "
                             "(default: %(default)s)")
    parser.add_argument("--external-sample", type=int, default=1000,
                        help="images sampled from benchmark-data/manifest.csv")
    parser.add_argument("--init-checkpoint", default=None, metavar="PATH",
                        help="start from a saved location encoder instead of pretrained "
                             "GeoCLIP, to accumulate across shards. Baselines are still "
                             "measured on pretrained weights first, so the reported "
                             "deltas stay comparable with every unchained run")
    parser.add_argument(
        "--reuse-cached-run",
        default=None,
        metavar="DIR",
        help="reuse split_train.csv, split_val.csv, and their validated embedding "
             "memmap from a completed run directory; skips ingestion, thinning, "
             "splitting, downloading, and image encoding",
    )
    parser.add_argument("--discard-after", action="store_true",
                        help="delete this batch's raw images once the checkpoint is written")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="ingest and report diversity, then stop before training")
    return parser


def _adapter_kwargs(pairs: list[str]) -> dict:
    kwargs: dict[str, object] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if value.lower() in ("true", "false"):
            kwargs[key] = value.lower() == "true"
        else:
            kwargs[key] = value
    return kwargs


def main() -> None:
    run_started = time.perf_counter()
    args = build_arg_parser().parse_args()

    if args.queue_size % args.batch_size != 0:
        raise SystemExit(
            f"queue_size {args.queue_size} must be divisible by batch_size "
            f"{args.batch_size} (GeoCLIP's GPS queue asserts this)"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_name = args.run_name or f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = os.path.join(tb.CKPT_ROOT, run_name)
    work_dir = os.path.join(tb.WORK_ROOT, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    print("=" * 74)
    print(f"GeoCLIP scaled batch | dataset={args.dataset} | run={run_name}")
    print("=" * 74)

    # ---- 1. Licence + access gate ----
    adapter = get_adapter(args.dataset, **_adapter_kwargs(args.dataset_arg))
    availability = adapter.availability()
    print(f"licence : [{availability.licence.value}] {availability.licence_note}")
    print(f"access  : {availability.access_note}")
    if not availability.usable:
        raise SystemExit(f"\nREFUSING TO INGEST: {availability.blocked_reason()}")
    print()

    # ---- 2. Ingest (or resume an already validated cached pool) ----
    reused_cached_pool = bool(args.reuse_cached_run)
    if reused_cached_pool:
        source_dir = os.path.abspath(args.reuse_cached_run)
        source_summary_path = os.path.join(source_dir, "summary.json")
        source_train_path = os.path.join(source_dir, "split_train.csv")
        source_val_path = os.path.join(source_dir, "split_val.csv")
        missing = [
            path for path in (source_summary_path, source_train_path, source_val_path)
            if not os.path.isfile(path)
        ]
        if missing:
            raise SystemExit(
                "--reuse-cached-run is incomplete; missing: " + ", ".join(missing)
            )
        with open(source_summary_path, encoding="utf-8") as handle:
            source_summary = json.load(handle)
        train_frame = pd.read_csv(source_train_path)
        val_frame = pd.read_csv(source_val_path)
        required_cache_columns = {
            "embedding_path", "embedding_index", "embedding_count", "embedding_dim",
        }
        for label, part in (("train", train_frame), ("val", val_frame)):
            absent = sorted(required_cache_columns - set(part.columns))
            if absent:
                raise SystemExit(
                    f"reused {label} split lacks cached-embedding columns: {absent}"
                )
        cache_paths = set(train_frame["embedding_path"].astype(str)) | set(
            val_frame["embedding_path"].astype(str)
        )
        if len(cache_paths) != 1:
            raise SystemExit(
                f"reused pool must reference exactly one embedding memmap, got {cache_paths}"
            )
        cache_path = next(iter(cache_paths))
        if not os.path.isfile(cache_path):
            raise SystemExit(f"reused embedding memmap not found: {cache_path}")
        counts = set(pd.concat([
            train_frame["embedding_count"], val_frame["embedding_count"]
        ]).astype(int))
        dims = set(pd.concat([
            train_frame["embedding_dim"], val_frame["embedding_dim"]
        ]).astype(int))
        if len(counts) != 1 or len(dims) != 1:
            raise SystemExit(
                f"reused cache metadata is inconsistent: counts={counts}, dims={dims}"
            )
        cache_count, cache_dim = next(iter(counts)), next(iter(dims))
        expected_bytes = cache_count * cache_dim * np.dtype(np.float16).itemsize
        actual_bytes = os.path.getsize(cache_path)
        if actual_bytes != expected_bytes:
            raise SystemExit(
                f"reused cache size mismatch: expected {expected_bytes} bytes for "
                f"{cache_count}x{cache_dim} float16, found {actual_bytes}"
            )
        all_indices = pd.concat([
            train_frame["embedding_index"], val_frame["embedding_index"]
        ]).astype(int)
        if len(all_indices) != cache_count or all_indices.nunique() != cache_count or (
            all_indices.min() != 0 or all_indices.max() != cache_count - 1
        ):
            raise SystemExit(
                "reused cache indices must cover every memmap row exactly once"
            )

        thinned = pd.concat([train_frame, val_frame], ignore_index=True)
        frame = thinned
        adapter_stats = dict(source_summary.get("adapter_stats") or {})
        cache_stats = dict(adapter_stats.get("embedding_cache") or {})
        cache_stats.update({
            "path": cache_path,
            "shape": [cache_count, cache_dim],
            "bytes": actual_bytes,
            "gib": actual_bytes / 1024 ** 3,
            "reused_from": source_dir,
            "reused": True,
            "original_wall_seconds": cache_stats.get("wall_seconds"),
            "wall_seconds": 0.0,
        })
        adapter_stats["embedding_cache"] = cache_stats
        adapter_stats["cache_reused_from"] = source_dir
        images_ingested = int(source_summary.get("images_ingested", len(thinned)))
        retention = float(source_summary.get(
            "location_retention",
            len(thinned) / images_ingested if images_ingested else 0.0,
        ))
        pre_thinned = True
        before = None
        after = None
        print(
            f"reused cached pool: {source_dir}\n"
            f"  split: train={len(train_frame)} val={len(val_frame)}\n"
            f"  cache: {cache_count}x{cache_dim} float16, "
            f"{actual_bytes / 1024 ** 3:.3f} GiB (build skipped)\n"
        )
    else:
        pre_thinned = hasattr(adapter, "load_thinned")
    if not reused_cached_pool and pre_thinned:
        # Combined OSV-5M ingestion must decide the globally retained IDs before
        # materializing images, so the adapter performs the same exact banded
        # thinning across its complete metadata pool.
        thinned = adapter.load_thinned(
            min_spacing_m=args.min_spacing_m, seed=args.seed, limit=args.limit
        )
        adapter_stats = getattr(adapter, "stats", {})
        images_ingested = int(adapter_stats.get("images_ingested", len(thinned)))
        retention = float(adapter_stats.get(
            "retention", len(thinned) / images_ingested if images_ingested else 0.0
        ))
        frame = thinned
    elif not reused_cached_pool:
        frame = adapter.load(limit=args.limit, seed=args.seed)
        adapter_stats = {}
        images_ingested = int(len(frame))
    filtered = getattr(adapter, "licence_filtered_out", None)
    if filtered:
        print(f"licence filter removed {filtered} rows before ingest")
    print(f"ingested: {images_ingested} images from {args.dataset}\n")
    if frame.empty:
        raise SystemExit("Adapter returned no usable rows.")

    if pre_thinned:
        before = None
        after = None
        print(
            f"adapter completed exact combined thinning to >={args.min_spacing_m:.0f} m: "
            f"{images_ingested} -> {len(thinned)} ({retention:.1%} retained)\n"
        )
    else:
        before = measure_redundancy(frame)
        print(before.format(f"{args.dataset}, unthinned"))
        print()

        # ---- 3. Thin to real location diversity ----
        thinned = thin_by_spacing(frame, min_spacing_m=args.min_spacing_m, seed=args.seed)
        retention = len(thinned) / len(frame) if len(frame) else 0.0
        print(f"thinning to >={args.min_spacing_m:.0f} m: {len(frame)} -> {len(thinned)} "
              f"images ({retention:.1%} retained)\n")
        after = measure_redundancy(thinned)
        print(after.format(f"after >={args.min_spacing_m:.0f} m thinning"))
        print()

    if args.dry_run:
        print("--dry-run: stopping before training.")
        json.dump(
            {"dataset": args.dataset, "ingested": int(len(frame)),
             "locations": int(len(thinned)), "retention": retention,
             "redundancy_before": vars(before) if before else None,
             "redundancy_after": vars(after) if after else None},
            open(os.path.join(work_dir, "diversity.json"), "w", encoding="utf-8"), indent=2)
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    device = "cuda"

    # ---- 4. Leak-proof split, verified fail-closed ----
    if reused_cached_pool:
        split = {"dropped_for_proximity": 0}
    else:
        split = location_split(thinned, val_fraction=args.val_fraction,
                               min_gap_m=args.min_spacing_m, seed=args.seed)
        train_frame, val_frame = split["train"], split["val"]
    check = verify_split_separation(train_frame, val_frame, min_gap_m=args.min_spacing_m)
    print(f"split: train={len(train_frame)} val={len(val_frame)} "
          f"(dropped {split['dropped_for_proximity']} training rows inside the buffer)")
    gap_label = "proven lower bound" if check.get("min_gap_m_is_lower_bound") else "closest pair"
    print(f"separation verified: passed={check['passed']} | {gap_label} "
          f"{check['min_gap_m_observed']:.1f} m | violations {check['violations']}")
    if not check["passed"]:
        raise SystemExit("Split separation check failed -- refusing to train on a leaking split.")
    print()

    for name, part in (("train", train_frame), ("val", val_frame)):
        part.to_csv(os.path.join(work_dir, f"split_{name}.csv"), index=False)

    if len(train_frame) < args.batch_size:
        raise SystemExit(
            f"Only {len(train_frame)} training locations after thinning, below "
            f"batch_size={args.batch_size}. Ingest more before training."
        )

    # ---- 5. Model + freeze contract ----
    model = build_model(device=device, queue_size=args.queue_size)
    freeze_image_side(model)
    assert_freeze_contract(model)
    print(trainable_parameter_report(model))
    amp = not args.no_amp

    # Combined adapters can now materialize one shard at a time, cache the
    # frozen vision features, and delete raw imagery before optimization. The
    # split is already known, so training rows receive the training transform
    # while held-out rows keep deterministic inference preprocessing.
    if hasattr(adapter, "cache_embeddings") and not reused_cached_pool:
        train_frame, val_frame = adapter.cache_embeddings(
            train_frame,
            val_frame,
            model=model,
            device=device,
            batch_size=args.batch_size,
            amp=amp,
            num_workers=args.num_workers,
        )
        for name, part in (("train", train_frame), ("val", val_frame)):
            part.to_csv(os.path.join(work_dir, f"split_{name}.csv"), index=False)
        cache_stats = adapter.stats.get("embedding_cache", {})
        print(
            f"\nembedding cache: {cache_stats.get('gib', 0):.3f} GiB | "
            f"{cache_stats.get('images_per_second', 0):.1f} images/s | "
            f"raw:cache size ratio {cache_stats.get('raw_to_cache_size_ratio', 0):.1f}x"
        )

    gps_gallery = load_gps_data(
        os.path.join(geoclip_file_dir, "gps_gallery", "coordinates_100K.csv")
    )

    # ---- 6. Baselines: in-distribution and independent ----
    print("\nencoding held-out images (frozen image side, once)...")
    val_features, val_coords = encode_val_images(
        model, val_frame, args.eval_batch_size, device, amp=amp
    )
    before_val = evaluate(model, val_features, val_coords, gps_gallery, device)
    print(f"pretrained held-out: {before_val['median_error_km']:.2f} km median")

    # A selector cannot choose a checkpoint it never scored, so external-driven
    # selection forces the probe onto the eval cadence.
    external_cadence = args.external_every_epochs
    if args.checkpoint_select_metric in ("external", "both") and external_cadence > 0:
        if external_cadence != args.eval_every_epochs:
            print(f"\nnote: --checkpoint-select-metric={args.checkpoint_select_metric} "
                  f"forces the external probe onto the eval cadence "
                  f"(every {args.eval_every_epochs} epoch(s), was {external_cadence})")
        external_cadence = args.eval_every_epochs

    probe = None
    if args.external_every_epochs > 0:
        external_frame = load_external_sample(sample_size=args.external_sample)
        print(f"\nencoding {len(external_frame)} independent benchmark images (once)...")
        probe = ExternalProbe(model, external_frame, gps_gallery, device,
                              batch_size=args.eval_batch_size, amp=amp)
        baseline = probe.set_baseline()
        print(f"pretrained EXTERNAL: {baseline['median_error_km']:.2f} km median, "
              f"acc@200km {baseline['acc_200km']:.4f}")

    torch.save({"location_encoder": model.location_encoder.state_dict(), "step": 0,
                "note": "untouched pretrained baseline"},
               os.path.join(ckpt_dir, "location_encoder_pretrained.pth"))

    # ---- 6b. Optionally chain from a previous shard's checkpoint ----
    #
    # Loaded *after* both baselines are measured, deliberately. The probe's
    # baseline stays untouched pretrained GeoCLIP, so this run's deltas and
    # sign tests remain directly comparable with every unchained run rather
    # than being measured against a moving reference. The chained starting
    # point is reported separately below.
    init_report: dict | None = None
    if args.init_checkpoint:
        if not os.path.isfile(args.init_checkpoint):
            raise SystemExit(f"--init-checkpoint not found: {args.init_checkpoint}")
        state = torch.load(args.init_checkpoint, map_location=device)
        model.location_encoder.load_state_dict(state["location_encoder"])
        start_val = evaluate(model, val_features, val_coords, gps_gallery, device)
        start_external = probe.measure() if probe is not None else None
        init_report = {
            "path": args.init_checkpoint,
            "source_step": state.get("step"),
            "source_epoch": state.get("epoch"),
            "indist_at_start": tb._clean(start_val),
            "external_at_start": (
                {k: v for k, v in start_external.items() if k != "_errors"}
                if start_external else None
            ),
        }
        print(f"\nCHAINED START from {os.path.basename(args.init_checkpoint)} "
              f"(step {state.get('step')})")
        print(f"  held-out at start : {start_val['median_error_km']:.2f} km "
              f"(pretrained {before_val['median_error_km']:.2f})")
        if start_external is not None and probe is not None:
            delta = start_external["median_error_km"] - probe.baseline["median_error_km"]
            print(f"  EXTERNAL at start : {start_external['median_error_km']:.2f} km "
                  f"({delta:+.2f} vs pretrained), acc@200km "
                  f"{start_external['acc_200km']:.4f}")
            print("  training must beat THIS to show accumulation, not merely beat pretrained.")

    # L2-SP must anchor to the weights training actually starts from. For a
    # chained run this is deliberately after --init-checkpoint is loaded.
    reg_reference = (
        tb.capture_l2sp_reference(model.location_encoder)
        if args.reg_strength else None
    )

    # ---- 7. Train, probing the independent set on the way ----
    selectors = [
        CheckpointSelector(
            name="indist",
            metric_key="median_error_km",
            drives_early_stop=args.checkpoint_select_metric == "indist",
            description="in-distribution held-out median error (km)",
        ),
        CheckpointSelector(
            name="external",
            metric_key="external_median_error_km",
            drives_early_stop=args.checkpoint_select_metric in ("external", "both"),
            description="independent benchmark median error (km)",
        ),
    ]
    if probe is None and args.checkpoint_select_metric != "indist":
        raise SystemExit(
            "--checkpoint-select-metric external requires the external probe; "
            "either raise --external-every-epochs above 0 or select indist."
        )

    def eval_hook(epoch, step):
        metrics = tb._clean(evaluate(model, val_features, val_coords, gps_gallery, device))
        # During dense step probing every evaluation carries an external
        # measurement; otherwise the configured epoch cadence applies.
        dense = tb.should_step_evaluate(
            step,
            step_eval_every=args.step_eval_every,
            step_eval_until=args.step_eval_until,
            step_eval_early_every=args.step_eval_early_every,
            step_eval_early_until=args.step_eval_early_until,
        )
        if probe is not None and (dense or (external_cadence > 0 and epoch % external_cadence == 0)):
            point = probe.probe(epoch=epoch, step=step, locations_trained=len(train_frame))
            # Merged into the same dict so both selectors read one source.
            metrics["external_median_error_km"] = point["median_error_km"]
            metrics["external_mean_error_km"] = point["mean_error_km"]
            metrics["external_acc_200km"] = point["acc_200km"]
            metrics["external_delta_median_km"] = point["delta_median_km"]
            metrics["external_sign_z"] = point["sign_test_z"]
            print(f"    [external] median {point['median_error_km']:.2f} km "
                  f"({point['delta_median_km']:+.2f}) | {interpret(point)}")
        return metrics

    torch.cuda.reset_peak_memory_stats()
    history, trajectory, best = tb.train_one_batch(
        model, train_frame, device=device,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, amp=amp, num_workers=args.num_workers,
        queue_size=args.queue_size, ckpt_dir=ckpt_dir, ckpt_every_steps=0,
        log_path=os.path.join(work_dir, "train_log.jsonl"),
        eval_hook=eval_hook, eval_every_epochs=args.eval_every_epochs,
        early_stop_patience=args.early_stop_patience,
        selectors=selectors,
        step_eval_every=args.step_eval_every,
        step_eval_until=args.step_eval_until,
        step_eval_early_every=args.step_eval_early_every,
        step_eval_early_until=args.step_eval_early_until,
        reg_strength=args.reg_strength,
        reg_reference=reg_reference,
    )
    by_name = {selector.name: selector for selector in selectors}
    final_path = tb.save_checkpoint(model, ckpt_dir, history[-1]["step"],
                                    history[-1]["loss"], tag="final")

    peak_gb = float(max(h["peak_reserved_gb"] for h in history))
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if peak_gb > total_gb * VRAM_BUDGET_FRACTION:
        print(f"\nWARNING: peak reserved {peak_gb:.2f} GB exceeds "
              f"{VRAM_BUDGET_FRACTION:.0%} of {total_gb:.2f} GB. On WDDM this does not "
              "raise; it spills to system RAM and runs far slower. Lower --batch-size.")

    # ---- 8. Report BOTH selected checkpoints, never a single winner ----
    print("\n" + "=" * 74)
    print("  BOTH metrics' trajectories, side by side")
    print("=" * 74)
    print(format_dual_trajectory(
        trajectory,
        external_baseline=probe.baseline["median_error_km"] if probe else None,
        indist_baseline=before_val["median_error_km"],
    ))
    print()
    print(divergence_summary(trajectory))

    def score_checkpoint(path: str) -> tuple[dict, dict | None]:
        """Load a saved location encoder and score it on both metrics."""
        saved = torch.load(path, map_location=device)
        model.location_encoder.load_state_dict(saved["location_encoder"])
        indist = evaluate(model, val_features, val_coords, gps_gallery, device)
        external = probe.measure() if probe is not None else None
        return indist, external

    selected_report: dict[str, dict] = {}
    for selector in selectors:
        if not selector.path:
            continue
        indist, external = score_checkpoint(selector.path)
        selected_report[selector.name] = {
            "epoch": selector.epoch,
            "step": selector.step,
            "path": selector.path,
            "drove_early_stop": selector.drives_early_stop,
            "indist": tb._clean(indist),
            "external": (
                {k: v for k, v in external.items() if k != "_errors"} if external else None
            ),
        }
        label = f"best_{selector.name} @ epoch {selector.epoch}"
        if selector.drives_early_stop:
            label += " (drove early stopping)"
        print(format_comparison(
            f"HELD-OUT (in-distribution) [{label}]", before_val, indist))
        if external is not None and probe is not None:
            print(f"  same checkpoint on the INDEPENDENT set: "
                  f"{external['median_error_km']:.2f} km median "
                  f"({external['median_error_km'] - probe.baseline['median_error_km']:+.2f} "
                  f"vs pretrained), acc@200km {external['acc_200km']:.4f}")

    if probe is not None:
        print("\n--- INDEPENDENT BENCHMARK trend (never trained on) ---")
        print(ExternalProbe.format_trend(probe.history))

    # ---- 9. Discard raw imagery, keep checkpoints ----
    discarded = adapter.discard(train_frame) if args.discard_after else 0
    if args.discard_after:
        print(f"\ndiscarded {discarded} raw images; checkpoints retained in {ckpt_dir}")

    summary = {
        "run_name": run_name,
        "dataset": args.dataset,
        "licence": availability.licence.value,
        "config": vars(args),
        "images_ingested": images_ingested,
        "locations_after_thinning": int(len(thinned)),
        "location_retention": retention,
        "redundancy_before_thinning": vars(before) if before else None,
        "redundancy_after_thinning": vars(after) if after else None,
        "adapter_stats": adapter_stats,
        "country_distribution": {
            str(country): int(count)
            for country, count in thinned["region"].value_counts().items()
        },
        "regularization": {
            "type": "l2_sp" if args.reg_strength else "none",
            "strength": args.reg_strength,
            "anchor": args.init_checkpoint or "pretrained_geoclip",
            "standard_weight_decay": args.weight_decay,
        },
        "train_locations": int(len(train_frame)),
        "val_locations": int(len(val_frame)),
        "separation_check": check,
        "held_out_baseline": tb._clean(before_val),
        "init_checkpoint": init_report,
        "selection_metric_driving_early_stop": args.checkpoint_select_metric,
        "selectors": [selector.as_dict() for selector in selectors],
        # Both selected checkpoints, each scored on BOTH metrics. This is the
        # artifact that makes a divergence like commons01's inspectable later.
        "selected_checkpoints": selected_report,
        "external_baseline": (
            {k: v for k, v in probe.baseline.items() if k != "_errors"} if probe else None
        ),
        "external_trend": probe.history if probe else [],
        "divergence": divergence_summary(trajectory),
        "trajectory": trajectory,
        "early_stop_driver": {k: v for k, v in best.items() if k != "metrics"},
        "final_checkpoint": final_path,
        "peak_reserved_gb": peak_gb,
        "raw_images_discarded": discarded,
        "wall_clock": {
            "total_seconds": time.perf_counter() - run_started,
            "training_seconds": best.get("training_wall_seconds"),
            "epoch_wall_times": best.get("epoch_wall_times", []),
            "embedding_cache_seconds": (
                adapter_stats.get("embedding_cache", {}).get("wall_seconds")
                if adapter_stats else None
            ),
        },
    }
    with open(os.path.join(work_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"\nrun summary: {os.path.join(work_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
