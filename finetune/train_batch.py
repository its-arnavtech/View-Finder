"""Batch-ingest-train-discard fine-tuning for GeoCLIP's location encoder.

One invocation processes exactly one batch of geotagged data:

    stage batch -> eval pretrained baseline -> train -> checkpoint -> eval
    fine-tuned -> report before/after -> (optionally) discard the batch's raw data

Only the location encoder is updated. The CLIP visual tower and its projection
MLP are frozen, so the batch's images are pushed through the image side under
no_grad purely to produce target embeddings for the contrastive loss.

The persisting artifact across batches is the location encoder checkpoint. Raw
image data is disposable by design -- but see `purge.py` for why that deletion
refuses to touch source datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from geoclip.model.misc import file_dir as geoclip_file_dir
from geoclip.model.misc import load_gps_data
from clip_transforms import clip_train_transform

from dataset_index import build_index
from evaluate import encode_val_images, evaluate, format_comparison
from model_setup import assert_freeze_contract, build_model, freeze_image_side, trainable_parameter_report
from purge import purge_batch
from split import make_split

# Input shapes are fixed (224x224, drop_last=True), so cuDNN can cache its
# algorithm choice. Marginal for a ViT -- only the patch-embedding conv benefits
# -- but free. Suggested by `gpuboost analyze`.
torch.backends.cudnn.benchmark = True

WORK_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")
CKPT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


class GeoTrainDataset(Dataset):
    """(augmented image tensor, [lat, lon]) pairs for contrastive training."""

    def __init__(self, frame):
        self.paths = frame["image_path"].tolist()
        self.coords = frame[["lat", "lon"]].to_numpy(dtype=np.float32)
        self.transform = clip_train_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), torch.tensor(self.coords[idx], dtype=torch.float32)


def train_one_batch(
    model,
    frame,
    *,
    device: str,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    num_workers: int,
    queue_size: int,
    ckpt_dir: str,
    ckpt_every_steps: int,
    log_path: str,
    eval_hook=None,
    eval_every_epochs: int = 0,
    early_stop_patience: int = 0,
    select_metric: str = "median_error_km",
) -> tuple[list[dict], list[dict], dict]:
    """Run the contrastive training loop over one batch of data.

    `eval_hook` is called every `eval_every_epochs` epochs with (epoch, step) and
    should return a metrics dict. Held-out image features are already cached, so
    these mid-training evaluations are cheap and show whether the run is actually
    converging rather than merely running.

    Training loss on this data keeps falling long after held-out error starts
    regressing, so the final weights are not the best weights. `select_metric`
    (lower is better) is tracked across evaluations, the best state is saved
    separately as location_encoder_best.pth, and `early_stop_patience`
    consecutive non-improving evaluations end the run.
    """
    dataset = GeoTrainDataset(frame)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # the GPS queue requires a fixed batch size
        persistent_workers=num_workers > 0,
    )
    if len(loader) == 0:
        raise RuntimeError(
            f"Batch of {len(dataset)} images is smaller than batch_size={batch_size} with drop_last=True."
        )

    optimizer = torch.optim.AdamW(
        (p for p in model.location_encoder.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    total_steps = len(loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()
    targets = torch.arange(batch_size, device=device)

    os.makedirs(ckpt_dir, exist_ok=True)
    history: list[dict] = []
    trajectory: list[dict] = []
    best = {"value": float("inf"), "epoch": None, "step": None, "path": None, "metrics": None}
    stale_evals = 0
    global_step = 0
    log_file = open(log_path, "a", encoding="utf-8")

    print(f"\ntraining: {len(dataset)} images | {len(loader)} steps/epoch | "
          f"{epochs} epochs | {total_steps} total steps | batch {batch_size} | amp={amp}")
    print(f"{'step':>6} {'epoch':>6} {'loss':>10} {'lr':>10} {'peak_GB':>9} {'sec':>7}")
    print("-" * 54)

    model.train()
    model.image_encoder.eval()  # keep the frozen tower in eval mode
    started = time.perf_counter()

    for epoch in range(epochs):
        epoch_losses = []
        for imgs, gps in loader:
            step_start = time.perf_counter()
            imgs = imgs.to(device, non_blocking=True)
            gps = gps.to(device, non_blocking=True)

            # GPS queue supplies extra negatives beyond the in-batch ones.
            gps_queue = model.get_gps_queue()
            gps_all = torch.cat([gps, gps_queue], dim=0)
            model.dequeue_and_enqueue(gps)

            optimizer.zero_grad(set_to_none=True)

            # Frozen image side: no graph, no stored activations.
            with torch.no_grad():
                if amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        image_features = model.image_encoder(imgs)
                    image_features = image_features.float()
                else:
                    image_features = model.image_encoder(imgs)

            # Trainable location encoder.
            location_features = model.location_encoder(gps_all)

            logits = model.logit_scale.exp() * (
                F.normalize(image_features, dim=1) @ F.normalize(location_features, dim=1).t()
            )
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            record = {
                "step": global_step,
                "epoch": epoch,
                "loss": loss_value,
                "lr": scheduler.get_last_lr()[0],
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
                "sec": time.perf_counter() - step_start,
            }
            history.append(record)
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            if ckpt_every_steps and global_step % ckpt_every_steps == 0:
                save_checkpoint(model, ckpt_dir, global_step, loss_value, tag="step")

            if global_step % 5 == 0 or global_step == 1:
                print(f"{record['step']:>6} {record['epoch']:>6} {record['loss']:>10.5f} "
                      f"{record['lr']:>10.2e} {record['peak_reserved_gb']:>9.3f} {record['sec']:>7.3f}")

        line = f"  epoch {epoch:>3} mean loss: {np.mean(epoch_losses):.5f}"

        # Checkpoint at the end of every epoch. A fixed step interval silently
        # skips the best state when the optimum lands mid-interval -- the first
        # run's best epoch was step 50 with checkpoints only at 100/200/300.
        epoch_ckpt = save_checkpoint(model, ckpt_dir, global_step, float(np.mean(epoch_losses)),
                                     tag=f"epoch{epoch:03d}_step")

        stop_now = False
        if eval_hook and eval_every_epochs and ((epoch + 1) % eval_every_epochs == 0 or epoch == epochs - 1):
            metrics = eval_hook(epoch, global_step)
            trajectory.append({"epoch": epoch, "step": global_step, "mean_loss": float(np.mean(epoch_losses)),
                               "epoch_checkpoint": epoch_ckpt, **metrics})
            line += (f"  | held-out median {metrics['median_error_km']:.1f} km"
                     f"  acc@25km {metrics['acc_25km']:.4f}")

            current = metrics[select_metric]
            if current < best["value"]:
                best.update(
                    value=float(current),
                    epoch=epoch,
                    step=global_step,
                    metrics=metrics,
                    path=save_checkpoint(model, ckpt_dir, global_step, float(np.mean(epoch_losses)), tag="best"),
                )
                stale_evals = 0
                line += "  <- best"
            else:
                stale_evals += 1
                if early_stop_patience and stale_evals >= early_stop_patience:
                    line += f"  | early stop ({stale_evals} evals without improvement)"
                    stop_now = True

            model.train()
            model.image_encoder.eval()

        print(line)
        if stop_now:
            break

    log_file.close()
    print(f"\ntraining wall time: {time.perf_counter() - started:.1f}s")
    return history, trajectory, best


def save_checkpoint(model, ckpt_dir: str, step: int, loss: float, tag: str = "step") -> str:
    """Persist ONLY the location encoder -- the single artifact that carries across batches.

    "best" uses a stable filename so it is overwritten in place rather than
    accumulating one file per improvement.
    """
    name = "location_encoder_best.pth" if tag == "best" else f"location_encoder_{tag}{step}.pth"
    path = os.path.join(ckpt_dir, name)
    torch.save(
        {
            "location_encoder": model.location_encoder.state_dict(),
            "step": step,
            "loss": loss,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "note": "location encoder only; image encoder and logit_scale are unchanged pretrained weights",
        },
        path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128, help="mini-batch size (must divide --queue-size)")
    parser.add_argument("--epochs", type=int, default=30, help="passes over this data batch")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--queue-size", type=int, default=1024)
    parser.add_argument("--no-amp", action="store_true", help="disable fp16 autocast on the frozen forward")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--ckpt-every-steps", type=int, default=0,
                        help="extra mid-epoch checkpoint interval (0 = end-of-epoch checkpoints only)")
    parser.add_argument("--eval-every-epochs", type=int, default=5,
                        help="held-out eval cadence during training (0 disables)")
    parser.add_argument("--early-stop-patience", type=int, default=3,
                        help="stop after N consecutive held-out evals without improvement (0 disables)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit-train", type=int, default=None, help="cap batch size in images (for smoke runs)")
    parser.add_argument(
        "--purge-after",
        action="store_true",
        help="discard this batch's raw image data after training (only ever deletes staged copies)",
    )
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    if args.queue_size % args.batch_size != 0:
        raise SystemExit(
            f"GeoCLIP's GPS queue requires queue_size % batch_size == 0 "
            f"(got {args.queue_size} % {args.batch_size} = {args.queue_size % args.batch_size})."
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    run_name = args.run_name or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = os.path.join(CKPT_ROOT, run_name)
    work_dir = os.path.join(WORK_ROOT, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    log_path = os.path.join(work_dir, "train_log.jsonl")

    print("=" * 70)
    print(f"GeoCLIP location-encoder fine-tune | run: {run_name}")
    print("=" * 70)
    print(f"device      : {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB)")
    print(f"torch/cuda  : {torch.__version__} / {torch.version.cuda}")

    # ---- 1. Stage the batch and split BEFORE any training touches it ----
    index, stats = build_index()
    split = make_split(index, seed=args.seed)
    train_frame = split["train"]
    if args.limit_train:
        train_frame = train_frame.head(args.limit_train)

    print(f"\ndata pool   : {stats.present_on_disk} images / {stats.distinct_tiles} tiles")
    print(f"train batch : {len(train_frame)} images / {train_frame['tile_id'].nunique()} tiles")
    print(f"held-out    : {len(split['val_same_region'])} images / "
          f"{split['val_same_region']['tile_id'].nunique()} tiles (same regions, unseen tiles)")
    print(f"control     : {len(split['val_untrained_region'])} images / "
          f"{split['val_untrained_region']['tile_id'].nunique()} tiles (regions never trained on)")

    # Persist the split so the held-out slice is auditable and reusable.
    for name, frame in split.items():
        frame.to_csv(os.path.join(work_dir, f"split_{name}.csv"), index=False)

    # ---- 2. Build model and enforce the freeze contract ----
    model = build_model(device=device, queue_size=args.queue_size)
    freeze_image_side(model)
    assert_freeze_contract(model)
    print("\n" + trainable_parameter_report(model))

    gps_gallery = load_gps_data(os.path.join(geoclip_file_dir, "gps_gallery", "coordinates_100K.csv"))
    amp = not args.no_amp

    # ---- 3. Baseline eval with PRETRAINED location encoder ----
    # Image features are computed once and reused for both evaluations: the image
    # side never changes, so this isolates the location encoder exactly.
    print("\nencoding held-out images (frozen image side, computed once)...")
    val_features, val_coords = encode_val_images(
        model, split["val_same_region"], args.eval_batch_size, device, amp=amp
    )
    ctl_features, ctl_coords = encode_val_images(
        model, split["val_untrained_region"], args.eval_batch_size, device, amp=amp
    )

    print("evaluating PRETRAINED baseline...")
    before_val = evaluate(model, val_features, val_coords, gps_gallery, device)
    before_ctl = evaluate(model, ctl_features, ctl_coords, gps_gallery, device)

    # Keep an untouched copy of the pretrained weights for the record.
    pretrained_path = os.path.join(ckpt_dir, "location_encoder_pretrained.pth")
    torch.save({"location_encoder": model.location_encoder.state_dict(), "step": 0,
                "note": "untouched pretrained baseline"}, pretrained_path)

    # ---- 4. Train ----
    def eval_hook(_epoch: int, _step: int) -> dict:
        metrics = evaluate(model, val_features, val_coords, gps_gallery, device)
        return _clean(metrics)

    torch.cuda.reset_peak_memory_stats()
    history, trajectory, best = train_one_batch(
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
        ckpt_every_steps=args.ckpt_every_steps,
        log_path=log_path,
        eval_hook=eval_hook,
        eval_every_epochs=args.eval_every_epochs,
        early_stop_patience=args.early_stop_patience,
    )

    # ---- 5. Final checkpoint ----
    final_path = save_checkpoint(model, ckpt_dir, history[-1]["step"], history[-1]["loss"], tag="final")
    print(f"final checkpoint: {final_path}")

    # ---- 6. Post-training eval, same images, same features ----
    print("\nevaluating FINE-TUNED location encoder (final weights)...")
    final_val = evaluate(model, val_features, val_coords, gps_gallery, device)
    final_ctl = evaluate(model, ctl_features, ctl_coords, gps_gallery, device)

    # The deliverable is the best-on-held-out checkpoint, not the last one: training
    # loss keeps falling well past the point where held-out error starts regressing.
    after_val, after_ctl = final_val, final_ctl
    selected = "final"
    if best.get("path") and best.get("step") != history[-1]["step"]:
        print(f"reloading BEST checkpoint (epoch {best['epoch']}, step {best['step']}) for the headline comparison...")
        state = torch.load(best["path"], map_location=device)
        model.location_encoder.load_state_dict(state["location_encoder"])
        after_val = evaluate(model, val_features, val_coords, gps_gallery, device)
        after_ctl = evaluate(model, ctl_features, ctl_coords, gps_gallery, device)
        selected = f"best@epoch{best['epoch']}"

    print(format_comparison(f"HELD-OUT: same regions, unseen tiles [{selected}]", before_val, after_val))
    print(format_comparison(f"CONTROL: regions never trained on [{selected}]", before_ctl, after_ctl))

    if selected != "final":
        print(format_comparison("HELD-OUT with FINAL weights (for reference)", before_val, final_val))

    if best.get("path"):
        print(f"\nBEST held-out state: epoch {best['epoch']} (step {best['step']}), "
              f"{best['value']:.2f} km median -> {best['path']}")
        if len(trajectory) < args.epochs:
            print(f"early stopping triggered: ran {len(trajectory)} of {args.epochs} requested epochs")

    if trajectory:
        print("\n--- held-out trajectory ---")
        print(f"{'epoch':>6} {'step':>6} {'train_loss':>11} {'median_km':>11} {'acc@25km':>10}")
        for point in trajectory:
            print(f"{point['epoch']:>6} {point['step']:>6} {point['mean_loss']:>11.5f} "
                  f"{point['median_error_km']:>11.2f} {point['acc_25km']:>10.4f}")

    losses = [h["loss"] for h in history]
    n_head = max(1, len(losses) // 10)
    print(f"\nloss: first {n_head} steps mean {np.mean(losses[:n_head]):.5f} -> "
          f"last {n_head} steps mean {np.mean(losses[-n_head:]):.5f}")
    print(f"peak reserved VRAM during training: {max(h['peak_reserved_gb'] for h in history):.3f} GB")

    # ---- 7. Persist the run summary ----
    summary = {
        "run_name": run_name,
        "config": vars(args),
        "gpu": torch.cuda.get_device_name(0),
        "data": {
            "pool_images": stats.present_on_disk,
            "train_images": int(len(train_frame)),
            "val_same_region_images": int(len(split["val_same_region"])),
            "val_untrained_region_images": int(len(split["val_untrained_region"])),
        },
        "loss_first": float(np.mean(losses[:n_head])),
        "loss_last": float(np.mean(losses[-n_head:])),
        "peak_reserved_gb": float(max(h["peak_reserved_gb"] for h in history)),
        "val_same_region": {"before": _clean(before_val), "after": _clean(after_val)},
        "val_untrained_region": {"before": _clean(before_ctl), "after": _clean(after_ctl)},
        "trajectory": trajectory,
        "selected_checkpoint": selected,
        "best_checkpoint": {k: v for k, v in best.items() if k != "metrics"},
        "val_same_region_final_weights": _clean(final_val),
        "final_checkpoint": final_path,
    }
    summary_path = os.path.join(work_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nrun summary: {summary_path}")

    # ---- 8. Discard the batch's raw data ----
    if args.purge_after:
        purge_batch(work_dir, train_frame)
    else:
        print("\n[purge] skipped (--purge-after not set).")


def _clean(metrics: dict) -> dict:
    """Drop the raw per-image error array before serialising."""
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


if __name__ == "__main__":
    main()
