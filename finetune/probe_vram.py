"""Measure real peak VRAM for one GeoCLIP location-encoder training step.

The step under test is exactly the one the training loop runs:
  1. frozen CLIP visual tower + frozen projection MLP forward, under no_grad
  2. location encoder forward on (batch GPS + GPS queue), with grad
  3. contrastive loss + backward + optimizer step, touching only the location encoder

We sweep batch sizes and report torch's peak allocated/reserved bytes so the
batch size is chosen from measurement rather than a guess. Both fp32 and
autocast(fp16) are measured for the frozen forward pass.
"""

from __future__ import annotations

import argparse
import gc
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_setup import build_model, freeze_image_side, trainable_parameter_report

TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0.0


def probe(batch_size: int, queue_size: int, amp: bool, device: str = "cuda") -> dict:
    """Run a few synthetic training steps at this batch size and report peak memory."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build_model(device=device)
    freeze_image_side(model)
    model.train()

    optimizer = torch.optim.AdamW(model.location_encoder.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    targets = torch.arange(batch_size, device=device)

    baseline = torch.cuda.memory_allocated()

    # Synthetic inputs matching the real pipeline's shapes.
    imgs = torch.randn(batch_size, 3, 224, 224, device=device)
    gps = torch.randn(batch_size, 2, device=device) * torch.tensor([45.0, 90.0], device=device)
    queue = torch.randn(queue_size, 2, device=device) * torch.tensor([45.0, 90.0], device=device)

    status = "ok"
    err = None
    try:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)

            # (1) Frozen image side — no graph is built, no activations retained.
            with torch.no_grad():
                if amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        image_features = model.image_encoder(imgs)
                    image_features = image_features.float()
                else:
                    image_features = model.image_encoder(imgs)

            # (2) Trainable location encoder over batch GPS + queue negatives.
            gps_all = torch.cat([gps, queue], dim=0)
            location_features = model.location_encoder(gps_all)

            # (3) Contrastive loss.
            image_features = F.normalize(image_features, dim=1)
            location_features = F.normalize(location_features, dim=1)
            logits = model.logit_scale.exp() * (image_features @ location_features.t())
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()
    except torch.cuda.OutOfMemoryError as exc:
        status = "OOM"
        err = str(exc).splitlines()[0]

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    result = {
        "batch_size": batch_size,
        "queue_size": queue_size,
        "amp": amp,
        "status": status,
        "model_resident_gb": round(baseline / 1024**3, 3),
        "peak_allocated_gb": round(peak_alloc / 1024**3, 3),
        "peak_reserved_gb": round(peak_reserved / 1024**3, 3),
        "headroom_gb": round(TOTAL_VRAM_GB - peak_reserved / 1024**3, 3),
        "error": err,
    }

    del model, optimizer, imgs, gps, queue
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--queue-size", type=int, default=1024)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available -- refusing to probe on CPU.")

    print(f"device            : {torch.cuda.get_device_name(0)}")
    print(f"total VRAM        : {TOTAL_VRAM_GB:.2f} GB")
    print(f"torch / cuda      : {torch.__version__} / {torch.version.cuda}")
    print()

    # One-off report of what is actually trainable.
    model = build_model(device="cuda")
    freeze_image_side(model)
    print(trainable_parameter_report(model))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print()

    results = []
    header = f"{'batch':>6} {'amp':>6} {'status':>7} {'resident':>9} {'peak_alloc':>11} {'peak_resv':>10} {'headroom':>9}"
    print(header)
    print("-" * len(header))
    for amp in (False, True):
        for bs in args.batch_sizes:
            r = probe(bs, args.queue_size, amp)
            results.append(r)
            print(
                f"{r['batch_size']:>6} {str(r['amp']):>6} {r['status']:>7} "
                f"{r['model_resident_gb']:>8.3f}G {r['peak_allocated_gb']:>10.3f}G "
                f"{r['peak_reserved_gb']:>9.3f}G {r['headroom_gb']:>8.3f}G"
            )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
