"""Measure real training-step throughput per batch size on this machine.

Motivation: on Windows WDDM the NVIDIA driver does not cleanly OOM when a
workload exceeds physical VRAM -- it spills into shared system memory and keeps
running, just far slower. A config can therefore look "ok" in a pure memory
probe while being unusable in practice, so throughput is what actually bounds
the batch size.

The model is built once and reused across configurations; rebuilding CLIP
ViT-L/14 per configuration dominated the runtime otherwise.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_setup import build_model, freeze_image_side

TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3


def bench(model, batch_size: int, queue_size: int, amp: bool, steps: int = 6, warmup: int = 2) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    optimizer = torch.optim.AdamW(model.location_encoder.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    targets = torch.arange(batch_size, device="cuda")

    scale = torch.tensor([45.0, 90.0], device="cuda")
    imgs = torch.randn(batch_size, 3, 224, 224, device="cuda")
    gps = torch.randn(batch_size, 2, device="cuda") * scale
    queue = torch.randn(queue_size, 2, device="cuda") * scale

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            if amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    feats = model.image_encoder(imgs)
                feats = feats.float()
            else:
                feats = model.image_encoder(imgs)
        loc = model.location_encoder(torch.cat([gps, queue], dim=0))
        logits = model.logit_scale.exp() * (F.normalize(feats, dim=1) @ F.normalize(loc, dim=1).t())
        criterion(logits, targets).backward()
        optimizer.step()

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
    result = {
        "batch_size": batch_size,
        "amp": amp,
        "sec_per_step": round(elapsed / steps, 4),
        "images_per_sec": round(batch_size * steps / elapsed, 1),
        "peak_reserved_gb": round(peak_reserved_gb, 3),
        "over_budget": bool(peak_reserved_gb > TOTAL_VRAM_GB * 0.95),
    }

    del optimizer, imgs, gps, queue
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--queue-size", type=int, default=1024)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}   total VRAM: {TOTAL_VRAM_GB:.2f} GB", flush=True)
    print("building model once and reusing across configs...", flush=True)

    model = build_model(device="cuda", queue_size=args.queue_size)
    freeze_image_side(model)
    model.train()

    header = f"{'batch':>6} {'amp':>6} {'s/step':>9} {'img/s':>9} {'peak_resv':>10} {'verdict':>14}"
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)

    results = []
    for amp in (False, True):
        for bs in args.batch_sizes:
            if args.queue_size % bs != 0:
                print(f"{bs:>6} {str(amp):>6}  skipped (queue_size not divisible)", flush=True)
                continue
            r = bench(model, bs, args.queue_size, amp)
            results.append(r)
            verdict = "SPILL->sysRAM" if r["over_budget"] else "fits"
            print(
                f"{r['batch_size']:>6} {str(r['amp']):>6} {r['sec_per_step']:>9.4f} "
                f"{r['images_per_sec']:>9.1f} {r['peak_reserved_gb']:>9.3f}G {verdict:>14}",
                flush=True,
            )
            sys.stdout.flush()

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
