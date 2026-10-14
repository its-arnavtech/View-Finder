"""Average compatible GeoCLIP location encoders and evaluate one fixed external probe."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import torch

from geoclip.model.misc import file_dir as geoclip_file_dir
from geoclip.model.misc import load_gps_data

from external_probe import ExternalProbe, interpret, load_external_sample
from model_setup import assert_freeze_contract, build_model, freeze_image_side


def average_state_dicts(
    states: list[dict[str, torch.Tensor]], weights: list[float] | None = None
) -> tuple[dict[str, torch.Tensor], list[float]]:
    """Return a float32 weighted parameter-wise mean after strict compatibility checks."""
    if not states:
        raise ValueError("At least one state dict is required")
    keys = list(states[0])
    for index, state in enumerate(states[1:], start=1):
        if list(state) != keys:
            raise ValueError(f"State dict {index} has different keys or key order")
        for key in keys:
            if state[key].shape != states[0][key].shape:
                raise ValueError(
                    f"State dict {index} shape mismatch for {key}: "
                    f"{state[key].shape} != {states[0][key].shape}"
                )
            if not torch.is_floating_point(state[key]):
                raise ValueError(f"Cannot average non-floating tensor {key}")

    raw_weights = [1.0] * len(states) if weights is None else [float(x) for x in weights]
    if len(raw_weights) != len(states):
        raise ValueError("Number of weights must equal number of checkpoints")
    if any(weight < 0 for weight in raw_weights) or sum(raw_weights) <= 0:
        raise ValueError("Weights must be non-negative with a positive sum")
    total = sum(raw_weights)
    normalized = [weight / total for weight in raw_weights]

    averaged: dict[str, torch.Tensor] = {}
    for key in keys:
        value = torch.zeros_like(states[0][key], dtype=torch.float32, device="cpu")
        for state, weight in zip(states, normalized):
            value.add_(state[key].detach().to(device="cpu", dtype=torch.float32), alpha=weight)
        averaged[key] = value
    return averaged, normalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="PATH")
    parser.add_argument("--weight", action="append", type=float, default=None)
    parser.add_argument("--name", required=True)
    parser.add_argument("--external-sample", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--queue-size", type=int, default=1024)
    parser.add_argument(
        "--output-dir", default=os.path.join(os.path.dirname(__file__), "work", "soups")
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    started = time.perf_counter()
    paths = [os.path.abspath(path) for path in args.checkpoint]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise SystemExit("Missing checkpoint(s): " + ", ".join(missing))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    payloads = [torch.load(path, map_location="cpu") for path in paths]
    try:
        states = [payload["location_encoder"] for payload in payloads]
    except KeyError as exc:
        raise SystemExit(f"Checkpoint lacks location_encoder: {exc}") from exc
    averaged, normalized_weights = average_state_dicts(states, args.weight)

    device = "cuda"
    model = build_model(device=device, queue_size=args.queue_size)
    freeze_image_side(model)
    assert_freeze_contract(model)
    gps_gallery = load_gps_data(
        os.path.join(geoclip_file_dir, "gps_gallery", "coordinates_100K.csv")
    )
    external_frame = load_external_sample(sample_size=args.external_sample)
    probe = ExternalProbe(
        model,
        external_frame,
        gps_gallery,
        device,
        batch_size=args.eval_batch_size,
        amp=not args.no_amp,
    )
    baseline = probe.set_baseline()
    model.location_encoder.load_state_dict(averaged, strict=True)
    result = probe.probe(epoch=0, step=0, locations_trained=0)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"{args.name}.pth")
    summary_path = os.path.join(output_dir, f"{args.name}.json")
    torch.save(
        {
            "location_encoder": averaged,
            "sources": paths,
            "weights": normalized_weights,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "note": "parameter-wise location-encoder checkpoint average",
        },
        checkpoint_path,
    )
    clean_baseline = {key: value for key, value in baseline.items() if key != "_errors"}
    summary = {
        "name": args.name,
        "sources": [
            {
                "path": path,
                "source_step": payload.get("step"),
                "source_epoch": payload.get("epoch"),
                "weight": weight,
            }
            for path, payload, weight in zip(paths, payloads, normalized_weights)
        ],
        "weighting": "uniform" if args.weight is None else "explicit_normalized",
        "external_sample": len(external_frame),
        "pretrained_baseline": clean_baseline,
        "external": result,
        "interpretation": interpret(result),
        "checkpoint": checkpoint_path,
        "wall_seconds": time.perf_counter() - started,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
