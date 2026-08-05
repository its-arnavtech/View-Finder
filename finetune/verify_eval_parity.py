"""Check that our evaluation path reproduces GeoCLIP's own predict().

Our eval batches images through a torchvision pipeline and scores them against
the GPS gallery ourselves, rather than calling GeoCLIP.predict() per image. If
those two disagree, the "pretrained baseline" we report is not the model's real
deployed behaviour and every before/after delta built on it is suspect.

This compares predicted coordinates from both routes on the same images.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from dataset_index import build_index
from evaluate import encode_val_images, predict_from_features
from model_setup import build_model, freeze_image_side
from split import make_split

from geoclip.model.misc import file_dir as geoclip_file_dir
from geoclip.model.misc import load_gps_data

N_IMAGES = 24


def main() -> None:
    device = "cuda"
    model = build_model(device=device)
    freeze_image_side(model)
    model.eval()

    index, _ = build_index()
    val = make_split(index)["val_same_region"].head(N_IMAGES).reset_index(drop=True)
    gps_gallery = load_gps_data(os.path.join(geoclip_file_dir, "gps_gallery", "coordinates_100K.csv"))

    # Route A: geoclip's own per-image predict().
    native = []
    for path in val["image_path"]:
        top_gps, _ = model.predict(path, top_k=1)
        native.append(top_gps[0].cpu().numpy())
    native = np.stack(native)

    # Route B: our batched pipeline, fp32 to remove autocast as a variable.
    feats, _coords = encode_val_images(model, val, batch_size=8, device=device, amp=False)
    ours = predict_from_features(model, feats, gps_gallery, device)

    exact = int((np.isclose(native, ours, atol=1e-6).all(axis=1)).sum())
    print(f"images compared           : {len(val)}")
    print(f"identical gallery pick    : {exact}/{len(val)}")

    # Where they differ, how far apart are the two predictions?
    from geopy.distance import geodesic

    gaps = np.array([geodesic(tuple(a), tuple(b)).km for a, b in zip(native, ours)])
    print(f"disagreement distance km  : median {np.median(gaps):.3f} | max {gaps.max():.3f}")

    # Also verify the same with autocast on, since training/eval uses fp16.
    feats_amp, _ = encode_val_images(model, val, batch_size=8, device=device, amp=True)
    ours_amp = predict_from_features(model, feats_amp, gps_gallery, device)
    exact_amp = int((np.isclose(native, ours_amp, atol=1e-6).all(axis=1)).sum())
    gaps_amp = np.array([geodesic(tuple(a), tuple(b)).km for a, b in zip(native, ours_amp)])
    print(f"\nwith fp16 autocast:")
    print(f"identical gallery pick    : {exact_amp}/{len(val)}")
    print(f"disagreement distance km  : median {np.median(gaps_amp):.3f} | max {gaps_amp.max():.3f}")


if __name__ == "__main__":
    main()
