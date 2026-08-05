"""Held-out evaluation: pretrained vs fine-tuned location encoder.

The comparison is tightly controlled. Because the entire image side is frozen
and never updated, a validation image's 512-d embedding is *bit-identical*
before and after fine-tuning. So we encode the validation images exactly once
and reuse those features for both model states -- the only thing that differs
between the two measurements is the location encoder's weights.

Prediction follows GeoCLIP's own inference procedure: score the image embedding
against the location embeddings of a candidate GPS gallery and take the argmax.
Error is the geodesic distance from the predicted gallery coordinate to the
image's true coordinate.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from geopy.distance import geodesic
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from clip_transforms import clip_val_transform

DISTANCE_THRESHOLDS_KM = (1, 25, 200, 750, 2500)


class ValImageDataset(Dataset):
    """Validation images with deterministic (non-augmented) preprocessing."""

    def __init__(self, frame):
        self.paths = frame["image_path"].tolist()
        self.coords = frame[["lat", "lon"]].to_numpy(dtype=np.float64)
        self.transform = clip_val_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), torch.tensor(self.coords[idx], dtype=torch.float64)


@torch.no_grad()
def encode_val_images(model, frame, batch_size: int, device: str, num_workers: int = 0, amp: bool = True):
    """Return (image_features [N,512] on device, true_coords [N,2] numpy)."""
    loader = DataLoader(
        ValImageDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    feats, coords = [], []
    model.image_encoder.eval()
    for imgs, gps in loader:
        imgs = imgs.to(device, non_blocking=True)
        if amp:
            with torch.autocast("cuda", dtype=torch.float16):
                f = model.image_encoder(imgs)
            f = f.float()
        else:
            f = model.image_encoder(imgs)
        feats.append(F.normalize(f, dim=1))
        coords.append(gps.numpy())

    return torch.cat(feats, dim=0), np.concatenate(coords, axis=0)


@torch.no_grad()
def predict_from_features(model, image_features, gps_gallery, device: str, chunk: int = 256):
    """Argmax over the GPS gallery for each image feature. Returns predicted coords [N,2] numpy."""
    gallery = gps_gallery.to(device)

    # Location embeddings for the whole gallery, computed in chunks to bound memory.
    loc_feats = []
    for start in range(0, gallery.shape[0], 8192):
        loc_feats.append(F.normalize(model.location_encoder(gallery[start : start + 8192]), dim=1))
    loc_feats = torch.cat(loc_feats, dim=0)

    preds = []
    for start in range(0, image_features.shape[0], chunk):
        logits = image_features[start : start + chunk] @ loc_feats.t()
        preds.append(torch.argmax(logits, dim=-1).cpu())
    idx = torch.cat(preds, dim=0).numpy()

    del loc_feats
    torch.cuda.empty_cache()
    return gps_gallery.cpu().numpy()[idx]


def distance_metrics(true_coords: np.ndarray, pred_coords: np.ndarray) -> dict:
    """Geodesic error stats between true and predicted coordinates."""
    errors_km = np.array(
        [geodesic(tuple(t), tuple(p)).km for t, p in zip(true_coords, pred_coords)],
        dtype=np.float64,
    )
    metrics = {
        "n": int(len(errors_km)),
        "median_error_km": float(np.median(errors_km)),
        "mean_error_km": float(np.mean(errors_km)),
        "p25_error_km": float(np.percentile(errors_km, 25)),
        "p75_error_km": float(np.percentile(errors_km, 75)),
    }
    for threshold in DISTANCE_THRESHOLDS_KM:
        metrics[f"acc_{threshold}km"] = float(np.mean(errors_km <= threshold))
    metrics["_errors_km"] = errors_km
    return metrics


@torch.no_grad()
def evaluate(model, image_features, true_coords, gps_gallery, device: str) -> dict:
    model.eval()
    pred_coords = predict_from_features(model, image_features, gps_gallery, device)
    return distance_metrics(true_coords, pred_coords)


def format_comparison(name: str, before: dict, after: dict) -> str:
    """Render a before/after table for one validation slice."""
    lines = [
        f"\n=== {name}  (n={before['n']}) ===",
        f"{'metric':<20} {'pretrained':>13} {'fine-tuned':>13} {'delta':>13}",
        "-" * 62,
    ]

    def row(label: str, key: str, fmt: str, better_lower: bool) -> str:
        b, a = before[key], after[key]
        delta = a - b
        arrow = ""
        if abs(delta) > 1e-12:
            improved = (delta < 0) if better_lower else (delta > 0)
            arrow = "  better" if improved else "  worse"
        return f"{label:<20} {b:>13{fmt}} {a:>13{fmt}} {delta:>+13{fmt}}{arrow}"

    lines.append(row("median error km", "median_error_km", ".2f", True))
    lines.append(row("mean error km", "mean_error_km", ".2f", True))
    lines.append(row("p25 error km", "p25_error_km", ".2f", True))
    lines.append(row("p75 error km", "p75_error_km", ".2f", True))
    for threshold in DISTANCE_THRESHOLDS_KM:
        lines.append(row(f"acc @ {threshold} km", f"acc_{threshold}km", ".4f", False))
    return "\n".join(lines)
