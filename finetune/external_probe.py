"""Periodic generalization check against the independent benchmark set.

Why this exists
---------------
The in-run held-out slice and the "control" slice of untrained regions both
come from the same collection as the training data and share its biases, so
they structurally cannot detect a failure to generalize. Measured once already:
a checkpoint improved its own held-out slice from 286 km to 160 km median while
moving an independent 1000-image set not at all (250 closer / 255 farther /
495 identical, sign-test z = -0.22).

So the only trustworthy signal is a fully independent set. This module runs
``benchmark-data/manifest.csv`` (14,002 street-view images, ODbL + LGPL,
never trained on) *during* training rather than once at the end, so the
generalization trend is visible as data scales rather than arriving as a single
verdict after hours of compute.

The cheap trick that makes it affordable
----------------------------------------
Only the location encoder trains; the image side is frozen. External image
features are therefore constant for the whole run, so they are encoded **once**
and every subsequent probe only re-runs the location encoder and the
nearest-gallery lookup. A probe after the first costs seconds, not minutes.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from evaluate import encode_val_images, predict_from_features
from location_sampling import haversine_km

THRESHOLDS_KM = (1, 25, 200, 750, 2500)
DEFAULT_MANIFEST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmark-data",
    "manifest.csv",
)


def load_external_sample(
    manifest: str = DEFAULT_MANIFEST,
    sample_size: int | None = 1000,
    seed: int = 20260805,
) -> pd.DataFrame:
    """Load the external benchmark, optionally subsampled to a fixed sample.

    The seed is fixed and shared across runs on purpose: comparing two training
    runs is only meaningful against the identical image set.
    """
    frame = pd.read_csv(manifest)
    frame = frame.dropna(subset=["latitude", "longitude", "image_path"])
    frame = frame[frame["image_path"].map(os.path.exists)]
    if sample_size is not None and len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=seed)
    # encode_val_images expects lat/lon column names.
    frame = frame.rename(columns={"latitude": "lat", "longitude": "lon"})
    return frame.reset_index(drop=True)


class ExternalProbe:
    """Encode the external set once, then score any location-encoder state cheaply."""

    def __init__(
        self,
        model,
        frame: pd.DataFrame,
        gps_gallery,
        device: str,
        batch_size: int = 64,
        amp: bool = True,
    ) -> None:
        self.model = model
        self.gps_gallery = gps_gallery
        self.device = device
        self.frame = frame
        # Frozen image side => these never change for the rest of the run.
        self.features, self.true_coords = encode_val_images(
            model, frame, batch_size, device, amp=amp
        )
        self.baseline: dict | None = None
        self.history: list[dict] = []

    def measure(self) -> dict:
        """Score the model's current location encoder on the external set."""
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            predicted = predict_from_features(
                self.model, self.features, self.gps_gallery, self.device
            )
        if was_training:
            self.model.train()

        errors = haversine_km(
            self.true_coords[:, 0], self.true_coords[:, 1],
            predicted[:, 0], predicted[:, 1],
        )
        metrics = {
            "n": int(len(errors)),
            "median_error_km": float(np.median(errors)),
            "mean_error_km": float(np.mean(errors)),
        }
        for threshold in THRESHOLDS_KM:
            metrics[f"acc_{threshold}km"] = float(np.mean(errors <= threshold))
        metrics["_errors"] = errors
        return metrics

    def set_baseline(self) -> dict:
        """Record the pretrained scores every later probe is compared against."""
        self.baseline = self.measure()
        return self.baseline

    def probe(self, *, epoch: int, step: int, locations_trained: int) -> dict:
        """Measure and append one point to the generalization trend.

        Includes the paired sign test against the baseline, because aggregate
        medians move on a handful of large swings while most images are
        untouched -- the earlier null result looked like a 145 km median
        regression but was 495/1000 images completely unchanged.
        """
        if self.baseline is None:
            self.set_baseline()

        current = self.measure()
        base_errors = self.baseline["_errors"]
        errors = current["_errors"]

        closer = int(np.sum(errors < base_errors - 1e-9))
        farther = int(np.sum(errors > base_errors + 1e-9))
        identical = int(len(errors) - closer - farther)
        decided = closer + farther
        # Normal approximation to the sign test on the decided pairs only.
        z = ((closer - farther) / np.sqrt(decided)) if decided > 0 else 0.0

        point = {
            "epoch": epoch,
            "step": step,
            "locations_trained": locations_trained,
            "median_error_km": current["median_error_km"],
            "mean_error_km": current["mean_error_km"],
            "acc_25km": current["acc_25km"],
            "acc_200km": current["acc_200km"],
            "acc_750km": current["acc_750km"],
            "delta_median_km": current["median_error_km"] - self.baseline["median_error_km"],
            "closer": closer,
            "farther": farther,
            "identical": identical,
            "sign_test_z": float(z),
        }
        self.history.append(point)
        return point

    @staticmethod
    def format_trend(history: list[dict]) -> str:
        """Render the trend as a table; the shape matters more than any one row."""
        if not history:
            return "(no external probes recorded)"
        header = (
            f"{'epoch':>6} {'locs':>7} {'median_km':>11} {'d_median':>10} "
            f"{'acc@200km':>10} {'closer':>7} {'farther':>8} {'same':>6} {'sign_z':>8}"
        )
        lines = [header, "-" * len(header)]
        for point in history:
            lines.append(
                f"{point['epoch']:>6} {point['locations_trained']:>7} "
                f"{point['median_error_km']:>11.2f} {point['delta_median_km']:>+10.2f} "
                f"{point['acc_200km']:>10.4f} {point['closer']:>7} "
                f"{point['farther']:>8} {point['identical']:>6} {point['sign_test_z']:>8.2f}"
            )
        return "\n".join(lines)


def interpret(point: dict, alpha_z: float = 1.96) -> str:
    """One-line verdict for a probe, in the language the earlier result needed."""
    if abs(point["sign_test_z"]) < alpha_z:
        return (
            f"NULL EFFECT (|z|={abs(point['sign_test_z']):.2f} < {alpha_z}): "
            f"{point['identical']} of "
            f"{point['closer'] + point['farther'] + point['identical']} images unchanged; "
            "no detectable transfer to independent data."
        )
    direction = "IMPROVED" if point["closer"] > point["farther"] else "DEGRADED"
    return (
        f"{direction} (z={point['sign_test_z']:+.2f}): "
        f"{point['closer']} closer vs {point['farther']} farther, "
        f"median {point['delta_median_km']:+.2f} km."
    )
