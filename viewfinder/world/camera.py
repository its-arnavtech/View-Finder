"""Pinhole camera model for unprojecting pixels into the scene.

Convention (glTF): right-handed, +Y up, the camera at the origin looking down
-Z. Image row ``v`` grows downward, so it maps to -Y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 35 mm film frame is 36 mm across its long side.
_FILM_LONG_MM = 36.0
# Phone main cameras are ~24-28 mm equivalent; 26 mm is the common midpoint.
DEFAULT_FOCAL_35MM = 26.0


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    focal_source: str

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.fx)))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.fy)))

    def scaled(self, sx: float, sy: float) -> "Intrinsics":
        """The same camera described on a grid scaled by ``sx``/``sy``."""
        return Intrinsics(
            width=round(self.width * sx),
            height=round(self.height * sy),
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            focal_source=self.focal_source,
        )

    def padded(self, pad_x: int, pad_y: int) -> "Intrinsics":
        """The same camera with ``pad`` pixels added on every side."""
        return Intrinsics(
            width=self.width + 2 * pad_x,
            height=self.height + 2 * pad_y,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx + pad_x,
            cy=self.cy + pad_y,
            focal_source=self.focal_source,
        )

    def rays(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Unit-depth ray directions (z = -1) for pixel coordinates ``u``, ``v``."""
        x = (u - self.cx) / self.fx
        y = -(v - self.cy) / self.fy
        return np.stack([x, y, -np.ones_like(x)], axis=-1)

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": round(self.fx, 4),
            "fy": round(self.fy, 4),
            "cx": round(self.cx, 4),
            "cy": round(self.cy, 4),
            "hfov_deg": round(self.hfov_deg, 3),
            "vfov_deg": round(self.vfov_deg, 3),
            "focal_source": self.focal_source,
        }


def intrinsics_for(width: int, height: int, focal_35mm: float | None = None) -> Intrinsics:
    """Square-pixel intrinsics, with FOV from the 35 mm-equivalent focal length."""
    source = "exif" if focal_35mm else "default"
    focal_mm = focal_35mm or DEFAULT_FOCAL_35MM
    long_side = max(width, height)
    focal_px = long_side * focal_mm / _FILM_LONG_MM
    return Intrinsics(
        width=width,
        height=height,
        fx=focal_px,
        fy=focal_px,
        cx=width / 2.0,
        cy=height / 2.0,
        focal_source=source,
    )
