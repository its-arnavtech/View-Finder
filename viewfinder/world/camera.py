"""Camera models for unprojecting pixels into the scene.

Convention (glTF): right-handed, +Y up, the camera at the origin looking down
-Z. Image row ``v`` grows downward, so it maps to -Y.

Two projections:

* ``pinhole``: an ordinary photo. Columns and rows both follow ``tan``.
* ``cylindrical``: a panorama. Columns map *linearly* to azimuth (which is
  how phone and stitched panoramas are stored), rows follow ``tan``. It can
  span anything up to a full 360°, which a pinhole cannot represent at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

# 35 mm film frame is 36 mm across its long side.
_FILM_LONG_MM = 36.0
# Phone main cameras are ~24-28 mm equivalent; 26 mm is the common midpoint.
DEFAULT_FOCAL_35MM = 26.0
# Wider than this (width / height) is treated as a panorama. Cinema 2.39:1
# crops stay pinhole; phone panoramas are typically 3:1 to 10:1.
PANORAMA_ASPECT = 2.8
# Vertical field of view assumed for a panorama: phone panoramas are swept
# with the phone upright, so their height is roughly a main camera's long side.
PANORAMA_VFOV_DEG = 60.0


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float  # pinhole: pixels per unit tan; cylindrical: pixels per radian
    fy: float
    cx: float
    cy: float
    focal_source: str
    projection: str = "pinhole"

    @property
    def hfov_deg(self) -> float:
        if self.projection == "cylindrical":
            return math.degrees(self.width / self.fx)
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.fx)))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.fy)))

    @property
    def wraps(self) -> bool:
        """True for a panorama that closes on itself (a full 360°)."""
        return self.projection == "cylindrical" and self.width / self.fx >= 2 * math.pi - 1e-6

    def padded(self, pad_x: int, pad_y: int) -> "Intrinsics":
        """The same camera with ``pad`` pixels added on every side."""
        return replace(
            self,
            width=self.width + 2 * pad_x,
            height=self.height + 2 * pad_y,
            cx=self.cx + pad_x,
            cy=self.cy + pad_y,
        )

    def rays(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Unit-depth ray directions for pixel coordinates ``u``, ``v``.

        "Unit depth" means the ray reaches distance 1 *horizontally* from the
        camera axis: z = -1 for a pinhole, and a point on the unit cylinder for
        a panorama. Multiplying by the estimated depth places the surface.
        """
        y = -(v - self.cy) / self.fy
        if self.projection == "cylindrical":
            azimuth = (u - self.cx) / self.fx
            return np.stack([np.sin(azimuth), y, -np.cos(azimuth)], axis=-1)
        x = (u - self.cx) / self.fx
        return np.stack([x, y, -np.ones_like(x)], axis=-1)

    def max_pad_x(self) -> int | None:
        """Largest horizontal padding that does not wrap past 360°."""
        if self.projection != "cylindrical":
            return None
        return max(0, int((2 * math.pi * self.fx - self.width) // 2))

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "projection": self.projection,
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
    """Square-pixel intrinsics for a photo of this size.

    Ordinary photos are pinhole, with FOV from the 35 mm-equivalent focal
    length (EXIF, else a phone default). Images at least 2.8 times wider than
    tall are panoramas: cylindrical, with a ~60° vertical field and a
    horizontal span set by the aspect ratio, capped at a full 360°. EXIF focal
    lengths are ignored for panoramas, because they describe the individual
    frames, not the stitched result.
    """
    if width / height >= PANORAMA_ASPECT:
        focal = (height / 2.0) / math.tan(math.radians(PANORAMA_VFOV_DEG) / 2.0)
        focal = max(focal, width / (2.0 * math.pi))  # never more than 360°
        return Intrinsics(
            width=width, height=height, fx=focal, fy=focal,
            cx=width / 2.0, cy=height / 2.0, focal_source="panorama", projection="cylindrical",
        )
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
