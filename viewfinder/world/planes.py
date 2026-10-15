"""Dominant planes: the ground, a ceiling, walls. A prior that keeps them flat.

Every generated view gets its own monocular depth, aligned to what already
exists. Open scenes forgive the small errors that leaves; enclosed ones do
not: a corridor's walls, a room's floor and ceiling, a canyon's sides are a
handful of large surfaces seen up close from every station, and each view's
noise showed as cracks and shards between patches.

So the big planes are found once, in the 360° panorama's geometry around the
start point, and any depth that lies near one of them (and faces the same
way) is snapped onto it, in the panorama, in the photo and in every view
generated while exploring. Snapping moves a point along its own ray, so the
start view (which looks along those rays) is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray  # unit (3,), pointing toward the start point
    offset: float  # plane: normal . p + offset = 0
    kind: str  # ground | ceiling | wall
    support: float  # share of the fitted points on it

    def scaled(self, factor: float) -> "Plane":
        return Plane(self.normal, self.offset * factor, self.kind, self.support)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "normal": [round(float(v), 4) for v in self.normal],
            "offset": round(float(self.offset), 4),
            "support": round(self.support, 3),
        }


# A point within this share of its distance from a plane, with a normal
# within SNAP_ANGLE of the plane's, is snapped onto it.
SNAP_TOLERANCE = 0.06
SNAP_ANGLE = np.radians(25.0)
# Planes need this share of the scene's points to count.
MIN_SUPPORT = 0.04
MAX_WALLS = 4


def normals_from_points(points: np.ndarray, step: int = 3) -> np.ndarray:
    """Unit normals of an organised (H, W, 3) point grid, facing the origin.
    Zero where undefined. Taken over ``step`` pixels of a lightly smoothed
    grid: one-pixel differences of monocular depth are mostly noise."""
    import cv2

    points = cv2.GaussianBlur(points.astype(np.float32), (0, 0), 1.0)
    du = np.zeros_like(points)
    dv = np.zeros_like(points)
    du[:, step:-step] = points[:, 2 * step :] - points[:, : -2 * step]
    dv[step:-step] = points[2 * step :] - points[: -2 * step]
    normal = np.cross(du, dv)
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.where(length > 1e-9, normal / np.maximum(length, 1e-9), 0.0)
    facing = np.sum(normal * points, axis=-1, keepdims=True) > 0
    return np.where(facing, -normal, normal).astype(np.float32)


def fit_planes(points: np.ndarray, normals: np.ndarray, ground: float, seed: int = 0) -> list[Plane]:
    """The ground (``ground`` below the eye, always), a ceiling and up to
    four walls, from scene points (N, 3) with normals (N, 3)."""
    planes = [Plane(np.array([0.0, 1.0, 0.0]), ground, "ground", 1.0)]
    valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(normals, axis=1) > 0.5)
    points, normals = points[valid], normals[valid]
    if len(points) < 200:
        return planes
    total = len(points)
    distance = np.linalg.norm(points, axis=1)

    # A ceiling: a horizontal surface above the eye that many points agree on.
    up = (normals[:, 1] < -np.cos(SNAP_ANGLE)) & (points[:, 1] > 0.3 * ground)
    if up.sum() > MIN_SUPPORT * total:
        heights = points[up, 1]
        histogram, edges = np.histogram(heights, bins=60)
        peak = int(np.argmax(histogram))
        height = float((edges[peak] + edges[peak + 1]) / 2)
        on = np.abs(heights - height) < SNAP_TOLERANCE * distance[up]
        if on.sum() > MIN_SUPPORT * total:
            height = float(np.median(heights[on]))
            planes.append(Plane(np.array([0.0, -1.0, 0.0]), height, "ceiling", float(on.sum() / total)))

    # Walls: vertical planes, found by RANSAC on the floor plan (x, z) of
    # points whose normals are horizontal.
    side = np.abs(normals[:, 1]) < np.sin(SNAP_ANGLE)
    candidates = points[side][:, [0, 2]].astype(np.float64)
    candidate_normals = normals[side][:, [0, 2]]
    candidate_distance = distance[side]
    rng = np.random.default_rng(seed)
    remaining = np.ones(len(candidates), dtype=bool)
    for _ in range(MAX_WALLS):
        index = np.nonzero(remaining)[0]
        if len(index) < MIN_SUPPORT * total:
            break
        best, best_line = None, None
        for _ in range(150):
            a, b = candidates[rng.choice(index, 2, replace=False)]
            direction = b - a
            length = np.linalg.norm(direction)
            if length < 0.2:
                continue
            normal = np.array([-direction[1], direction[0]]) / length
            offset = -float(normal @ a)
            residual = np.abs(candidates[index] @ normal + offset)
            agree = np.abs(candidate_normals[index] @ normal) > np.cos(SNAP_ANGLE)
            inliers = index[(residual < SNAP_TOLERANCE * candidate_distance[index]) & agree]
            if best is None or len(inliers) > len(best):
                best, best_line = inliers, (normal, offset)
        if best is None or len(best) < MIN_SUPPORT * total:
            break
        # Refine on the inliers (total least squares).
        centre = candidates[best].mean(axis=0)
        _, _, vt = np.linalg.svd(candidates[best] - centre, full_matrices=False)
        normal2 = vt[-1]
        offset = -float(normal2 @ centre)
        normal3 = np.array([normal2[0], 0.0, normal2[1]])
        if offset < 0:  # face the origin: n . 0 + offset > 0
            normal3, offset = -normal3, -offset
        planes.append(Plane(normal3, offset, "wall", float(len(best) / total)))
        remaining[best] = False
    return planes


def snap_depth(
    origin: np.ndarray,
    rays: np.ndarray,
    depth: np.ndarray,
    planes: list[Plane],
    editable: np.ndarray | None = None,
) -> np.ndarray:
    """Depth (along ``rays``, which have unit length in the depth's sense)
    with points near a plane, and facing like it, moved onto it along their
    ray. ``editable`` limits which pixels may change."""
    if not planes:
        return depth
    points = origin[None, None, :] + rays * depth[..., None]
    normals = normals_from_points(points - origin[None, None, :])
    out = depth.copy()
    best_gap = np.full(depth.shape, np.inf, np.float32)
    distance = np.linalg.norm(points - origin, axis=-1)
    for plane in planes:
        gap = np.abs(points @ plane.normal + plane.offset)
        facing = np.abs(normals @ plane.normal) > np.cos(SNAP_ANGLE)
        along = rays @ plane.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            hit = -(origin @ plane.normal + plane.offset) / along
        close = (gap < SNAP_TOLERANCE * distance) & facing & (depth > 0) & np.isfinite(hit) & (hit > 0)
        if editable is not None:
            close &= editable
        better = close & (gap < best_gap)
        out = np.where(better, hit, out)
        best_gap = np.where(better, gap, best_gap)
    return out.astype(np.float32)
