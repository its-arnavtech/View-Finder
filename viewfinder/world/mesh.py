"""Grid triangulation with occlusion-edge culling and vertex compaction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeshData:
    name: str
    positions: np.ndarray  # (N, 3) float32
    indices: np.ndarray  # (3M,) uint32
    uvs: np.ndarray | None = None  # (N, 2) float32
    colors: np.ndarray | None = None  # (N, 4) uint8
    texture: str | None = None  # key into the world's texture table
    fog: bool = True

    @property
    def vertex_count(self) -> int:
        return int(len(self.positions))

    @property
    def triangle_count(self) -> int:
        return int(len(self.indices) // 3)

    def stats(self) -> dict[str, object]:
        return {
            "vertices": self.vertex_count,
            "triangles": self.triangle_count,
            "index_type": "uint16" if self.vertex_count <= 0xFFFF else "uint32",
            "fog": self.fog,
        }


def grid_triangles(height: int, width: int) -> np.ndarray:
    """(2*(h-1)*(w-1), 3) vertex indices of a regular grid, two per cell.

    Order: for each cell, the upper-left triangle then the lower-right one.
    """
    rows = np.arange(height - 1)[:, None]
    cols = np.arange(width - 1)[None, :]
    a = (rows * width + cols).reshape(-1)  # top-left
    b = a + 1  # top-right
    c = a + width  # bottom-left
    d = c + 1  # bottom-right
    upper = np.stack([a, c, b], axis=1)
    lower = np.stack([b, c, d], axis=1)
    return np.stack([upper, lower], axis=1).reshape(-1, 3)


def triangle_spread(per_vertex: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """max - min of a per-vertex scalar over each triangle."""
    values = per_vertex.reshape(-1)[triangles]
    return values.max(axis=1) - values.min(axis=1)


def compact(
    name: str,
    positions: np.ndarray,
    triangles: np.ndarray,
    *,
    uvs: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    texture: str | None = None,
    fog: bool = True,
) -> MeshData | None:
    """Keep only vertices referenced by ``triangles`` and reindex. None if empty."""
    if len(triangles) == 0:
        return None
    flat_positions = positions.reshape(-1, 3)
    used = np.unique(triangles)
    remap = np.full(len(flat_positions), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return MeshData(
        name=name,
        positions=np.ascontiguousarray(flat_positions[used], dtype=np.float32),
        indices=remap[triangles].reshape(-1).astype(np.uint32),
        uvs=None if uvs is None else np.ascontiguousarray(uvs.reshape(-1, 2)[used], dtype=np.float32),
        colors=None if colors is None else np.ascontiguousarray(colors.reshape(-1, 4)[used]),
        texture=texture,
        fog=fog,
    )


def surface_mesh(
    name: str,
    positions: np.ndarray,
    uvs: np.ndarray,
    disparity: np.ndarray,
    include: np.ndarray,
    jump: float,
    texture: str,
) -> MeshData | None:
    """Triangulate a depth surface, dropping triangles that span an occlusion.

    Args:
        positions: (gh, gw, 3) unprojected vertices.
        disparity: (gh, gw) scene disparity used for the jump test.
        include: (gh, gw) bool; a triangle is kept only if all three of its
            vertices are included.
        jump: maximum disparity spread within a kept triangle.
    """
    height, width = disparity.shape
    triangles = grid_triangles(height, width)
    keep = include.reshape(-1)[triangles].all(axis=1)
    keep &= triangle_spread(disparity, triangles) <= jump
    return compact(name, positions, triangles[keep], uvs=uvs, texture=texture)


def sky_mesh(
    rays: np.ndarray, uvs: np.ndarray, sky: np.ndarray, radius: float, texture: str
) -> MeshData | None:
    """Sky pixels placed on a sphere along their rays: real clouds, no false
    parallax. Triangles touching any sky vertex are included, and their
    non-sky corners are pushed out too, which fills the seam behind
    silhouettes."""
    height, width = sky.shape
    triangles = grid_triangles(height, width)
    keep = sky.reshape(-1)[triangles].any(axis=1)
    directions = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    return compact(name="sky", positions=directions * radius, triangles=triangles[keep],
                   uvs=uvs, texture=texture, fog=False)
