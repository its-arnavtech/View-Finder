"""Grid triangulation, occlusion-edge handling and vertex compaction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
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
    mask: str | None = None  # texture-table key of a keep-mask (discard where < 0.5)

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


def foreground_mesh(
    name: str,
    rays: np.ndarray,
    uvs: np.ndarray,
    disparity: np.ndarray,
    usable: np.ndarray,
    jump: float,
    texture: str,
) -> tuple[MeshData | None, MeshData | None, np.ndarray, np.ndarray]:
    """The photo surface with *pixel-accurate* silhouettes.

    A triangle spanning a disparity jump (an occlusion edge) is not dropped,
    which left a staircase of missing cells along every silhouette, and is not
    stretched across the gap either, which gives a "rubber sheet". Instead it
    is rebuilt *at the near side's depth*: its far-side corners are replaced
    by private duplicates on the near corners' disparity, so the near object
    simply continues, flat, one cell past its edge. The pipeline then gives
    those edge cells a per-pixel mask that discards the far-side texels, and
    the background layer behind shows through exactly there. The silhouette
    follows the texture's pixels, not the grid.

    Args:
        rays: (gh, gw, 3) unit-depth rays of the grid vertices.
        disparity: (gh, gw) scene disparity (1/z).
        usable: (gh, gw) vertices that may carry surface (not sky).
        jump: disparity spread above which a triangle is an occlusion edge.

    Returns:
        (surface, silhouettes, cell_edge, cell_mid): the ordinary triangles,
        the extended edge triangles (to be drawn with the mask), a
        (gh-1, gw-1) map of grid cells containing an edge triangle, and each
        cell's midpoint disparity, the near/far split for the pixel mask.
    """
    height, width = disparity.shape
    triangles = grid_triangles(height, width)
    flat = disparity.reshape(-1).astype(np.float64)
    values = flat[triangles]
    top, bottom = values.max(axis=1), values.min(axis=1)
    usable_corners = usable.reshape(-1)[triangles]
    normal = (top - bottom <= jump) & usable_corners.all(axis=1)
    edge = (top - bottom > jump) & usable_corners.any(axis=1)

    flat_rays = rays.reshape(-1, 3).astype(np.float64)
    positions = flat_rays / flat[:, None]
    flat_uvs = uvs.reshape(-1, 2)
    count = len(flat)

    # Each edge triangle is extended *flat at its own near depth*: its far
    # corners get private vertices on the near corners' mean disparity. A
    # shared neighbourhood value (an earlier version used the 3x3 maximum)
    # can belong to a different, nearer object in clutter (crowds, palm
    # fronds) and stretched the cell into a sheet between two objects.
    edge_triangles = triangles[edge]
    edge_values = values[edge]
    middle = ((top + bottom) / 2.0)[edge]
    far_corner = edge_values < middle[:, None]
    near_values = np.where(far_corner, np.nan, edge_values)
    near_level = np.nanmean(near_values, axis=1)
    near_spread = np.nanmax(near_values, axis=1) - np.nanmin(near_values, axis=1)
    # Three surfaces meeting in one cell leave no single silhouette to
    # extend: such cells are cut, as before.
    clean = near_spread <= jump
    edge_triangles, far_corner, near_level = edge_triangles[clean], far_corner[clean], near_level[clean]
    extra = int(far_corner.sum())
    corner_level = np.broadcast_to(near_level[:, None], far_corner.shape)[far_corner]
    extended = flat_rays[edge_triangles[far_corner]] / corner_level[:, None]
    edge_indices = edge_triangles.copy()
    edge_indices[far_corner] = count + np.arange(extra)

    all_positions = np.concatenate([positions, extended]).astype(np.float32)
    all_uvs = np.concatenate([flat_uvs, flat_uvs[edge_triangles[far_corner]]])
    surface = compact(name, all_positions, triangles[normal], uvs=all_uvs, texture=texture)
    # Edge cells go in their own mesh: only they need the per-pixel discard,
    # and a shader that can discard disables early depth testing on many
    # mobile GPUs, so the bulk of the surface stays discard-free.
    silhouettes = compact("silhouettes", all_positions, edge_indices, uvs=all_uvs, texture=texture)

    corners = np.stack(
        [disparity[:-1, :-1], disparity[:-1, 1:], disparity[1:, :-1], disparity[1:, 1:]]
    )
    cell_mid = ((corners.max(axis=0) + corners.min(axis=0)) / 2.0).astype(np.float32)
    kept_edge = np.zeros(len(triangles), dtype=bool)
    kept_edge[np.flatnonzero(edge)[clean]] = True
    cell_edge = kept_edge.reshape(height - 1, width - 1, 2).any(axis=2)
    return surface, silhouettes, cell_edge, cell_mid


def silhouette_mask(
    pixel_disparity: np.ndarray,
    cell_edge: np.ndarray,
    cell_mid: np.ndarray,
) -> np.ndarray:
    """Per-pixel keep-mask for the foreground texture.

    Inside an edge cell, a texel is kept only if it is on the near side of
    the cell's midpoint disparity. Everywhere else every texel is kept.
    """
    image_height, image_width = pixel_disparity.shape
    cells_y, cells_x = cell_edge.shape
    # Grid vertices sit at pixel centres 0.5 + k * step, so pixel x falls in
    # cell floor(x / step).
    step_x = (image_width - 1) / max(1, cells_x)
    step_y = (image_height - 1) / max(1, cells_y)
    cx = np.clip((np.arange(image_width) / step_x).astype(int), 0, cells_x - 1)
    cy = np.clip((np.arange(image_height) / step_y).astype(int), 0, cells_y - 1)
    edge_pixels = cell_edge[cy[:, None], cx[None, :]]
    mid_pixels = cell_mid[cy[:, None], cx[None, :]]
    return ~(edge_pixels & (pixel_disparity < mid_pixels))
