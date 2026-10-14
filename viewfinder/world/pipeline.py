"""Orchestration: photo in, layered 3D world (GLB + manifest) out."""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from . import SCHEMA
from .camera import intrinsics_for
from .depth import DepthEstimator, load_estimator
from .environment import (
    dome_colors,
    dome_mesh,
    extend_frame,
    fit_ground,
    fog_density,
    ground_disc_mesh,
    pad_size,
)
from .geometry import (
    make_grid,
    navigation_bounds,
    near_side_of_jumps,
    normalize_disparity,
    smooth_disparity,
    to_scene_disparity,
    unproject,
)
from .gltf import GlbBuilder
from .ingest import ImageSource, load_source
from .layers import background_layer, detect_sky, sky_colors
from .mesh import MeshData, compact, sky_mesh, surface_mesh
from .profiles import QualityProfile, get_profile, profile_dict

logger = logging.getLogger(__name__)

Progress = Callable[[str, float], None]

# Radii as multiples of `far`, nested so nothing intersects: the sky sphere
# sits inside the dome, and the ground disc ends inside both.
SKY_RADIUS = 2.5
DOME_RADIUS = 3.0
GROUND_RADIUS = 2.4
# The background layer is pushed back this much so it never z-fights the
# foreground where the two overlap.
BACKGROUND_PUSH = 1.01

STAGES = (
    ("ingest", 0.05),
    ("depth", 0.30),
    ("layers", 0.55),
    ("environment", 0.70),
    ("mesh", 0.85),
    ("export", 1.0),
)


@dataclass
class WorldResult:
    glb: bytes
    manifest: dict[str, Any]
    meshes: list[MeshData] = field(default_factory=list)

    @property
    def size_bytes(self) -> int:
        return len(self.glb)


def _encode_jpeg(rgb: np.ndarray, long_side: int, quality: int) -> tuple[bytes, list[int]]:
    image = Image.fromarray(rgb)
    scale = long_side / max(image.size)
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=False)
    return buffer.getvalue(), [image.width, image.height]


def _rgb01(color: np.ndarray) -> list[float]:
    return [round(float(c), 4) for c in np.clip(color, 0.0, 1.0)]


def build_world(
    source: ImageSource,
    *,
    profile: str | QualityProfile | None = None,
    estimator: DepthEstimator | None = None,
    progress: Progress | None = None,
) -> WorldResult:
    """Build an explorable layered 3D world from one photo."""
    profile = get_profile(profile)
    estimator = estimator or load_estimator()
    report = progress or (lambda stage, fraction: None)
    timings: dict[str, float] = {}
    warnings: list[str] = []
    clock = time.perf_counter()

    def mark(stage: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[stage] = round((now - clock) * 1000.0, 1)
        clock = now
        report(stage, dict(STAGES)[stage])

    report("start", 0.0)
    # -- 1. ingest -------------------------------------------------------
    photo = load_source(source, profile.working_long_side)
    height, width = photo.rgb.shape[:2]
    intrinsics = intrinsics_for(width, height, photo.focal_35mm)
    mark("ingest")

    # -- 2. depth --------------------------------------------------------
    raw = estimator.estimate(photo.rgb)
    if raw.shape != (height, width) or not np.isfinite(raw).all():
        raise RuntimeError(f"depth estimator returned {raw.shape} / non-finite values")
    mark("depth")

    # -- 3. sky + conditioning (two-pass normalisation; PDD-REVIEW R1) ----
    provisional = normalize_disparity(raw)
    sky = detect_sky(provisional) if estimator.name != "heuristic" else np.zeros_like(raw, bool)
    normalized = normalize_disparity(raw, exclude=sky)
    normalized = smooth_disparity(normalized, max(width, height))
    normalized[sky] = 0.0
    zenith, horizon = sky_colors(photo.rgb, sky)
    # Below-horizon dome colour: the median of the bottom band, so the dome,
    # the frame fade and the ground disc all agree on what "ground" looks like.
    bottom = photo.rgb[int(height * 0.75) :][~sky[int(height * 0.75) :]]
    ground_tone = (
        np.median(bottom, axis=0).astype(np.float32) / 255.0 if len(bottom) else horizon * 0.6
    )

    pad_y, pad_x = pad_size(height, width, profile.pad_fraction)
    canvas_intrinsics = intrinsics.padded(pad_x, pad_y)
    pixel_v, pixel_u = np.mgrid[0 : height + 2 * pad_y, 0 : width + 2 * pad_x].astype(np.float32) + 0.5
    pixel_rays = canvas_intrinsics.rays(pixel_u, pixel_v)
    pixel_rays /= np.linalg.norm(pixel_rays, axis=-1, keepdims=True)
    fade = dome_colors(pixel_rays, np.array([0.0, 1.0, 0.0]), zenith, horizon, ground_tone) * 255.0
    canvas = extend_frame(photo.rgb, normalized, sky, profile.pad_fraction, fade_to=fade)
    grid = make_grid(canvas.rgb.shape[1], canvas.rgb.shape[0], profile.grid_long)
    pixel_disparity = to_scene_disparity(canvas.disparity, profile.far)
    # A 3x3 median at grid resolution removes single-vertex depth spikes
    # (grass blades, wires narrower than a grid cell) that would otherwise
    # shred the near field into slivers, while real silhouettes, which span
    # several vertices, keep their jump.
    disparity = cv2.medianBlur(grid.sample(pixel_disparity).astype(np.float32), 3)
    sky_grid = grid.sample(canvas.sky)
    depth = 1.0 / disparity
    positions = unproject(grid, canvas_intrinsics, depth)
    uvs = grid.uv()

    near = near_side_of_jumps(np.where(sky_grid, 1.0 / profile.far, disparity), profile.edge_disparity_jump)
    near &= ~sky_grid
    dilation_px = max(2, int(round(profile.occlusion_dilation_fraction * max(canvas.rgb.shape[:2]))))
    background = background_layer(
        canvas.rgb,
        pixel_disparity,
        disparity,
        sky_grid,
        near,
        dilation_px,
        profile.edge_disparity_jump,
    )
    mark("layers")

    # -- 4. environment ---------------------------------------------------
    original_grid = grid.sample(canvas.original)
    lower = np.zeros_like(sky_grid)
    lower[int(grid.height * 0.62) :, :] = True
    ground_sample = lower & original_grid & ~sky_grid
    ground = fit_ground(
        positions[ground_sample],
        grid.sample(canvas.rgb)[ground_sample].astype(np.float32) / 255.0,
        scene_points=positions[original_grid & ~sky_grid],
    )
    if not ground.accepted:
        warnings.append(f"no ground plane extended: {ground.reason}")
    up = ground.normal if ground.accepted else np.array([0.0, 1.0, 0.0])
    ground_color = ground_tone
    mark("environment")

    # -- 5. meshes --------------------------------------------------------
    meshes: list[MeshData] = []
    dome_positions, dome_rgba, dome_indices = dome_mesh(
        profile.far * DOME_RADIUS, profile.sky_sphere_segments, up, zenith, horizon, ground_color
    )
    meshes.append(MeshData("dome", dome_positions, dome_indices, colors=dome_rgba, fog=False))

    sky_layer = None
    if sky_grid.any():
        rays = canvas_intrinsics.rays(grid.u, grid.v)
        sky_layer = sky_mesh(rays, uvs, sky_grid, profile.far * SKY_RADIUS, texture="photo")
        if sky_layer:
            meshes.append(sky_layer)

    if ground.accepted:
        disc_positions, disc_colors, disc_indices = ground_disc_mesh(
            ground,
            profile.far * GROUND_RADIUS,
            max(24, profile.sky_sphere_segments),
            horizon,
            # Deeper than the photo ground's own depth noise (~2-3% of camera
            # height on real photos), so the disc never pokes through it. The
            # viewer adds a slope-scaled polygon offset on top.
            sink=0.05 * abs(ground.offset),
        )
        meshes.append(MeshData("ground", disc_positions, disc_indices, colors=disc_colors))

    background_positions = unproject(
        grid, canvas_intrinsics, BACKGROUND_PUSH / background.disparity
    )
    background_mesh = surface_mesh(
        "background",
        background_positions,
        uvs,
        background.disparity,
        background.region,
        profile.edge_disparity_jump * 2.0,
        texture="background",
    )
    if background_mesh:
        meshes.append(background_mesh)

    foreground = surface_mesh(
        "foreground", positions, uvs, disparity, ~sky_grid, profile.edge_disparity_jump, texture="photo"
    )
    if foreground is None:
        raise RuntimeError("the depth map produced no foreground surface")
    meshes.append(foreground)
    mark("mesh")

    # -- 6. export --------------------------------------------------------
    textures: dict[str, tuple[bytes, list[int]]] = {
        "photo": _encode_jpeg(canvas.rgb, profile.texture_long_side, profile.jpeg_quality)
    }
    if background_mesh:
        # Built at half resolution (layers.background_layer): the layer is
        # only ever seen in thin slivers behind silhouettes.
        textures["background"] = _encode_jpeg(
            background.rgb, profile.texture_long_side // 2, profile.jpeg_quality
        )

    valid = original_grid & ~sky_grid
    sky_fraction = float(sky.mean())
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "name": photo.name,
        "source": {
            "width": photo.source_width,
            "height": photo.source_height,
            "working": [width, height],
            "padded": [canvas.rgb.shape[1], canvas.rgb.shape[0]],
        },
        "profile": profile.name,
        "profile_settings": profile_dict(profile),
        "depth": estimator.describe(),
        "camera": {
            **intrinsics.as_dict(),
            "position": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, -1.0],
            "up": [0.0, 1.0, 0.0],
            "near_clip": 0.05,
            "far_clip": round(profile.far * DOME_RADIUS * 1.5, 3),
        },
        "navigation": navigation_bounds(depth, valid),
        "environment": {
            "sky_fraction": round(sky_fraction, 4),
            "zenith": _rgb01(zenith),
            "horizon": _rgb01(horizon),
            "up": [round(float(v), 4) for v in up],
            "fog": {"color": _rgb01(horizon), "density": round(fog_density(profile.far), 6)},
            "ground": ground.as_dict(),
            "radii": {
                "sky": profile.far * SKY_RADIUS,
                "dome": profile.far * DOME_RADIUS,
                "ground": profile.far * GROUND_RADIUS,
            },
        },
        "layers": {mesh.name: mesh.stats() for mesh in meshes},
        "textures": {key: {"size": size, "bytes": len(data)} for key, (data, size) in textures.items()},
        "timings_ms": timings,
        "warnings": warnings,
    }

    builder = GlbBuilder()
    texture_index = {key: builder.add_texture(data) for key, (data, _) in textures.items()}
    for mesh in meshes:
        builder.add_mesh(
            mesh.name,
            mesh.positions,
            mesh.indices,
            uvs=mesh.uvs,
            colors=mesh.colors,
            texture=texture_index.get(mesh.texture) if mesh.texture else None,
        )
    glb = builder.build(extras={"viewfinder_world": manifest})
    mark("export")
    manifest["timings_ms"]["total"] = round(sum(v for k, v in timings.items() if k != "total"), 1)
    manifest["glb_bytes"] = len(glb)
    logger.info(
        "Built %s world: %d triangles, %.1f MB, %.0f ms",
        profile.name,
        sum(m.triangle_count for m in meshes),
        len(glb) / 1e6,
        manifest["timings_ms"]["total"],
    )
    return WorldResult(glb=glb, manifest=manifest, meshes=meshes)
