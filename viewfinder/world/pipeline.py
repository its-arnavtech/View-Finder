"""Orchestration: photo in, layered 3D world (GLB + manifest) out."""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import cv2
import numpy as np
from PIL import Image

from . import SCHEMA
from .camera import intrinsics_for
from .depth import DepthEstimator, load_estimator
from .environment import (
    environment_dome,
    environment_map,
    extend_frame,
    fit_ground,
    fog_density,
    ground_disc_mesh,
    pad_size,
    sample_environment,
)
from .geometry import (
    guided_filter,
    sharpen_depth_edges,
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
from .inpaint import Inpainter
from .layers import background_layer, detect_sky, sky_colors
from .mesh import MeshData, foreground_mesh, grid_triangles, silhouette_mask, sky_mesh, surface_mesh
from .surround import land_mask, surround_layer
from .profiles import QualityProfile, get_profile, profile_dict

if TYPE_CHECKING:
    from .generate import ViewBackend
    from .scene import Scene

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
# The generated surround is cut where a triangle's disparity spread
# exceeds this fraction of its nearest vertex's (mesh.surround_mesh).
SURROUND_RELATIVE_JUMP = 0.3

STAGES = (
    ("ingest", 0.02),
    ("depth", 0.05),
    # Generating the surround and exploring dominate a build when they run
    # (minutes on a laptop GPU); both report their own progress in between.
    ("generate", 0.50),
    ("layers", 0.52),
    ("environment", 0.53),
    ("mesh", 0.55),
    ("explore", 0.97),
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


def _fit(image: Image.Image, long_side: int) -> Image.Image:
    scale = long_side / max(image.size)
    if scale >= 1.0:
        return image
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS if image.mode == "RGB" else Image.Resampling.BOX)


def _encode_jpeg(rgb: np.ndarray, long_side: int, quality: int) -> tuple[bytes, list[int], str]:
    image = _fit(Image.fromarray(rgb), long_side)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=False)
    return buffer.getvalue(), [image.width, image.height], "image/jpeg"


def _encode_mask(keep: np.ndarray, long_side: int) -> tuple[bytes, list[int], str]:
    """Keep-mask as an 8-bit PNG. Mostly solid white, so it compresses to a
    few tens of KB; box-filtered on downscale so its edge stays anti-aliased."""
    image = _fit(Image.fromarray(np.where(keep, 255, 0).astype(np.uint8), mode="L"), long_side)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), [image.width, image.height], "image/png"


def _panorama_points(disparity: np.ndarray) -> np.ndarray:
    """(H, W, 3) points of an equirectangular inverse-distance map (inf where sky)."""
    from .panorama import equirect_directions

    directions = equirect_directions(disparity.shape[1], disparity.shape[0])
    distance = np.where(disparity > 1e-6, 1.0 / np.maximum(disparity, 1e-6), np.inf)
    return directions * distance[..., None]


def _panorama_planes(disparity: np.ndarray, ground: float) -> list:
    """The ground, a ceiling and walls in the panorama's geometry (scene units)."""
    from .planes import fit_planes, normals_from_points

    points = _panorama_points(disparity)
    finite = np.where(np.isfinite(points), points, 0.0)
    normals = normals_from_points(finite)
    solid = np.isfinite(points).all(axis=-1)
    step = (slice(None, None, 2), slice(None, None, 2))  # a quarter of the points is plenty
    return fit_planes(points[step][solid[step]], normals[step][solid[step]], ground)


def _snap_panorama(disparity: np.ndarray, planes: list) -> np.ndarray:
    from .panorama import equirect_directions
    from .planes import snap_depth

    directions = equirect_directions(disparity.shape[1], disparity.shape[0]).astype(np.float32)
    solid = disparity > 1e-6
    distance = np.where(solid, 1.0 / np.maximum(disparity, 1e-6), 0.0).astype(np.float32)
    snapped = snap_depth(np.zeros(3), directions, distance, planes, editable=solid)
    return np.where(solid, 1.0 / np.maximum(snapped, 1e-6), 0.0).astype(np.float32)


def _walk_bounds(exploration) -> dict[str, Any]:
    """Where a walker may go: the explored stations, plus a margin (metres)."""
    from .explore import EYE_HEIGHT

    positions = [np.zeros(3)] + [s.position for s in exploration.stations if s.built]
    reach = np.max(np.abs(np.array(positions)), axis=0) + 1.5
    return {
        "center": [0.0, 0.0, 0.0],
        "half_extents": [round(float(reach[0]), 3), 0.4, round(float(reach[2]), 3)],
        "eye_height": EYE_HEIGHT,
        "walk": True,
        # The viewer keeps a walker within ``reach`` metres of a point the
        # world was generated around: elsewhere nothing was ever looked at.
        "stations": [[round(float(p[0]), 2), round(float(p[2]), 2)] for p in positions],
        "reach": 2.5,
    }


def _explore_summary(exploration, scale: float) -> dict[str, Any]:
    if exploration is None:
        return {"views": 0}
    return {
        "metres_per_unit": round(scale, 4),
        "views": exploration.views,
        "bubbles": len(exploration.bubbles),
        "depth_guided": exploration.depth_guided,
        "stations": [
            {"position": [round(float(v), 2) for v in s.position], "views": s.views, "source": s.source,
             "built": s.built, "seconds": round(s.seconds, 1)}
            for s in exploration.stations
        ],
        "seconds": round(exploration.seconds, 1),
    }


def _rgb01(color: np.ndarray) -> list[float]:
    return [round(float(c), 4) for c in np.clip(color, 0.0, 1.0)]


def build_world(
    source: ImageSource,
    *,
    profile: str | QualityProfile | None = None,
    estimator: DepthEstimator | None = None,
    inpainter: Inpainter | None = None,
    generator: "ViewBackend | None" = None,
    classify: "Callable[[np.ndarray], Scene] | None" = None,
    progress: Progress | None = None,
    surround_cache: "Path | None" = None,
) -> WorldResult:
    """Build an explorable layered 3D world from one photo.

    ``generator`` (an image model, see generate.py) invents the full 360°
    world around the photo: a generated panorama with its own depth, meshed
    all the way round. ``classify`` names the scene for its prompts.
    Without a generator, the surround is the photo's soft colour profile.

    ``inpainter`` (LaMa, see inpaint.py) fills what foreground objects hide,
    and outpaints the frame border when there is no generated surround.
    """
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
    # Snap depth boundaries onto the photo's colour edges (see guided_filter):
    # silhouettes are classified per pixel, so a depth edge one or two pixels
    # off its colour edge shows as a striped halo when the viewer moves.
    luminance = cv2.cvtColor(photo.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    normalized = np.clip(
        guided_filter(luminance, normalized, max(2, round(max(width, height) / 160)), 4e-3), 0.0, 1.0
    )
    normalized[sky] = 0.0
    zenith, horizon = sky_colors(photo.rgb, sky)
    # Below-horizon dome colour: the median of the bottom band, so the dome,
    # the frame fade and the ground disc all agree on what "ground" looks like.
    bottom = photo.rgb[int(height * 0.75) :][~sky[int(height * 0.75) :]]
    ground_tone = (
        np.median(bottom, axis=0).astype(np.float32) / 255.0 if len(bottom) else horizon * 0.6
    )

    surround = None
    ground_scene = None
    ground_prior = True
    planes: list = []
    if generator is not None:
        from .surround import build_surround

        scene_disparity = to_scene_disparity(normalized, profile.far)
        scene_disparity[sky] = 0.0
        surround = build_surround(
            photo.rgb,
            intrinsics,
            scene_disparity,
            generator,
            estimator.estimate,
            classify,
            progress=lambda fraction: report("generate", 0.05 + 0.45 * fraction),
            cache_dir=surround_cache,
        )
        mark("generate")
        from .explore import floor_disparity_equirect, ground_height

        # Nothing below the horizon is farther than the ground: a depth
        # network reads water reflections (and reflected sky) as far away,
        # which tore lakes and wet streets open the moment you walked. Not
        # from a height, though: there the land below the horizon is far
        # below, and the prior flattened a valley into a plain at your feet.
        ground_scene = ground_height(surround.disparity)
        ground_prior = not surround.scene.elevated
        if ground_scene is not None:
            if ground_prior:
                floor = floor_disparity_equirect(*surround.disparity.shape, ground_scene)
                surround.disparity = np.maximum(surround.disparity, floor)
            planes = _panorama_planes(surround.disparity, ground_scene)
            surround.disparity = _snap_panorama(surround.disparity, planes)

    if surround is not None:
        # The generated world continues the photo on every side, so its frame
        # needs no padding: the surround mesh meets the photo's edge.
        pad_y = pad_x = 0
    else:
        pad_y, pad_x = pad_size(height, width, profile.pad_fraction)
    if intrinsics.max_pad_x() is not None:
        # A panorama's sides must not be extended past 360°; a full 360° one
        # gets no side padding at all, because its edges meet.
        pad_x = min(pad_x, intrinsics.max_pad_x())
    canvas_intrinsics = intrinsics.padded(pad_x, pad_y)
    # The 360° surround the viewer sees when looking away from the photo. The
    # frame's padding fades into exactly this, so the photo's edge meets the
    # surround without a seam.
    if surround is not None:
        environment = surround.texture(profile.surround_size)
    else:
        environment = environment_map(
            photo.rgb, intrinsics, profile.environment_size, zenith, ground_tone
        )
    # Sampled on a quarter-resolution ray grid: the surround is blurry, so
    # per-pixel rays would cost ~10x more for an identical result.
    canvas_size = (width + 2 * pad_x, height + 2 * pad_y)
    coarse_w, coarse_h = max(2, canvas_size[0] // 4), max(2, canvas_size[1] // 4)
    coarse_v, coarse_u = np.mgrid[0:coarse_h, 0:coarse_w].astype(np.float32)
    coarse_rays = canvas_intrinsics.rays(
        (coarse_u + 0.5) * canvas_size[0] / coarse_w, (coarse_v + 0.5) * canvas_size[1] / coarse_h
    )
    coarse_rays /= np.linalg.norm(coarse_rays, axis=-1, keepdims=True)
    fade = cv2.resize(
        sample_environment(environment, coarse_rays), canvas_size, interpolation=cv2.INTER_LINEAR
    )
    canvas = extend_frame(
        photo.rgb,
        normalized,
        sky,
        (pad_y, pad_x),
        fade_to=fade,
        inpainter=inpainter if surround is None else None,
    )
    grid = make_grid(canvas.rgb.shape[1], canvas.rgb.shape[0], profile.grid_long)
    pixel_disparity = to_scene_disparity(canvas.disparity, profile.far)
    # Collapse soft depth ramps at occlusions into single steps. Mesh and
    # silhouette mask both read this same map, so they agree on every edge.
    pixel_disparity = sharpen_depth_edges(
        pixel_disparity, canvas.rgb, profile.edge_disparity_jump, grid.stride
    )
    if ground_scene is not None and ground_prior:
        # The ground prior again, on the photo: a ray below the horizon meets
        # the ground at depth h / -y, so its disparity is at least -y / h.
        pixel_rows = np.arange(canvas.rgb.shape[0], dtype=np.float32)
        ray_y = -(pixel_rows - canvas_intrinsics.cy) / canvas_intrinsics.fy
        floor_rows = np.where(ray_y < -1e-3, -ray_y / ground_scene * 0.9, 0.0).astype(np.float32)
        pixel_disparity = np.maximum(pixel_disparity, floor_rows[:, None])
    if planes:
        # The same planes the panorama snapped to: walls, floor and ceiling
        # stay one flat surface from the photo into the generated world.
        from .planes import snap_depth

        pixel_rows, pixel_columns = np.mgrid[0 : canvas.rgb.shape[0], 0 : canvas.rgb.shape[1]].astype(np.float32)
        pixel_rays = canvas_intrinsics.rays(pixel_columns + 0.5, pixel_rows + 0.5).astype(np.float32)
        solid = ~canvas.sky & (pixel_disparity > 1e-6)
        snapped = snap_depth(np.zeros(3), pixel_rays, np.where(solid, 1.0 / np.maximum(pixel_disparity, 1e-6), 0.0),
                             planes, editable=solid)
        pixel_disparity = np.where(solid, 1.0 / np.maximum(snapped, 1e-6), pixel_disparity).astype(np.float32)
    # A 3x3 median at grid resolution removes single-vertex depth spikes
    # (grass blades, wires narrower than a grid cell) that would otherwise
    # shred the near field into slivers, while real silhouettes, which span
    # several vertices, keep their jump.
    disparity = cv2.medianBlur(grid.sample(pixel_disparity).astype(np.float32), 3)
    sky_grid = grid.sample(canvas.sky)
    # With a generated surround, the photo's meshes stop short of its border:
    # the surround mesh draws that ring, where generated content fades into
    # the photo, so no hard seam is left at the frame.
    ring_grid = np.zeros_like(sky_grid)
    if surround is not None:
        from .surround import photo_interior

        rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
        interior = photo_interior(intrinsics, columns, rows, np.ones((height, width), bool))
        ring_grid = ~grid.sample(interior)
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
        inpainter=inpainter,
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
    mark("environment")

    # -- 5. meshes --------------------------------------------------------
    meshes: list[MeshData] = []
    dome_positions, dome_uvs, dome_indices = environment_dome(
        profile.far * DOME_RADIUS, profile.sky_sphere_segments
    )
    meshes.append(
        MeshData("dome", dome_positions, dome_indices, uvs=dome_uvs, texture="environment", fog=False)
    )

    sky_layer = None
    if sky_grid.any():
        rays = canvas_intrinsics.rays(grid.u, grid.v)
        sky_layer = sky_mesh(rays, uvs, sky_grid & ~ring_grid, profile.far * SKY_RADIUS, texture="photo")
        if sky_layer:
            meshes.append(sky_layer)

    if surround is not None:
        layer = surround_layer(
            # Cut only where the far side is ~2.5x farther (see surround_mesh).
            surround, intrinsics, profile.surround_grid, SURROUND_RELATIVE_JUMP, profile.far
        )
        if layer is not None:
            meshes.append(layer)
    elif ground.accepted:
        # Only as far as the floor the photo actually shows, a few times
        # over. A disc reaching the horizon builds an infinite flat plain
        # through walls and hills (a slot canyon's river floor became one).
        # Past the rim, the dome's lower half (the same colours) takes over.
        floor = positions[ground_sample].reshape(-1, 3)
        foot = -ground.normal * ground.offset
        offsets = floor - foot
        across = offsets - np.outer(offsets @ ground.normal, ground.normal)
        reach = float(np.percentile(np.linalg.norm(across, axis=1), 90)) if len(floor) else profile.far
        disc_radius = float(np.clip(2.5 * reach, 0.25 * profile.far, GROUND_RADIUS * profile.far))
        disc_positions, disc_colors, disc_indices = ground_disc_mesh(
            ground,
            disc_radius,
            # Enough segments for the surround's colours to vary around the disc.
            max(48, 2 * profile.sky_sphere_segments),
            horizon,
            # Deeper than the photo ground's own depth noise (~2-3% of camera
            # height on real photos), so the disc never pokes through it. The
            # viewer adds a slope-scaled polygon offset on top.
            sink=0.05 * abs(ground.offset),
            environment=environment,
        )
        # Unfogged: its colours already come from the (unfogged) surround, and
        # fogging only one of the two drew a visible line where they meet.
        meshes.append(MeshData("ground", disc_positions, disc_indices, colors=disc_colors, fog=False))

    background_positions = unproject(
        grid, canvas_intrinsics, BACKGROUND_PUSH / background.disparity
    )
    background_mesh = surface_mesh(
        "background",
        background_positions,
        uvs,
        background.disparity,
        background.region & ~ring_grid,
        profile.edge_disparity_jump * 2.0,
        texture="background",
    )
    if background_mesh:
        meshes.append(background_mesh)

    grid_rays = canvas_intrinsics.rays(grid.u, grid.v)
    foreground, silhouettes, cell_edge, cell_mid = foreground_mesh(
        "foreground", grid_rays, uvs, disparity, ~sky_grid & ~ring_grid, profile.edge_disparity_jump, texture="photo"
    )
    keep_mask = silhouette_mask(pixel_disparity, cell_edge, cell_mid)
    # Drop isolated keep/discard pixels so silhouettes are clean curves.
    keep_mask = cv2.medianBlur(keep_mask.astype(np.uint8) * 255, 5) > 127
    if silhouettes is not None:
        silhouettes.mask = "photo_mask"
    candidates = int(
        np.count_nonzero((~sky_grid).reshape(-1)[grid_triangles(grid.height, grid.width)].any(axis=1))
    )
    if foreground is None and silhouettes is None and sky_layer is None:
        # Every image must produce a world. If the depth map left nothing
        # (pathological input: pure noise, a uniform frame), fall back to the
        # photo as one uncut surface rather than failing.
        warnings.append("depth gave no coherent surface; the photo is shown uncut")
        foreground = surface_mesh(
            "foreground", positions, uvs, disparity, np.ones_like(sky_grid), np.inf, texture="photo"
        )
        candidates = foreground.triangle_count
    if foreground is not None:
        meshes.append(foreground)
    if silhouettes is not None:
        meshes.append(silhouettes)
    if foreground is None and silhouettes is None and sky_layer is not None:
        warnings.append("the photo is all sky; it is shown on the sky sphere with no foreground")
    mark("mesh")

    # -- 5b. exploration: generate what walking into the world reveals -------
    scale = 1.0
    exploration = None
    land = land_mask(surround, profile.far) if surround is not None else None
    if surround is not None and profile.explore_stations > 0:
        from . import surround as surround_module
        from .explore import explore_world
        from .render import RendererUnavailable

        images = {
            "photo": canvas.rgb,
            "environment": environment,
            "background": background.rgb,
            "photo_mask": np.where(keep_mask, 255, 0).astype(np.uint8),
            "surround_mask": np.where(land, 255, 0).astype(np.uint8),
        }
        if sky.shape == canvas.rgb.shape[:2]:
            images["photo_land"] = np.where(sky, 0, 255).astype(np.uint8)
        try:
            scale, exploration = explore_world(
                meshes,
                images,
                cv2.resize(land.astype(np.uint8), environment.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0,
                surround.disparity,
                ground_scene if ground_scene is not None else 1.0,
                generator,
                estimator.estimate,
                surround.scene,
                profile.explore_stations,
                progress=lambda fraction: report("explore", 0.55 + 0.42 * fraction),
                planes=planes,
                ground_prior=ground_prior,
                panorama_size=surround_module.PANORAMA_SIZE,
                grid=profile.surround_grid,
                jump=SURROUND_RELATIVE_JUMP,
                far=profile.far,
            )
        except RendererUnavailable as exc:
            warnings.append(f"no exploration (the world is seen from one point): {exc}")
        if exploration is not None:
            dome = next((m for m in meshes if m.name == "dome"), None)
            for bubble in exploration.bubbles:
                meshes.append(MeshData(
                    bubble.name, bubble.positions, bubble.indices, uvs=bubble.uvs,
                    texture=bubble.name, mask=f"{bubble.name}_mask",
                ))
                if dome is not None:
                    # Behind each bubble its own panorama at infinity, as the
                    # start point's dome is behind its surround: where the
                    # bubble is cut at a depth edge, the same colours show.
                    # The start point's dome, seen from a station, put
                    # bright blobs of the wrong scenery in every cut.
                    meshes.append(MeshData(
                        f"{bubble.name}_dome", (dome.positions + bubble.station).astype(np.float32), dome.indices,
                        uvs=dome.uvs, texture=bubble.name, fog=False,
                    ))
        mark("explore")

    # -- 6. export --------------------------------------------------------
    textures: dict[str, tuple[bytes, list[int], str]] = {
        "photo": _encode_jpeg(canvas.rgb, profile.texture_long_side, profile.jpeg_quality),
        # The generated surround is real content; the profile surround is
        # blurry by construction, so a lower JPEG quality costs nothing there.
        "environment": _encode_jpeg(
            environment,
            max(environment.shape[:2]),
            profile.jpeg_quality if surround is not None else 80,
        ),
    }
    if silhouettes is not None:
        textures["photo_mask"] = _encode_mask(keep_mask, profile.texture_long_side)
    if surround is not None and any(m.name == "surround" for m in meshes):
        textures["surround_mask"] = _encode_mask(land, 2048)
    if exploration is not None:
        for bubble in exploration.bubbles:
            textures[bubble.name] = _encode_jpeg(bubble.texture, profile.surround_size[0], profile.jpeg_quality)
            textures[f"{bubble.name}_mask"] = _encode_mask(bubble.mask, 2048)
    if background_mesh:
        # Full resolution: the far-side texels of every silhouette are shown
        # from this layer, so it must be as sharp as the photo.
        textures["background"] = _encode_jpeg(
            background.rgb, profile.texture_long_side, profile.jpeg_quality
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
        "inpainting": inpainter.describe() if inpainter is not None else {"inpainter": "none"},
        "surround": surround.manifest() if surround is not None else {"generator": "none"},
        "explore": _explore_summary(exploration, scale),
        "planes": [plane.scaled(scale).as_dict() for plane in planes],
        # Each station's bubble and dome, and where it stands (metres). The
        # viewer draws only the current station's meshes; every other mesh but
        # the sky belongs to the start point (render.py station_fade).
        "overlays": {
            name: {"station": [round(float(v), 3) for v in bubble.station]}
            for bubble in (exploration.bubbles if exploration is not None else [])
            for name in (bubble.name, f"{bubble.name}_dome")
        },
        "camera": {
            **intrinsics.as_dict(),
            "position": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, -1.0],
            "up": [0.0, 1.0, 0.0],
            "near_clip": 0.05,
            "far_clip": round(profile.far * DOME_RADIUS * 1.5 * scale, 3),
        },
        "navigation": _walk_bounds(exploration) if exploration is not None else navigation_bounds(depth, valid),
        "environment": {
            "sky_fraction": round(sky_fraction, 4),
            "zenith": _rgb01(zenith),
            "horizon": _rgb01(horizon),
            "up": [round(float(v), 4) for v in up],
            # Fog only where the photo is foggy: on a clear day it washed the
            # generated hills behind you out to a white haze, cut off along
            # the mesh's cells against the unfogged sky. The single-viewpoint
            # world keeps it, to hide where its mesh meets the dome.
            "fog": {
                "color": _rgb01(horizon),
                "density": round(fog_density(profile.far * scale), 6)
                if surround is None or surround.scene.lighting == "in fog" else 0.0,
            },
            "ground": ground.as_dict(),
            "radii": {
                "sky": profile.far * SKY_RADIUS * scale,
                "dome": profile.far * DOME_RADIUS * scale,
                "ground": round(disc_radius, 3) if ground.accepted and surround is None else 0.0,
            },
        },
        "layers": {mesh.name: mesh.stats() for mesh in meshes},
        "grid": [grid.width, grid.height],
        "quality": {
            # Share of non-sky grid triangles kept after occlusion cutting. Low
            # values mean a torn, sliver-heavy surface.
            "foreground_coverage": round(
                sum(m.triangle_count for m in (foreground, silhouettes) if m is not None) / max(1, candidates),
                4,
            ),
            # Share of grid cells that are occlusion edges (rendered with the
            # silhouette mask), and of texels the mask discards.
            "edge_cell_fraction": round(float(cell_edge.mean()), 4),
            "masked_texel_fraction": round(float(1.0 - keep_mask.mean()), 4),
            "cut_vertices_fraction": round(float(near.mean()), 4),
        },
        "textures": {
            key: {"size": size, "bytes": len(data), "mime": mime}
            for key, (data, size, mime) in textures.items()
        },
        "timings_ms": timings,
        "warnings": warnings,
    }

    builder = GlbBuilder()
    texture_index = {key: builder.add_texture(data, mime) for key, (data, _, mime) in textures.items()}
    for mesh in meshes:
        builder.add_mesh(
            mesh.name,
            mesh.positions,
            mesh.indices,
            uvs=mesh.uvs,
            colors=mesh.colors,
            texture=texture_index.get(mesh.texture) if mesh.texture else None,
            mask=texture_index.get(mesh.mask) if mesh.mask else None,
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
