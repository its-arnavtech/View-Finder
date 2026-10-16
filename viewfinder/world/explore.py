"""Walking into the world: a generated 360° bubble at every standing point.

From the start point the world is complete: the photo and its generated
panorama, meshed from that point (the *start bubble*). A few metres away
that mesh breaks down. The photo's layers tear at every depth edge, the
panorama stretches and melts, and nothing was ever generated behind things.
Patching only the holes and blurs was not enough: the torn and melted parts
are sharp, so no test finds them, and a dozen patches painted from different
places disagreed by centimetres about every wall and shredded each other.

So every station gets a bubble of its own, as the start point has:

1. render the nearest existing bubble from the station, all the way round,
   on the GPU (render.py), into an equirectangular canvas;
2. let the image model fill what is missing outright, and repaint everything
   near image-to-image (the layout stays; tears, smears and melt go). Far
   scenery and sky are left as they are, so they match the start view;
3. estimate depth for the repainted canvas, each view aligned to the
   rendered depth (right in scale, torn in detail), then the ground and
   plane priors;
4. mesh it from the station as one continuous surface, like the panorama.

The viewer draws only the nearest bubble (render.py ``station_fade``), so
wherever you stand you see one coherent surface made for that spot.

All positions are in metres, with the start camera at the origin, +Y up
and the eye 1.6 m above the ground.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .generate import ViewBackend, match_colours, view_plan
from .layers import propagate_background
from .panorama import equirect_directions, render_view, sample_equirect
from .render import HOLE, SKY, SOLID, Camera, WorldRenderer
from .scene import Scene

logger = logging.getLogger(__name__)

EYE_HEIGHT = 1.6  # metres
# Surfaces nearer than this (metres) are repainted at each station; farther
# ones (mountains across a lake) look the same from a few metres away and
# are kept, so they match the start view exactly. At 20 m the boundary ran
# along the horizon of open ground and broke it into a strip.
HEAL_DISTANCE = 50.0
# ...and so is anything magnified past this many screen pixels per texel.
BLUR_DENSITY = 0.5
# Image-to-image strength of the repaint (SDEdit). At 0.45-0.6 the radial
# smears of the stretched start bubble survived as "structure" (and read as
# motion blur); at 0.8 the model followed the prompt more than the picture (a
# Gothic stone corridor became a hotel hallway). Smears are wiped first
# (SMEAR_DENSITY), and the bubble's depth is estimated from the repainted
# image, so the geometry follows what was painted.
HEAL_STRENGTH = 0.7
# Every pixel gets its own strength (generate.SDXLBackend, differential
# diffusion): what nothing showed is generated outright, what was wiped
# (below) nearly so, and near surfaces are repainted at HEAL_STRENGTH plus
# this much for every repaint already behind the bubble a station starts
# from, since each one leaves the picture a little softer, rising towards
# WIPED_STRENGTH the more they are stretched (canopy stretched 1.5x from a
# few metres away kept its streaks at 0.7). A single strength per view made
# a trunk wiped down to its colours come back as a blur.
HOLE_STRENGTH = 0.98
WIPED_STRENGTH = 0.9
HOP_STRENGTH = 0.05
MAX_HEAL_STRENGTH = 0.8
# Surfaces magnified past this many screen pixels per texel (the ground
# around a station, seen from the start a few metres off, is magnified 5-10x)
# are long radial streaks, which survive even 0.8 as motion blur. They are
# wiped before the repaint: blurred down to their colour layout (road here,
# grass there; an 8-pixel blur left 100-pixel streaks pointing the same way)
# plus noise, so the model paints fresh texture over the right colours.
SMEAR_DENSITY = 0.35
# A view is generated only if this much of it needs it.
MIN_FILL_FRACTION = 0.003
MIN_HEAL_FRACTION = 0.03
# Side of the cube faces the bubble's canvas is rendered from (pixels).
FACE_SIZE = 1024
# A station starts from the bubble with the fewest repaints behind it among
# those within this many metres of the nearest. Each hop is another partial
# repaint and detail softens with every one (a tree two hops from the photo
# came out motion-blurred); a later bubble is only worth it when it is
# clearly nearer.
SOURCE_SLACK = 1.0


@dataclass
class Bubble:
    """A station's 360° mesh in world metres."""

    name: str
    station: np.ndarray  # (3,) metres
    positions: np.ndarray  # (N, 3) float32
    uvs: np.ndarray  # (N, 2) float32
    indices: np.ndarray  # (3M,) uint32
    texture: np.ndarray  # (H, W, 3) uint8, equirectangular
    mask: np.ndarray  # (H, W) bool, equirectangular: True = land (sky is the dome's)

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3


@dataclass
class Station:
    position: np.ndarray  # (3,) metres
    source: int = -1  # bubble rendered to start from (-1: the start bubble)
    views: int = 0  # views generated here
    built: bool = False  # it has a bubble: the viewer lets a visitor stand here
    seconds: float = 0.0


@dataclass
class Exploration:
    bubbles: list[Bubble] = field(default_factory=list)
    stations: list[Station] = field(default_factory=list)
    views: int = 0
    seconds: float = 0.0


# -- the ground -----------------------------------------------------------


def ground_height(disparity: np.ndarray) -> float | None:
    """Camera height above a level ground, in the panorama's units.

    A level ground at height ``h`` is seen at inverse distance
    ``sin(-elevation) / h``. Looking steeply down (the generated 'ground
    underfoot'), almost every pixel is ground, so the median of
    ``sin(-e) / disparity`` there is the height.
    """
    height, width = disparity.shape
    elevation = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    rows = elevation < -np.radians(35.0)
    values = disparity[rows]
    sines = np.repeat(np.sin(-elevation[rows])[:, None], width, axis=1)
    valid = values > 1e-6
    if valid.sum() < 100:
        return None
    return float(np.median(sines[valid] / values[valid]))


def floor_disparity_equirect(height_px: int, width_px: int, ground: float, slack: float = 0.9) -> np.ndarray:
    """Least inverse distance a level ground at ``ground`` allows per
    equirectangular pixel: nothing below the horizon is farther than the
    ground it would have to be under (``slack`` absorbs gentle slopes)."""
    elevation = np.pi / 2 - (np.arange(height_px) + 0.5) / height_px * np.pi
    floor = np.where(elevation < -np.radians(1.0), np.sin(-elevation) / ground * slack, 0.0)
    return np.repeat(floor[:, None], width_px, axis=1).astype(np.float32)


# -- stations -------------------------------------------------------------

#: Standing points as (distance m, heading deg; 0 = the photo's direction),
#: in the order the budget reaches them: walking into the photo first, then
#: to either side, then behind. Neighbours are about 3.5 m apart, so nowhere
#: on the way between two is far from the nearer.
STATION_PLAN: tuple[tuple[float, float], ...] = (
    (3.0, 0.0), (6.5, 0.0), (3.0, -60.0), (3.0, 60.0), (10.0, 0.0),
    (3.0, 180.0), (6.5, -35.0), (6.5, 35.0), (3.0, -120.0), (3.0, 120.0),
)


#: A standing point needs the panorama to see ground at least this share as
#: near as a level ground at its distance would be.
GROUND_SUPPORT = 0.5


def plan_stations(
    disparity: np.ndarray,
    scale: float,
    plan: tuple[tuple[float, float], ...] = STATION_PLAN,
    clearance: float = 0.6,
) -> list[Station]:
    """Standing points on the walkable ground, in ``plan`` order. A heading
    is walkable to ``clearance`` of the distance to the nearest obstacle at
    or above eye level (from the panorama's depth): walls, trees and
    boulders block, rocks you can step past do not. And there must be ground
    there: where the panorama sees much farther than a level ground at that
    distance would be (the edge of a cliff, a hillside over a valley), a
    standing point would hang in the air."""
    height, width = disparity.shape
    elevation = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    band = (elevation >= 0.0) & (elevation < np.radians(12.0))
    stations: list[Station] = []
    for distance, heading in plan:
        column = int(((math.radians(heading) + np.pi) / (2 * np.pi)) * width) % width
        columns = [(column + k) % width for k in range(-width // 72, width // 72 + 1)]  # +-5 deg
        nearest = float(disparity[np.ix_(band, columns)].max()) if band.any() else 0.0
        free = (1.0 / nearest) * scale if nearest > 1e-6 else float("inf")
        if distance > clearance * free:
            continue
        # Where a level ground at this distance is seen, and how far it is.
        down = math.atan2(EYE_HEIGHT, distance)
        rows = np.abs(elevation + down) < np.radians(2.0)
        seen = float(np.median(disparity[np.ix_(rows, columns)])) if rows.any() else 0.0
        expected = scale / math.hypot(distance, EYE_HEIGHT)
        if seen < GROUND_SUPPORT * expected:
            continue
        yaw = math.radians(heading)
        stations.append(Station(np.array([math.sin(yaw) * distance, 0.0, -math.cos(yaw) * distance])))
    return stations


# -- one bubble -----------------------------------------------------------


@dataclass
class Canvas:
    """The world rendered from a station into equirectangular maps."""

    rgb: np.ndarray  # (H, W, 3) uint8
    distance: np.ndarray  # (H, W) float32 metres from the eye, 0 = none
    coverage: np.ndarray  # (H, W) uint8: HOLE, SKY or SOLID
    density: np.ndarray  # (H, W) float32 texels per screen pixel


#: Cube faces as (yaw, pitch): together they see every direction.
_FACES = ((0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (-90.0, 0.0), (0.0, 90.0), (0.0, -90.0))


def render_canvas(
    renderer: WorldRenderer, eye: np.ndarray, size: tuple[int, int], show: list, face: int = FACE_SIZE
) -> Canvas:
    """Render the world from ``eye`` all the way round: six 90° cube faces,
    resampled into equirectangular maps of ``size`` (width, height)."""
    width, height = size
    directions = equirect_directions(width, height)
    rgb = np.zeros((height, width, 3), np.uint8)
    distance = np.zeros((height, width), np.float32)
    coverage = np.full((height, width), HOLE, np.uint8)
    density = np.zeros((height, width), np.float32)
    for yaw, pitch in _FACES:
        camera = Camera.looking(eye, yaw, pitch, 90.0, face, face)
        frame = renderer.render(camera, show)
        right, up, back = camera.basis()
        along = directions @ -back
        safe = np.maximum(along, 1e-6)
        x, y = (directions @ right) / safe, (directions @ up) / safe
        inside = (along > 1e-6) & (np.abs(x) <= 1.0) & (np.abs(y) <= 1.0)
        columns = (x * camera.focal + face / 2 - 0.5).astype(np.float32)
        rows = (-y * camera.focal + face / 2 - 0.5).astype(np.float32)
        border = cv2.BORDER_REPLICATE
        sampled = cv2.remap(frame.rgb, columns, rows, cv2.INTER_LINEAR, borderMode=border)
        depth = cv2.remap(frame.depth, columns, rows, cv2.INTER_NEAREST, borderMode=border)
        cover = cv2.remap(frame.coverage, columns, rows, cv2.INTER_NEAREST, borderMode=border)
        texels = cv2.remap(frame.density, columns, rows, cv2.INTER_NEAREST, borderMode=border)
        rgb[inside] = sampled[inside]
        distance[inside] = (depth / safe)[inside]  # depth is along the face's axis
        coverage[inside] = cover[inside]
        density[inside] = texels[inside]
    distance[coverage != SOLID] = 0.0
    return Canvas(rgb, distance, coverage, density)


def heal_canvas(
    canvas: Canvas,
    scene: Scene,
    backend: ViewBackend,
    view_size: int,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
    hops: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill the canvas's holes and repaint its near surfaces, view by view
    round the panorama (modifies ``canvas.rgb`` in place). ``hops`` is how
    many repaints are already behind the canvas (0: rendered from the photo
    and its panorama).

    One pass per view with a strength per pixel: holes are seeded with a
    soft continuation plus noise (generate.complete_panorama) and generated
    outright, wiped smears nearly so, near surfaces repainted lightly. Each
    view repaints what earlier views did not, and blends over their work
    across a band, so views meet without a seam.

    Returns (pixels the model filled, pixels meant for repainting, views
    generated).
    """
    rgb = canvas.rgb
    height, width = canvas.coverage.shape
    directions = equirect_directions(width, height)
    known = canvas.coverage != HOLE
    filled = np.zeros((height, width), bool)
    target = (canvas.coverage == SOLID) & ((canvas.distance < HEAL_DISTANCE) | (canvas.density < BLUR_DENSITY))
    # The sky just beyond a near silhouette is written with it: a repaint
    # moves edges by a pixel or two, and written only up to the old
    # silhouette it would leave the old edge showing beside the new one.
    edge = max(3, width // 512)
    beyond = cv2.dilate(target.astype(np.uint8), np.ones((2 * edge + 1, 2 * edge + 1), np.uint8)).astype(bool)
    editable = target | (beyond & (canvas.coverage == SKY))
    healed = np.zeros((height, width), bool)
    rng = np.random.default_rng(seed)
    plan = view_plan(view_size)
    generated_views = 0
    band = max(4, view_size // 20)
    near_strength = min(HEAL_STRENGTH + HOP_STRENGTH * hops, MAX_HEAL_STRENGTH)
    for index, view in enumerate(plan):
        known_view = render_view(known.astype(np.float32), view) > 0.999
        known_view = cv2.erode(known_view.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        hole = ~known_view
        todo = render_view((target & ~healed).astype(np.float32), view) > 0.5
        heal = todo.mean() > MIN_HEAL_FRACTION
        if not heal and hole.mean() <= MIN_FILL_FRACTION:
            if progress:
                progress(index + 1, len(plan))
            continue
        direction = "up" if view.pitch_deg > 30 else ("down" if view.pitch_deg < -30 else "around")
        if direction == "up" and not scene.enclosed:
            # Tilted up outdoors, a station repaints treetops, cliffs and roofs
            # that are already there. The panorama's "open sky, no trees,
            # no land" prompt for that direction fought them into streaks.
            direction = "around"
        image = render_view(rgb, view)
        strength = np.zeros(hole.shape, np.float32)
        if heal:
            density = render_view(np.where(target & ~healed, canvas.density, 9.0).astype(np.float32), view)
            # The more a surface is stretched from where it was made, the less
            # of it to keep: full trust at a texel per pixel or better,
            # rising to the wiped strength where it is smeared.
            stretched = np.clip((1.0 - density) / (1.0 - SMEAR_DENSITY), 0.0, 1.0)
            strength[todo] = (near_strength + (WIPED_STRENGTH - near_strength) * stretched)[todo]
            smeared = todo & (density < SMEAR_DENSITY)
            if smeared.any():
                wiped = cv2.GaussianBlur(image, (0, 0), view_size / 24).astype(np.float32)
                wiped += rng.normal(0.0, 16.0, image.shape).astype(np.float32)
                image = np.where(smeared[..., None], np.clip(wiped, 0, 255), image).astype(np.uint8)
        if hole.any():
            image = propagate_background(image, hole, known_view)
            noise = rng.normal(0.0, 12.0, image.shape).astype(np.float32)
            image = np.where(hole[..., None], np.clip(image + noise, 0, 255), image).astype(np.uint8)
        strength[hole] = HOLE_STRENGTH
        need = hole | (todo if heal else False)
        distance = cv2.distanceTransform((~need).astype(np.uint8), cv2.DIST_L2, 5)
        grown = distance <= band
        # The band around it is touched up like near surfaces, to blend in.
        strength[grown & (strength == 0)] = near_strength
        generated = backend.inpaint(
            image, grown, scene.prompt(direction), scene.negative(direction), seed + index,
            strength=float(strength[grown].max()), strength_map=strength,
        )
        generated = match_colours(generated, image, grown & known_view)
        ramp = np.clip(1.0 - distance / band, 0.0, 1.0)
        soft = (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)
        merged = (image * (1.0 - soft[..., None]) + generated * soft[..., None]).astype(np.uint8)
        # Back into the canvas: new pixels in the view's interior are taken
        # whole; near surfaces (repainted or not yet) blend by the ramp,
        # tapered at the view's border. Far scenery and sky never change.
        i, j, inside = view.project(directions)
        border = np.minimum(np.minimum(i, j), np.minimum(view.size - 1 - i, view.size - 1 - j))
        taper = np.clip(border / max(1.0, 0.04 * view.size), 0.0, 1.0)
        taper = np.where(inside, taper * taper * (3 - 2 * taper), 0.0)
        weight = cv2.remap(soft, i, j, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * taper
        sampled = cv2.remap(merged, i, j, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        new = ~known & (taper >= 1.0)
        rgb[new] = sampled[new]
        weight = np.where(editable, weight, 0.0)[..., None]
        rgb[:] = (rgb * (1.0 - weight) + sampled * weight).astype(np.uint8)
        known |= new
        filled |= new
        healed |= (weight[..., 0] > 0.95) & target
        generated_views += 1
        if progress:
            progress(index + 1, len(plan))
    if not known.all():
        rgb[:] = propagate_background(rgb, ~known, known)  # slivers no view's interior reached
        filled |= ~known
    return filled, editable, generated_views


def bubble_disparity(
    canvas: Canvas,
    repainted: np.ndarray,
    estimate: Callable[[np.ndarray], np.ndarray],
    eye: np.ndarray,
    planes: list | None,
    size: tuple[int, int],
    ground_prior: bool = True,
) -> np.ndarray:
    """Inverse distance (1/m) from the station for every direction of the
    healed canvas, at ``size`` (width, height); 0 = sky. ``ground_prior``:
    nothing below the horizon is farther than a level ground (off for views
    from a height)."""
    from .panorama import separate_sky, stitch_disparity
    from .surround import DEPTH_VIEW_SIZE

    width, height = size

    def small(image: np.ndarray, interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
        return cv2.resize(image, (width, height), interpolation=interpolation)

    solid = small(canvas.coverage) == SOLID
    sky = small(canvas.coverage) == SKY
    rendered = np.where(solid, 1.0 / np.maximum(small(canvas.distance), 1e-3), 0.0).astype(np.float32)
    changed = small(repainted.astype(np.uint8)) > 0
    # Kept: surfaces left as they were (far scenery) and the sky. Repainted
    # surfaces and new ones get estimated depth, aligned to the rendered.
    keep = (solid & ~changed) | sky
    disparity = stitch_disparity(
        small(canvas.rgb, cv2.INTER_AREA), rendered, keep, view_plan(DEPTH_VIEW_SIZE), estimate,
        anchor=solid & changed,
    )
    cleaned = cv2.medianBlur(disparity.astype(np.float32), 5)
    disparity = np.where(keep, disparity, cleaned).astype(np.float32)
    # The sky the depth network glued to trees and posts, back to sky
    # (panorama.separate_sky); far scenery kept as rendered stays as it is.
    disparity = separate_sky(disparity, small(canvas.rgb, cv2.INTER_AREA), editable=~keep)
    if ground_prior:
        # Nothing below the horizon is farther than the ground.
        floor = floor_disparity_equirect(height, width, EYE_HEIGHT)
        disparity = np.where(~keep & (disparity > 0), np.maximum(disparity, floor), disparity).astype(np.float32)
    if planes:
        from .planes import snap_depth

        directions = equirect_directions(width, height).astype(np.float32)
        land = disparity > 1e-6
        distance = np.where(land, 1.0 / np.maximum(disparity, 1e-6), 0.0).astype(np.float32)
        snapped = snap_depth(eye, directions, distance, planes, editable=land & ~keep)
        disparity = np.where(land, 1.0 / np.maximum(snapped, 1e-6), 0.0).astype(np.float32)
    return disparity


def bubble_mesh(
    name: str,
    eye: np.ndarray,
    texture: np.ndarray,
    disparity: np.ndarray,
    grid: tuple[int, int],
    jump: float,
    sky_disparity: float,
) -> Bubble | None:
    """The bubble's closed 360° mesh around ``eye`` (as the start point's
    surround), with its land keep-mask."""
    from .mesh import drop_islands, surround_mesh

    columns, rows = grid
    azimuth = np.linspace(-np.pi, np.pi, columns + 1, dtype=np.float32)
    elevation = np.linspace(np.pi / 2, -np.pi / 2, rows, dtype=np.float32)
    a, e = np.meshgrid(azimuth, elevation)
    directions = np.stack([np.sin(a) * np.cos(e), np.sin(e), -np.cos(a) * np.cos(e)], axis=-1)
    at_vertices = cv2.medianBlur(sample_equirect(disparity, directions, cv2.INTER_NEAREST).astype(np.float32), 3)
    mesh = surround_mesh(name, at_vertices, np.zeros(at_vertices.shape, bool), jump, sky_disparity, texture=name)
    if mesh is None:
        return None
    land = drop_islands(disparity >= sky_disparity).astype(np.float32)
    land = cv2.resize(land, texture.shape[1::-1], interpolation=cv2.INTER_LINEAR) >= 0.5
    land = cv2.erode(land.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return Bubble(
        name, eye.copy(), (mesh.positions + eye[None, :]).astype(np.float32), mesh.uvs, mesh.indices, texture, land,
    )


# -- the walk -------------------------------------------------------------


def choose_source(eye: np.ndarray, built: list[np.ndarray], hops: list[int]) -> int:
    """Index of the bubble a new station at ``eye`` starts from: the fewest
    repaints from the photo among those within ``SOURCE_SLACK`` of the
    nearest, then the nearest."""
    gaps = [float(np.hypot(*(eye - b)[[0, 2]])) for b in built]
    nearest = min(gaps)
    candidates = [i for i, gap in enumerate(gaps) if gap <= nearest + SOURCE_SLACK]
    return min(candidates, key=lambda i: (hops[i], gaps[i]))



def explore_world(
    meshes: list,
    images: dict[str, np.ndarray],
    land: np.ndarray,
    surround_disparity: np.ndarray,
    ground: float,
    backend: ViewBackend,
    estimate: Callable[[np.ndarray], np.ndarray],
    scene: Scene,
    budget: int,
    progress: Callable[[float], None] | None = None,
    planes: list | None = None,
    ground_prior: bool = True,
    panorama_size: tuple[int, int] = (4096, 2048),
    grid: tuple[int, int] = (384, 192),
    jump: float = 0.3,
    far: float = 1000.0,
) -> tuple[float, Exploration]:
    """Scale the world to metres (in place) and build up to ``budget``
    station bubbles.

    Args:
        meshes: the world's MeshData so far (scene units), with ``texture``
            and ``mask`` keys into ``images``. They form the start bubble.
        land: equirectangular keep-mask of the panorama (True = land).
        surround_disparity: the panorama's inverse distance (scene units).
        ground: camera height above the ground, scene units.
        panorama_size, grid, jump: each bubble's texture size, mesh grid and
            relative depth jump (as the start point's surround).
        far: distance (scene units) beyond which the panorama is sky.
        ground_prior: keep below-horizon land within a level ground (off
            for views from a height).

    Returns (metres per scene unit, the exploration).
    """
    from .render import BACKDROP, DOME, SURFACE

    scale = EYE_HEIGHT / ground
    for mesh in meshes:
        mesh.positions = (mesh.positions * scale).astype(np.float32)
    renderer = WorldRenderer()
    # Where the dome (the start point's panorama at infinity) or the photo's
    # sky layer shows land, nothing was built for this viewpoint: a hole, to
    # generate. Both land masks are widened a little: from a station the
    # ridge at infinity peeks out past the parallax-shifted 3D ridge, and
    # with its outermost pixels counted as sky the repaint read that thin
    # dark sliver as a cable strung along every ridge.
    widen = np.ones((7, 7), np.uint8)
    land_images = {"dome": cv2.dilate(np.where(land, 255, 0).astype(np.uint8), widen)}
    if "photo_land" in images:
        land_images["sky"] = cv2.dilate(images["photo_land"], widen)
    start = np.zeros(3)
    for mesh in meshes:
        if mesh.uvs is None or mesh.texture not in images:
            continue  # vertex-coloured layers (none with a surround)
        kind = SURFACE
        if mesh.name in land_images:
            kind = DOME
        elif mesh.name == "sky":
            kind = BACKDROP
        mask = images.get(mesh.mask) if mesh.mask else None
        renderer.add(
            mesh.positions, mesh.uvs, mesh.indices, images[mesh.texture], mask, kind,
            land=land_images.get(mesh.name),
            station=start if kind == SURFACE else None,
        )
    stations = plan_stations(surround_disparity, scale)
    try:
        result = explore(
            renderer, stations, backend, estimate, scene, budget, progress,
            planes=[plane.scaled(scale) for plane in (planes or [])],
            panorama_size=panorama_size, grid=grid, jump=jump, sky_disparity=1.0 / (2.0 * far * scale),
            ground_prior=ground_prior,
        )
    finally:
        renderer.release()
    return scale, result


def explore(
    renderer: WorldRenderer,
    stations: list[Station],
    backend: ViewBackend,
    estimate: Callable[[np.ndarray], np.ndarray],
    scene: Scene,
    budget: int,
    progress: Callable[[float], None] | None = None,
    seed: int = 1000,
    planes: list | None = None,
    panorama_size: tuple[int, int] = (4096, 2048),
    grid: tuple[int, int] = (384, 192),
    jump: float = 0.3,
    sky_disparity: float = 1e-3,
    ground_prior: bool = True,
) -> Exploration:
    """Build a bubble at each station in order, ``budget`` at most. Each
    starts from a nearby bubble already built (``choose_source``), so
    neighbours continue each other."""
    from .surround import DEPTH_SIZE

    started = time.perf_counter()
    result = Exploration(stations=stations)
    built: list[np.ndarray] = [np.zeros(3)]
    hops: list[int] = [0]  # repaints between each bubble and the photo
    view_size = panorama_size[0] // 4
    todo = stations[:budget]
    for number, station in enumerate(todo):
        clock = time.perf_counter()
        eye = station.position
        source = choose_source(eye, built, hops)
        station.source = source - 1
        canvas = render_canvas(renderer, eye, panorama_size, [built[source]], face=min(FACE_SIZE, panorama_size[0] // 4))

        def report(done: int, total: int, number: int = number) -> None:
            if progress:
                progress((number + done / total) / len(todo))

        filled, repainted, views = heal_canvas(
            canvas, scene, backend, view_size, seed + 100 * number, report, hops=hops[source]
        )
        disparity = bubble_disparity(canvas, filled | repainted, estimate, eye, planes, DEPTH_SIZE, ground_prior)
        bubble = bubble_mesh(f"walk{number}", eye, canvas.rgb, disparity, grid, jump, sky_disparity)
        station.views = views
        result.views += views
        if bubble is not None:
            result.bubbles.append(bubble)
            renderer.add(bubble.positions, bubble.uvs, bubble.indices, bubble.texture,
                         np.where(bubble.mask, 255, 0).astype(np.uint8), station=eye)
            built.append(eye.copy())
            hops.append(hops[source] + 1)
            station.built = True
        station.seconds = time.perf_counter() - clock
        logger.info("station %s: %d views, %.0f s", np.round(eye, 1), views, station.seconds)
    result.seconds = time.perf_counter() - started
    return result
