"""The generated world around the photo: panorama, its depth, and its mesh.

Stages (each in its own module):

1. scene.py: what kind of place this is, for the prompts.
2. generate.py: the 360° panorama, the photo kept pixel-exact inside it.
3. panorama.stitch_disparity: depth for every direction, each generated view
   aligned to the photo's own depth.
4. here: the closed surround mesh, which meets the photo's mesh at its edge
   at the same depth (inside the photo's frame both read the photo's depth).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .camera import Intrinsics
from .generate import Completion, ViewBackend, complete_panorama, view_plan
from .mesh import MeshData, surround_mesh
from .panorama import (
    equirect_directions,
    photo_disparity_to_equirect,
    photo_to_equirect,
    project_to_photo,
    sample_equirect,
)
from .scene import GENERIC, Scene

logger = logging.getLogger(__name__)

#: Bump when generation changes enough that cached panoramas are stale.
GENERATION_VERSION = "9"

#: Generation canvas. Views are a quarter of its width: 1024 px (SDXL's
#: native size) at 100° give ~10 px/degree, which a 4096-wide
#: equirectangular image holds.
PANORAMA_SIZE = (4096, 2048)
#: Depth is stitched at a coarser size: the mesh samples it at grid density.
DEPTH_SIZE = (1024, 512)
DEPTH_VIEW_SIZE = 518  # Depth Anything's native input
#: Width of the ring just inside the photo's border where generated content
#: fades into the photo. The photo's own meshes stop at the ring and the
#: surround mesh draws it, so no hard seam is left at the frame. It matches
#: the half-band over which generate.complete_panorama fades into the photo.
RING_DEG = 2.5


def ring_pixels(camera: Intrinsics) -> int:
    """The edge ring's width in photo pixels."""
    return max(2, int(round(camera.fx * np.radians(RING_DEG))))


def photo_interior(camera: Intrinsics, u: np.ndarray, v: np.ndarray, inside: np.ndarray, extra: float = 0.0) -> np.ndarray:
    """Photo pixels at least the ring (plus ``extra`` pixels) from the
    frame's border. A full 360° panorama has no left or right border."""
    ring = ring_pixels(camera) + extra
    keep = inside & (v >= ring) & (v <= camera.height - 1 - ring)
    if not camera.wraps:
        keep &= (u >= ring) & (u <= camera.width - 1 - ring)
    return keep


@dataclass
class Surround:
    panorama: np.ndarray  # (H, W, 3) uint8, PANORAMA_SIZE
    disparity: np.ndarray  # DEPTH_SIZE inverse distance in scene units; 0 = sky
    scene: Scene
    completion: Completion
    timings: dict[str, float] = field(default_factory=dict)
    backend: dict[str, str] = field(default_factory=dict)

    def texture(self, size: tuple[int, int]) -> np.ndarray:
        return cv2.resize(self.panorama, size, interpolation=cv2.INTER_AREA)

    def manifest(self) -> dict[str, Any]:
        return {
            **self.backend,
            "scene": self.scene.as_dict(),
            "views": len(self.completion.views),
            "generated_fraction": round(float(self.completion.generated.mean()), 4),
            "timings_ms": {k: round(v * 1000.0, 1) for k, v in self.timings.items()},
        }


def build_surround(
    rgb: np.ndarray,
    camera: Intrinsics,
    scene_disparity: np.ndarray,
    backend: ViewBackend,
    estimate: Callable[[np.ndarray], np.ndarray],
    classify: Callable[[np.ndarray], Scene] | None = None,
    progress: Callable[[float], None] | None = None,
    cache_dir: Path | None = None,
) -> Surround:
    """Generate the world around ``rgb``.

    Args:
        scene_disparity: the photo's disparity in scene units (inverse depth
            along the camera axis, nearest surface ~1, sky 0).
        estimate: monocular depth for an RGB view (affine-invariant disparity).
        classify: photo -> Scene; a generic outdoor scene when not given.
        cache_dir: keep generated panoramas here, keyed by the photo, camera,
            model and prompt. The same photo at another quality, or after a
            change to depth or meshing, then skips the minutes of generation.
    """
    timings: dict[str, float] = {}
    clock = time.perf_counter()
    scene = classify(rgb) if classify else GENERIC
    timings["scene"] = time.perf_counter() - clock
    describe = getattr(backend, "describe", None)
    backend_info = describe() if describe else {}

    clock = time.perf_counter()
    cached = _load_cached(cache_dir, rgb, camera, scene, backend_info)
    if cached is not None:
        panorama, completion = cached
        timings["cache"] = time.perf_counter() - clock
    else:
        panorama, known = photo_to_equirect(rgb, camera, *PANORAMA_SIZE)
        u, v, inside = project_to_photo(camera, equirect_directions(*PANORAMA_SIZE))
        completion = complete_panorama(
            panorama,
            known,
            scene,
            backend,
            view_size=PANORAMA_SIZE[0] // 4,
            progress=(lambda done, total: progress(done / total)) if progress else None,
            protected=photo_interior(camera, u, v, inside & known),
        )
        timings["generate"] = time.perf_counter() - clock
        _store_cached(cache_dir, rgb, camera, scene, backend_info, completion)

    clock = time.perf_counter()
    known_disparity, inside = photo_disparity_to_equirect(scene_disparity, camera, *DEPTH_SIZE)
    small = cv2.resize(panorama, DEPTH_SIZE, interpolation=cv2.INTER_AREA)
    disparity = stitch_and_clean(small, known_disparity, inside, estimate)
    timings["depth"] = time.perf_counter() - clock
    return Surround(panorama, disparity, scene, completion, timings, backend_info)


def _cache_key(rgb: np.ndarray, camera: Intrinsics, scene: Scene, backend: dict[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(rgb).tobytes())
    # Only what changes the generated pixels: the viewpoint changes depth, not
    # prompts, so a panorama stays valid across it.
    prompts = {k: v for k, v in asdict(scene).items() if k not in ("elevated", "location")}
    digest.update(json.dumps(
        [GENERATION_VERSION, PANORAMA_SIZE, camera.as_dict(), prompts, backend], sort_keys=True
    ).encode())
    return digest.hexdigest()[:32]


def _load_cached(cache_dir, rgb, camera, scene, backend) -> tuple[np.ndarray, Completion] | None:
    if cache_dir is None:
        return None
    entry = Path(cache_dir) / _cache_key(rgb, camera, scene, backend)
    try:
        panorama = cv2.cvtColor(cv2.imread(str(entry / "panorama.png")), cv2.COLOR_BGR2RGB)
        generated = np.load(entry / "generated.npy")
    except Exception:  # missing or unreadable: generate afresh
        return None
    logger.info("Reusing the generated surround from %s", entry.name[:8])
    return panorama, Completion(panorama, generated, [], 0.0)


def _store_cached(cache_dir, rgb, camera, scene, backend, completion: Completion) -> None:
    if cache_dir is None:
        return
    entry = Path(cache_dir) / _cache_key(rgb, camera, scene, backend)
    try:
        entry.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(entry / "panorama.png"), cv2.cvtColor(completion.panorama, cv2.COLOR_RGB2BGR))
        np.save(entry / "generated.npy", completion.generated)
    except OSError as exc:  # a cache is never worth failing a build over
        logger.warning("Could not cache the generated surround: %s", exc)


def stitch_and_clean(
    panorama: np.ndarray,
    known_disparity: np.ndarray,
    known: np.ndarray,
    estimate: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    from .panorama import separate_sky, stitch_disparity

    disparity = stitch_disparity(panorama, known_disparity, known, view_plan(DEPTH_VIEW_SIZE), estimate)
    # A 5x5 median removes seam speckle between views; the photo region keeps
    # its own (already conditioned) values.
    cleaned = cv2.medianBlur(disparity.astype(np.float32), 5)
    disparity = np.where(known, disparity, cleaned).astype(np.float32)
    # The sky the depth network glued to trees and posts, back to sky.
    return separate_sky(disparity, panorama, editable=~known)


def surround_layer(
    surround: Surround,
    camera: Intrinsics,
    grid: tuple[int, int],
    jump: float,
    far: float,
    texture: str = "environment",
) -> MeshData | None:
    """The 360° mesh, minus the photo's own frame (the photo's mesh draws
    that), overlapping it by about one surround cell so no crack shows."""
    columns, rows = grid
    azimuth = np.linspace(-np.pi, np.pi, columns + 1, dtype=np.float32)
    elevation = np.linspace(np.pi / 2, -np.pi / 2, rows, dtype=np.float32)
    a, e = np.meshgrid(azimuth, elevation)
    directions = np.stack([np.sin(a) * np.cos(e), np.sin(e), -np.cos(a) * np.cos(e)], axis=-1)
    disparity = sample_equirect(surround.disparity, directions, cv2.INTER_NEAREST)
    disparity = cv2.medianBlur(disparity.astype(np.float32), 3)
    u, v, inside = project_to_photo(camera, directions)
    # The photo's meshes stop at the edge ring; the surround draws the ring
    # and overlaps them by one surround cell (in photo pixels).
    cell = max(1.0, float(np.tan(2 * np.pi / columns) * camera.fx))
    exclude = photo_interior(camera, u, v, inside, extra=cell)
    mesh = surround_mesh(
        "surround",
        disparity,
        exclude,
        jump,
        # Farther than the sky sphere is sky or haze: the dome shows it.
        min_disparity=sky_disparity(far),
        texture=texture,
    )
    if mesh is not None:
        mesh.mask = "surround_mask"
    return mesh


def sky_disparity(far: float) -> float:
    """Inverse distance below which the surround is sky (the dome draws it)."""
    return 1.0 / (2.0 * far)


def land_mask(surround: Surround, far: float, size: tuple[int, int] = (2048, 1024)) -> np.ndarray:
    """Equirectangular keep-mask for the surround mesh: land is kept, sky is
    discarded, so skylines are drawn per pixel rather than per grid cell."""
    from .mesh import drop_islands

    land = drop_islands(surround.disparity >= sky_disparity(far)).astype(np.float32)
    land = cv2.resize(land, size, interpolation=cv2.INTER_LINEAR) >= 0.5
    # A pixel inside the skyline: the depth behind this mask is coarser than
    # the texture, and sky texels kept on land-depth geometry showed as bright
    # streaks along ridges once the eye moved.
    return cv2.erode(land.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
