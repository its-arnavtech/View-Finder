"""Invariants every built world must satisfy, whatever the input photo.

Shared by the test suite and ``benchmark-harness/world_stress.py`` so "the
engine produced a usable world" means the same thing in both.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .gltf import read_glb
from .pipeline import WorldResult
from .profiles import get_profile

REQUIRED_KEYS = {
    "schema", "name", "source", "profile", "depth", "camera", "navigation",
    "environment", "layers", "textures", "timings_ms", "warnings", "quality",
}
# NFR2: a mobile world must stay a quick download on a phone connection.
MOBILE_MAX_BYTES = 4 * 1024 * 1024


def validate_world(result: WorldResult) -> list[str]:
    """Return a list of violated invariants (empty means the world is good)."""
    problems: list[str] = []
    manifest = result.manifest
    missing = REQUIRED_KEYS - set(manifest)
    if missing:
        problems.append(f"manifest missing {sorted(missing)}")
        return problems
    profile = get_profile(manifest["profile"])

    try:
        glb = read_glb(result.glb)
    except Exception as exc:  # noqa: BLE001 - any parse failure is the finding
        return [f"GLB does not parse: {exc}"]

    names = [mesh["name"] for mesh in glb.document["meshes"]]
    if "dome" not in names:
        problems.append("no dome: turning away from the photo would show nothing")
    textured = [n for n in names if n in ("foreground", "background", "sky")]
    if not textured:
        problems.append("no photo-textured layer")

    for mesh in glb.document["meshes"]:
        primitive = mesh["primitives"][0]
        positions = glb.accessor(primitive["attributes"]["POSITION"])
        if not np.isfinite(positions).all():
            problems.append(f"{mesh['name']}: non-finite positions")
        if profile.name == "mobile" and len(positions) > 0xFFFF:
            problems.append(f"{mesh['name']}: {len(positions)} vertices exceeds 16-bit indices on mobile")

    for index, image in enumerate(glb.document.get("images", [])):
        with Image.open(io.BytesIO(glb.image_bytes(index))) as decoded:
            budget = max(profile.texture_long_side, profile.environment_size[0], profile.surround_size[0])
            if max(decoded.size) > budget:
                problems.append(f"texture {index} is {decoded.size}, over the {profile.name} budget")

    if profile.name == "mobile" and len(result.glb) > MOBILE_MAX_BYTES:
        problems.append(f"mobile world is {len(result.glb) / 1e6:.1f} MB (budget 4 MB)")

    navigation = manifest["navigation"]["half_extents"]
    if not all(np.isfinite(v) and v > 0 for v in navigation):
        problems.append(f"bad navigation bounds {navigation}")
    camera = manifest["camera"]
    # A pinhole cannot reach 180°; a cylindrical panorama can wrap to 360°.
    max_hfov = 360.01 if camera.get("projection") == "cylindrical" else 179.0
    if not (1.0 < camera["hfov_deg"] <= max_hfov and 1.0 < camera["vfov_deg"] < 179.0):
        problems.append(f"implausible field of view {camera['hfov_deg']}x{camera['vfov_deg']}")
    return problems
