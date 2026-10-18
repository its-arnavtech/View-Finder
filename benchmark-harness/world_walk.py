"""Walk through a built world: a contact sheet of eye-height views along a path.

The single-viewpoint renders (world_preview.py) cannot show whether a world
holds up when you walk into it, which is how the user judges it. This renders
a world's GLB on the GPU (viewfinder.world.render, milliseconds per frame)
from the start point and from points 3 m and 6.5 m into the scene, turning
round at each, the way a visitor would.

    python benchmark-harness/world_walk.py local-data/worlds/<id> --out walk.png [--holes]

``--holes`` paints in magenta every pixel where the dome shows through in a
direction that should hold land: what exploration has not filled yet.
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from viewfinder.world.gltf import read_glb  # noqa: E402
from viewfinder.world.render import BACKDROP, DOME, HOLE, SURFACE, Camera, WorldRenderer  # noqa: E402

PATH = (
    ("start: ahead", (0.0, 0.0, 0.0), 0.0),
    ("start: left", (0.0, 0.0, 0.0), -90.0),
    ("start: behind", (0.0, 0.0, 0.0), 180.0),
    ("3 m in: ahead", (0.0, 0.0, -3.0), 0.0),
    ("3 m in: left", (0.0, 0.0, -3.0), -90.0),
    ("3 m in: right", (0.0, 0.0, -3.0), 90.0),
    ("3 m in: back to start", (0.0, 0.0, -3.0), 180.0),
    ("6.5 m in: ahead", (0.0, 0.0, -6.5), 0.0),
    ("6.5 m in: back", (0.0, 0.0, -6.5), 180.0),
    ("2 m right: ahead", (2.0, 0.0, 0.0), 0.0),
    ("2 m left: ahead", (-2.0, 0.0, 0.0), 0.0),
    ("3 m behind: ahead", (0.0, 0.0, 3.0), 0.0),
)


def path_for(manifest: dict) -> tuple:
    """The world's own walk: the start, then each standing point it was
    explored from (where the viewer lets a visitor go), looking ahead and
    back. A fixed path walks into the walls of a narrow canyon or a room."""
    stations = manifest.get("navigation", {}).get("stations") or [
        [s["position"][0], s["position"][2]] if isinstance(s, dict) else [s[0], s[2]]
        for s in manifest.get("explore", {}).get("stations", [])
    ]
    if not stations:
        return PATH
    import math

    path = [("start: ahead", (0.0, 0.0, 0.0), 0.0), ("start: left", (0.0, 0.0, 0.0), -90.0),
            ("start: behind", (0.0, 0.0, 0.0), 180.0)]
    others = [s for s in stations if abs(s[0]) + abs(s[1]) > 1e-6]
    others.sort(key=lambda s: math.hypot(s[0], s[1]))
    picked = others[:2] + others[-1:] if len(others) > 3 else others
    for x, z in picked:
        heading = math.degrees(math.atan2(x, -z))
        where = f"({x:+.1f}, {z:+.1f}) m"
        path.append((f"{where}: walking on", (x, 0.0, z), heading))
        path.append((f"{where}: left", (x, 0.0, z), heading - 90.0))
        path.append((f"{where}: back toward start", (x, 0.0, z), heading + 180.0))
    return tuple(path[:12])


def load(world: Path) -> tuple[WorldRenderer, dict, bool]:
    path = world / "world.glb" if world.is_dir() else world
    data = path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    glb = read_glb(data)
    manifest = glb.extras["viewfinder_world"]
    images = [
        np.asarray(Image.open(io.BytesIO(glb.image_bytes(i))).convert("RGB"))
        for i in range(len(glb.document.get("images", [])))
    ]
    return glb, manifest, images


def build_renderer(glb, images, holes: bool) -> WorldRenderer:
    renderer = WorldRenderer()
    overlays = glb.extras["viewfinder_world"].get("overlays", {})
    land = None
    for mesh in glb.document["meshes"]:
        if mesh["name"] == "surround":
            material = glb.document["materials"][mesh["primitives"][0]["material"]]
            ref = material.get("extras", {}).get("viewfinder_mask")
            if ref is not None:
                land = images[glb.document["textures"][ref]["source"]][..., 0]
    for mesh in glb.document["meshes"]:
        primitive = mesh["primitives"][0]
        attributes = primitive["attributes"]
        if "TEXCOORD_0" not in attributes:
            continue
        material = glb.document["materials"][primitive["material"]]
        texture = images[glb.document["textures"][material["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]]
        ref = material.get("extras", {}).get("viewfinder_mask")
        mask = images[glb.document["textures"][ref]["source"]][..., 0] if ref is not None else None
        kind = SURFACE
        if mesh["name"] == "sky" or mesh["name"].endswith("_dome"):
            kind = BACKDROP
        elif mesh["name"] == "dome":
            kind = DOME if holes and land is not None else BACKDROP
        renderer.add(
            glb.accessor(attributes["POSITION"]), glb.accessor(attributes["TEXCOORD_0"]),
            glb.accessor(primitive["indices"]), texture, mask, kind, land=land if kind == DOME else None,
            # Every mesh but the sky belongs to a station: its bubble's (and
            # dome's), or the start point's.
            station=None if mesh["name"] == "sky" else overlays.get(mesh["name"], {}).get("station", [0.0, 0.0, 0.0]),
        )
    return renderer


def walk_sheet(world: Path, holes: bool = False, width: int = 640, height: int = 400, fov: float = 75.0) -> Image.Image:
    glb, manifest, images = load(world)
    renderer = build_renderer(glb, images, holes)
    explore = manifest.get("explore", {})
    title = (
        f"{world.name[:8]}  {manifest['profile']}  views generated: surround {manifest.get('surround', {}).get('views', 0)}"
        f" + walking {explore.get('views', 0)}  ({manifest.get('glb_bytes', 0) / 1e6:.1f} MB)"
    )
    tiles = []
    for label, eye, yaw in path_for(manifest):
        frame = renderer.render(Camera.looking(np.array(eye), yaw, -5.0, fov, width, height))
        rgb = frame.rgb.copy()
        if holes:
            rgb[frame.coverage == HOLE] = (255, 0, 255)
        tiles.append((label, rgb))
    renderer.release()
    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + 18) + 20), (20, 20, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 4), title, fill=(230, 230, 230))
    for index, (label, rgb) in enumerate(tiles):
        x, y = (index % columns) * width, 20 + (index // columns) * (height + 18)
        sheet.paste(Image.fromarray(rgb), (x, y + 16))
        draw.text((x + 6, y + 2), label, fill=(210, 210, 210))
    return sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path, help="a world directory (local-data/worlds/<id>) or a .glb")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holes", action="store_true", help="paint unfilled land directions magenta")
    args = parser.parse_args(argv)
    walk_sheet(args.world, args.holes).save(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
