"""Render World Engine GLBs offline from fixed camera poses (for visual QA).

A small point-splatting rasteriser in NumPy. Every triangle is sampled
densely enough to cover its projected area, samples are z-buffered, and
backdrop layers (dome, sky) are painted first without depth, mirroring the
browser viewer. It honours the per-layer fog flags and the foreground's
discard mask, and uses the viewer's camera model (look mode with the neck
pivot, the same field-of-view rule), so a pose here matches a pose there.

It exists so visual changes can be judged repeatably across many photos and
poses without a browser:

    python benchmark-harness/world_preview.py photo.jpg [more.jpg ...] --out local-data/world-eval
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from viewfinder.world.gltf import read_glb  # noqa: E402

BACKDROP = ("dome", "sky")

# (label, yaw, pitch, position as multiples of navigation half-extents, free roam)
POSES = (
    ("front", 0.0, 0.0, (0.0, 0.0, 0.0), False),
    ("look-left", -0.6, 0.0, (0.0, 0.0, 0.0), False),
    ("look-right", 0.6, 0.0, (0.0, 0.0, 0.0), False),
    ("look-up", 0.0, 0.95, (0.0, 0.0, 0.0), False),
    ("step-right", -0.25, 0.0, (1.0, 0.0, -0.3), False),
    ("roam", -0.35, 0.05, (2.2, 0.0, -0.5), True),
)


def _forward(yaw: float, pitch: float) -> np.ndarray:
    return np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), -math.cos(yaw) * math.cos(pitch)])


def camera_for(manifest: dict, yaw: float, pitch: float, offset, free: bool):
    """Eye/forward exactly as viewer/controls.js computes them in look mode."""
    half = np.array(manifest["navigation"]["half_extents"])
    position = np.asarray(offset, float) * half
    forward = _forward(yaw, pitch)
    eye = position + (forward - np.array([0, 0, -1.0])) * half[0] * 0.6
    if not free:
        eye = np.clip(eye, -half, half)
    return eye, forward


def fov_for(manifest: dict, aspect: float) -> float:
    """Mirror of fovFor() in viewer/app.js."""
    camera = manifest["camera"]
    vfov = math.radians(camera["vfov_deg"])
    if camera.get("projection") == "cylindrical":
        return min(max(vfov * 1.25, math.radians(50)), math.radians(80))
    hfov = math.radians(camera["hfov_deg"])
    padded_height = manifest["source"].get("padded", [0, camera["height"]])[1]
    padded_vfov = 2 * math.atan(padded_height / 2 / camera["fy"])
    fit_width = 2 * math.atan(math.tan(hfov / 2) / aspect)
    return min(max(vfov, fit_width), padded_vfov * 0.96, math.radians(95))


def _view_basis(forward: np.ndarray):
    z = -forward / np.linalg.norm(forward)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x) or 1.0
    y = np.cross(z, x)
    return x, y, z


class Scene:
    def __init__(self, glb_bytes: bytes) -> None:
        glb = read_glb(glb_bytes)
        self.manifest = glb.extras["viewfinder_world"]
        images = [
            np.asarray(Image.open(io.BytesIO(glb.image_bytes(i))).convert("RGB"), np.float32) / 255.0
            for i in range(len(glb.document.get("images", [])))
        ]
        self.meshes = []
        for mesh in glb.document["meshes"]:
            primitive = mesh["primitives"][0]
            attributes = primitive["attributes"]
            material = glb.document["materials"][primitive["material"]]
            pbr = material.get("pbrMetallicRoughness", {})
            texture = None
            if "baseColorTexture" in pbr:
                texture = images[glb.document["textures"][pbr["baseColorTexture"]["index"]]["source"]]
            mask = None
            mask_ref = material.get("extras", {}).get("viewfinder_mask")
            if mask_ref is not None:
                mask = images[glb.document["textures"][mask_ref]["source"]][..., 0]
            self.meshes.append({
                "name": mesh["name"],
                "positions": glb.accessor(attributes["POSITION"]).astype(np.float64),
                "uvs": glb.accessor(attributes["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in attributes else None,
                "colors": glb.accessor(attributes["COLOR_0"]).astype(np.float64)[:, :3] / 255.0 if "COLOR_0" in attributes else None,
                "indices": glb.accessor(primitive["indices"]).reshape(-1, 3).astype(np.int64),
                "texture": texture,
                "mask": mask,
                "fog": self.manifest["layers"].get(mesh["name"], {}).get("fog", True),
            })

    def render(self, eye, forward, width=480, height=270) -> np.ndarray:
        fovy = fov_for(self.manifest, width / height)
        focal = (height / 2) / math.tan(fovy / 2)
        x_axis, y_axis, z_axis = _view_basis(forward)
        env = self.manifest["environment"]
        fog_color = np.array(env["fog"]["color"])
        fog_density = env["fog"]["density"]

        image = np.zeros((height, width, 3))
        depth = np.full((height, width), np.inf)
        order = sorted(self.meshes, key=lambda m: (m["name"] not in BACKDROP, m["name"] != "dome"))
        for mesh in order:
            if mesh["name"] == "dome" and mesh["texture"] is not None:
                # The dome is a far sphere: look the environment up per pixel
                # instead of splatting its huge triangles (exact, and most of
                # the render time otherwise).
                image[:] = self._environment(mesh["texture"], x_axis, y_axis, z_axis, focal, width, height)
                continue
            samples = self._samples(mesh, eye, x_axis, y_axis, z_axis, focal, width, height)
            if samples is None:
                continue
            px, py, dist, view_z, rgb = samples
            if mesh["fog"]:
                fog = 1.0 - np.exp(-((fog_density * dist) ** 2))
                rgb = rgb * (1 - fog[:, None]) + fog_color * fog[:, None]
            flat = py * width + px
            if mesh["name"] in BACKDROP:
                # Painted without depth: later samples simply overwrite.
                image.reshape(-1, 3)[flat] = rgb
                continue
            keep = view_z < depth.reshape(-1)[flat]
            flat, view_z, rgb = flat[keep], view_z[keep], rgb[keep]
            sort = np.lexsort((view_z, flat))
            flat, view_z, rgb = flat[sort], view_z[sort], rgb[sort]
            first = np.ones(len(flat), bool)
            first[1:] = flat[1:] != flat[:-1]
            depth.reshape(-1)[flat[first]] = view_z[first]
            image.reshape(-1, 3)[flat[first]] = rgb[first]
        return np.clip(image * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def _environment(texture, x_axis, y_axis, z_axis, focal, width, height):
        px, py = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        directions = (
            (px - width / 2)[..., None] * x_axis
            - (py - height / 2)[..., None] * y_axis
            - focal * z_axis
        )
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        azimuth = np.arctan2(directions[..., 0], -directions[..., 2])
        elevation = np.arcsin(np.clip(directions[..., 1], -1, 1))
        u = (azimuth + np.pi) / (2 * np.pi)
        v = (np.pi / 2 - elevation) / np.pi
        tx = np.clip((u * texture.shape[1]).astype(int), 0, texture.shape[1] - 1)
        ty = np.clip((v * texture.shape[0]).astype(int), 0, texture.shape[0] - 1)
        return texture[ty, tx]

    @staticmethod
    def _samples(mesh, eye, x_axis, y_axis, z_axis, focal, width, height):
        positions, tris = mesh["positions"], mesh["indices"]
        relative = positions - eye
        cam = np.stack([relative @ x_axis, relative @ y_axis, relative @ z_axis], axis=1)
        in_front = cam[:, 2] < -1e-3
        safe_z = np.where(in_front, -cam[:, 2], 1e-3)
        sx = width / 2 + focal * cam[:, 0] / safe_z
        sy = height / 2 - focal * cam[:, 1] / safe_z
        # Screen-space extent of each triangle decides how densely to sample it.
        tri_front = in_front[tris].any(axis=1)
        tris = tris[tri_front]
        if len(tris) == 0:
            return None
        txs, tys = sx[tris], sy[tris]
        visible = (txs.max(1) >= 0) & (txs.min(1) < width) & (tys.max(1) >= 0) & (tys.min(1) < height)
        crosses_near = ~in_front[tris].all(axis=1)
        tris = tris[visible | crosses_near]
        if len(tris) == 0:
            return None
        txs, tys = sx[tris], sy[tris]
        extent = np.maximum(txs.max(1) - txs.min(1), tys.max(1) - tys.min(1))
        extent = np.where(~in_front[tris].all(axis=1), 2 * max(width, height), extent)
        levels = np.clip(np.ceil(np.minimum(extent, 3 * max(width, height)) * 1.6), 1, 900).astype(int)

        out = []
        for level in np.unique(levels):
            group = tris[levels == level]
            # Barycentric lattice with `level` subdivisions per edge.
            i, j = np.meshgrid(np.arange(level + 1), np.arange(level + 1), indexing="ij")
            valid = i + j <= level
            u = (i[valid] + 0.5 * (level == 0)) / max(level, 1)
            v = j[valid] / max(level, 1)
            w = 1 - u - v
            bary = np.stack([w, u, v], axis=1)  # (S, 3)
            for chunk in np.array_split(group, max(1, len(group) * len(bary) // 2_000_000 + 1)):
                p = np.einsum("sk,tkd->tsd", bary, positions[chunk]).reshape(-1, 3)
                rel = p - eye
                cz = rel @ z_axis
                front = cz < -1e-3
                if not front.any():
                    continue
                rel, cz = rel[front], cz[front]
                px = (width / 2 + focal * (rel @ x_axis) / -cz).astype(np.int64)
                py = (height / 2 - focal * (rel @ y_axis) / -cz).astype(np.int64)
                on = (px >= 0) & (px < width) & (py >= 0) & (py < height)
                if not on.any():
                    continue
                if mesh["texture"] is not None:
                    uv = np.einsum("sk,tkd->tsd", bary, mesh["uvs"][chunk]).reshape(-1, 2)[front][on]
                    tex = mesh["texture"]
                    tx = np.clip((uv[:, 0] * tex.shape[1]).astype(int), 0, tex.shape[1] - 1)
                    ty = np.clip((uv[:, 1] * tex.shape[0]).astype(int), 0, tex.shape[0] - 1)
                    rgb = tex[ty, tx]
                    if mesh["mask"] is not None:
                        m = mesh["mask"]
                        mx = np.clip((uv[:, 0] * m.shape[1]).astype(int), 0, m.shape[1] - 1)
                        my = np.clip((uv[:, 1] * m.shape[0]).astype(int), 0, m.shape[0] - 1)
                        keep = m[my, mx] >= 0.5
                    else:
                        keep = np.ones(len(rgb), bool)
                else:
                    rgb = np.einsum("sk,tkd->tsd", bary, mesh["colors"][chunk]).reshape(-1, 3)[front][on]
                    keep = np.ones(len(rgb), bool)
                dist = np.linalg.norm(rel[on], axis=1)
                out.append((px[on][keep], py[on][keep], dist[keep], -cz[on][keep], rgb[keep]))
        if not out:
            return None
        return tuple(np.concatenate([o[k] for o in out]) for k in range(5))


def contact_sheet(glb_bytes: bytes, title: str, width=480, height=270) -> Image.Image:
    scene = Scene(glb_bytes)
    tiles = []
    for label, yaw, pitch, offset, free in POSES:
        eye, forward = camera_for(scene.manifest, yaw, pitch, offset, free)
        tile = Image.fromarray(scene.render(eye, forward, width, height))
        tiles.append((label, tile))
    columns = 3
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * width, rows * (height + 18) + 20), (20, 20, 24))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    draw.text((6, 4), title, fill=(230, 230, 230))
    for index, (label, tile) in enumerate(tiles):
        x, y = (index % columns) * width, 20 + (index // columns) * (height + 18)
        sheet.paste(tile, (x, y + 16))
        draw.text((x + 6, y + 2), label, fill=(200, 200, 200))
    return sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos", nargs="+", type=Path)
    parser.add_argument("--quality", default="desktop")
    parser.add_argument("--out", type=Path, default=REPO / "local-data" / "world-eval")
    parser.add_argument("--tag", default="", help="suffix for output names, e.g. before/after")
    parser.add_argument("--depth-model", default="depth-anything-v2-small")
    args = parser.parse_args(argv)

    from viewfinder.world import build_world
    from viewfinder.world.depth import load_estimator

    estimator = load_estimator(args.depth_model)
    args.out.mkdir(parents=True, exist_ok=True)
    for photo in args.photos:
        result = build_world(photo, profile=args.quality, estimator=estimator)
        name = f"{photo.stem[:40]}_{args.quality}{('_' + args.tag) if args.tag else ''}.png"
        contact_sheet(result.glb, f"{photo.name}  ({args.quality}, {len(result.glb) / 1e6:.1f} MB)").save(args.out / name)
        print(args.out / name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
