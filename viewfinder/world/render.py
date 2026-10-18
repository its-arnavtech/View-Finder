"""Offscreen GPU rendering of a world: what it looks like from anywhere.

Exploration (explore.py) needs to see the world-in-progress from a new
standing point: which pixels already show something, how far away it is,
and which pixels show nothing (the holes a move reveals). Previews need the
same. OpenGL through moderngl does it in milliseconds; the NumPy splatter
in benchmark-harness/world_preview.py took minutes per sheet.

Each mesh is drawn with its texture and optional keep-mask exactly as the
viewer draws it. Three outputs per pixel:

* colour (RGB)
* depth along the camera axis, metres (0 where there is no surface)
* coverage: surface, sky, or hole

"Sky" is the dome in a direction the panorama says is open sky (infinitely
far, so it looks the same from anywhere). The dome in a direction that should
hold land is a *hole*: nothing has been built there for this viewpoint.

A mesh may belong to a *station*, the standing point it was generated
around (the photo's layers and panorama belong to the start point): only the
nearest station's meshes are drawn (``station_fade``), so wherever you stand
you see one coherent world made for that spot, never a mosaic of surfaces
made for different places fighting over depth.

On a laptop with two GPUs (NVIDIA Optimus) the renderer asks for the one
named in ``VIEWFINDER_RENDER_GPU``: ``discrete`` (the default, the NVIDIA
card that also runs the generator) or ``integrated`` (leaves the NVIDIA
card's memory to the generator, but loads the integrated GPU, which the
desktop also draws on). Anything else leaves the choice to the system.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SURFACE, BACKDROP, DOME = 0, 1, 2
HOLE, SKY, SOLID = 0, 1, 2

RENDER_GPU_ENV = "VIEWFINDER_RENDER_GPU"
# NVIDIA Optimus reads this per process when OpenGL loads.
_OPTIMUS = {"discrete": "0x800000001", "integrated": "0x800000000"}


def _prefer_gpu() -> None:
    """Point this process's OpenGL at the GPU ``VIEWFINDER_RENDER_GPU`` names
    (default discrete). Only before the first context: the driver keeps the
    GPU it loaded on. A ``SHIM_MCCOMPAT`` set by the user wins."""
    choice = os.environ.get(RENDER_GPU_ENV, "discrete").strip().lower()
    if sys.platform == "win32" and choice in _OPTIMUS:
        os.environ.setdefault("SHIM_MCCOMPAT", _OPTIMUS[choice])

_VERTEX = """
#version 330
uniform mat4 u_view;
uniform mat4 u_projection;
in vec3 in_position;
in vec2 in_uv;
out vec2 v_uv;
out float v_depth;
void main() {
    vec4 eye = u_view * vec4(in_position, 1.0);
    v_uv = in_uv;
    v_depth = -eye.z;
    gl_Position = u_projection * eye;
}
"""

_FRAGMENT = """
#version 330
uniform sampler2D u_texture;
uniform sampler2D u_mask;
uniform bool u_masked;
uniform int u_kind;          // 0 surface, 1 backdrop (sky), 2 dome
uniform bool u_overlay;      // generated at a station other than the start
uniform float u_fade;        // station meshes: share of fragments drawn (1 = all)
uniform sampler2D u_land;    // dome only: 1 where the panorama shows land
in vec2 v_uv;
in float v_depth;
layout(location = 0) out vec4 out_color;
// (depth metres, coverage code, texels per screen pixel, 1 if an overlay)
layout(location = 1) out vec4 out_depth;
void main() {
    if (u_masked && texture(u_mask, v_uv).r < 0.5) discard;
    // Screen-door fade, no sorting needed: the same rule as the viewer.
    float noise = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    if (noise >= u_fade) discard;
    out_color = vec4(texture(u_texture, v_uv).rgb, 1.0);
    // How many texels a pixel spans along the screen direction in which the
    // texture is most stretched: the smaller singular value of the
    // screen-to-texel Jacobian. Well below 1 means a few texels are smeared
    // across the screen. Per texture axis (fwidth), a sheet stretched
    // diagonally, as foliage at different depths is from a few metres
    // away, still read as sharp.
    vec2 size = vec2(textureSize(u_texture, 0));
    vec2 along_x = dFdx(v_uv * size);
    vec2 along_y = dFdy(v_uv * size);
    float a = dot(along_x, along_x);
    float b = dot(along_x, along_y);
    float c = dot(along_y, along_y);
    float half_trace = 0.5 * (a + c);
    float spread = sqrt(max(half_trace * half_trace - (a * c - b * b), 0.0));
    float density = sqrt(max(half_trace - spread, 0.0));
    if (u_kind == 0) {
        out_depth = vec4(v_depth, 2.0, density, u_overlay ? 1.0 : 0.0);
    } else if (u_kind == 1) {
        out_depth = vec4(0.0, 1.0, density, 0.0);
    } else {
        float land = texture(u_land, v_uv).r;
        out_depth = vec4(0.0, land >= 0.5 ? 0.0 : 1.0, density, 0.0);
    }
}
"""


#: Station meshes are drawn from the nearest station, cross-fading over this
#: band (metres) at the boundary with the next. (The viewer dissolves over
#: time instead, as you cross; at the stations themselves the two agree.)
STATION_BLEND = 0.5


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def station_fade(eye: np.ndarray, station: np.ndarray, stations: list | None = None) -> float:
    """Share of a station mesh's fragments drawn from ``eye``: 1 when
    ``station`` is the nearest of
    ``stations`` (on the ground plane), 0 when another is, with a short
    dithered cross-fade between."""
    distance = float(np.hypot(eye[0] - station[0], eye[2] - station[2]))
    others = [
        float(np.hypot(eye[0] - s[0], eye[2] - s[2]))
        for s in (stations or [])
        if abs(s[0] - station[0]) + abs(s[2] - station[2]) > 1e-6
    ]
    nearest_other = min(others) if others else float("inf")
    return _smoothstep(-STATION_BLEND, STATION_BLEND, nearest_other - distance)


class RendererUnavailable(RuntimeError):
    """No OpenGL 3.3 context could be created (e.g. a headless server)."""


@dataclass
class Camera:
    """A perspective camera in world coordinates (glTF: +Y up)."""

    eye: np.ndarray
    forward: np.ndarray
    fov_deg: float
    width: int
    height: int

    @staticmethod
    def looking(eye, yaw_deg: float, pitch_deg: float, fov_deg: float, width: int, height: int) -> "Camera":
        yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
        forward = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), -math.cos(yaw) * math.cos(pitch)])
        return Camera(np.asarray(eye, np.float64), forward, fov_deg, width, height)

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(right, up, back) unit vectors; the camera looks along -back."""
        back = -self.forward / np.linalg.norm(self.forward)
        right = np.cross([0.0, 1.0, 0.0], back)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        return right, up, back

    def view_matrix(self) -> np.ndarray:
        right, up, back = self.basis()
        rotation = np.stack([right, up, back])
        view = np.eye(4)
        view[:3, :3] = rotation
        view[:3, 3] = -rotation @ self.eye
        return view

    def projection_matrix(self, near: float = 0.05, far: float = 2000.0) -> np.ndarray:
        """``fov_deg`` is the vertical field of view."""
        f = 1.0 / math.tan(math.radians(self.fov_deg) / 2)
        aspect = self.width / self.height
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0],
        ])

    @property
    def focal(self) -> float:
        """Focal length in pixels."""
        return (self.height / 2) / math.tan(math.radians(self.fov_deg) / 2)

    def rays(self) -> np.ndarray:
        """(H, W, 3) world directions with unit depth along the camera axis."""
        right, up, back = self.basis()
        f = self.focal
        u = (np.arange(self.width, dtype=np.float64) + 0.5 - self.width / 2) / f
        v = (np.arange(self.height, dtype=np.float64) + 0.5 - self.height / 2) / f
        x, y = np.meshgrid(u, -v)
        return (x[..., None] * right + y[..., None] * up - back).astype(np.float32)


@dataclass
class Frame:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32 metres along the camera axis, 0 = none
    coverage: np.ndarray  # (H, W) uint8: HOLE, SKY or SOLID
    density: np.ndarray | None = None  # (H, W) texels per screen pixel
    generated: np.ndarray | None = None  # (H, W) bool: drawn by a refinement

    @property
    def holes(self) -> np.ndarray:
        return self.coverage == HOLE


class WorldRenderer:
    """Holds a world's meshes on the GPU and renders them from any camera."""

    def __init__(self) -> None:
        try:
            import moderngl
        except ImportError as exc:
            raise RendererUnavailable("moderngl is not installed") from exc
        _prefer_gpu()
        try:
            self._ctx = moderngl.create_standalone_context(require=330)
        except Exception as exc:  # no GL driver / headless
            raise RendererUnavailable(f"no OpenGL 3.3 context: {exc}") from exc
        self.gpu = str(self._ctx.info.get("GL_RENDERER", "unknown"))
        logger.info("rendering on %s", self.gpu)
        self._gl = moderngl
        self._program = self._ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        self._meshes: list[dict] = []
        self._textures: dict[int, object] = {}
        self._white = self._texture(np.full((1, 1, 3), 255, np.uint8))
        self._lock = threading.Lock()

    def _texture(self, image: np.ndarray):
        image = np.ascontiguousarray(image)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        texture = self._ctx.texture((image.shape[1], image.shape[0]), 3, np.flipud(image).tobytes())
        texture.build_mipmaps()
        texture.filter = (self._gl.LINEAR_MIPMAP_LINEAR, self._gl.LINEAR)
        texture.anisotropy = 8.0  # as the viewer: sharp where the texture is squeezed
        texture.repeat_x = False
        texture.repeat_y = False
        return texture

    def add(
        self,
        positions: np.ndarray,
        uvs: np.ndarray,
        indices: np.ndarray,
        texture: np.ndarray,
        mask: np.ndarray | None = None,
        kind: int = SURFACE,
        land: np.ndarray | None = None,
        station: np.ndarray | None = None,
    ) -> int:
        """Upload a textured mesh. ``texture`` and ``mask`` are images whose
        UV origin is the top-left (glTF convention). A mesh with a
        ``station`` is drawn only while that is the nearest station; one
        without (the dome, the sky) always."""
        if len(indices) == 0:
            return -1
        vertices = np.concatenate(
            [positions.astype("f4"), np.stack([uvs[:, 0], 1.0 - uvs[:, 1]], axis=1).astype("f4")], axis=1
        )
        vbo = self._ctx.buffer(np.ascontiguousarray(vertices).tobytes())
        ibo = self._ctx.buffer(np.ascontiguousarray(indices.astype("u4")).tobytes())
        vao = self._ctx.vertex_array(self._program, [(vbo, "3f 2f", "in_position", "in_uv")], ibo)
        self._meshes.append({
            "vao": vao,
            "buffers": (vbo, ibo),
            "texture": self._texture(texture),
            "mask": self._texture(mask) if mask is not None else None,
            "land": self._texture(land) if land is not None else None,
            "kind": kind,
            "station": None if station is None else np.asarray(station, np.float64),
        })
        return len(self._meshes) - 1

    def stations(self) -> list[np.ndarray]:
        """The start point and every station with meshes."""
        found = [np.zeros(3)]
        for mesh in self._meshes:
            station = mesh["station"]
            if station is not None and not any(np.allclose(station, s) for s in found):
                found.append(station)
        return found

    def render(self, camera: Camera, show: list | None = None) -> Frame:
        """Render from ``camera``. Station meshes show as the viewer shows
        them (the nearest station's), or exactly those of the stations in
        ``show``."""
        ctx = self._ctx
        width, height = camera.width, camera.height
        with self._lock:
            color = ctx.texture((width, height), 4)
            depth_target = ctx.texture((width, height), 4, dtype="f4")
            depth_buffer = ctx.depth_renderbuffer((width, height))
            fbo = ctx.framebuffer(color_attachments=[color, depth_target], depth_attachment=depth_buffer)
            fbo.use()
            ctx.viewport = (0, 0, width, height)
            fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
            ctx.enable(self._gl.DEPTH_TEST)
            ctx.disable(self._gl.CULL_FACE)
            program = self._program
            program["u_view"].write(camera.view_matrix().T.astype("f4").tobytes())
            program["u_projection"].write(camera.projection_matrix().T.astype("f4").tobytes())
            stations = self.stations()
            for mesh in self._meshes:
                station = mesh["station"]
                if station is None:
                    fade = 1.0
                elif show is not None:
                    fade = 1.0 if any(np.allclose(station, s) for s in show) else 0.0
                else:
                    fade = station_fade(camera.eye, station, stations)
                if fade <= 0.0:
                    continue
                program["u_fade"].value = fade
                mesh["texture"].use(location=0)
                program["u_texture"].value = 0
                (mesh["mask"] or self._white).use(location=1)
                program["u_mask"].value = 1
                program["u_masked"].value = mesh["mask"] is not None
                (mesh["land"] or self._white).use(location=2)
                program["u_land"].value = 2
                program["u_kind"].value = mesh["kind"]
                program["u_overlay"].value = station is not None and bool(np.any(station))
                mesh["vao"].render()
            rgba = np.frombuffer(color.read(), np.uint8).reshape(height, width, 4)
            data = np.frombuffer(depth_target.read(), np.float32).reshape(height, width, 4)
            for resource in (fbo, color, depth_target, depth_buffer):
                resource.release()
        rgba = np.flipud(rgba)
        data = np.flipud(data)
        drawn = rgba[..., 3] > 0
        coverage = np.where(drawn, np.rint(data[..., 1]).astype(np.uint8), HOLE)
        depth = np.where(coverage == SOLID, data[..., 0], 0.0).astype(np.float32)
        density = np.ascontiguousarray(data[..., 2]).astype(np.float32)
        generated = (data[..., 3] > 0.5) & drawn
        return Frame(np.ascontiguousarray(rgba[..., :3]), depth, coverage, density, generated)

    def release(self) -> None:
        for mesh in self._meshes:
            mesh["vao"].release()
            for buffer in mesh["buffers"]:
                buffer.release()
            for key in ("texture", "mask", "land"):
                if mesh[key] is not None:
                    mesh[key].release()
        self._meshes.clear()
        self._ctx.release()
