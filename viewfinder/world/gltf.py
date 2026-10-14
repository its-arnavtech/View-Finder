"""A minimal, dependency-free GLB 2.0 writer and reader.

Only the subset the World Engine emits is supported: triangle meshes with
POSITION, optional TEXCOORD_0 or COLOR_0, 16/32-bit indices, unlit materials
(``KHR_materials_unlit``) and embedded JPEG/PNG textures. The reader is the
writer's mirror and validates every structural rule the browser viewer relies
on, so a malformed file fails in Python tests rather than as a black canvas on
a phone.

Spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

GLB_MAGIC = 0x46546C67  # b"glTF"
CHUNK_JSON = 0x4E4F534A  # b"JSON"
CHUNK_BIN = 0x004E4942  # b"BIN\0"

ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963

FLOAT = 5126
UNSIGNED_BYTE = 5121
UNSIGNED_SHORT = 5123
UNSIGNED_INT = 5125

_COMPONENT_DTYPES = {
    FLOAT: np.float32,
    UNSIGNED_BYTE: np.uint8,
    UNSIGNED_SHORT: np.uint16,
    UNSIGNED_INT: np.uint32,
}
_TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

LINEAR = 9729
LINEAR_MIPMAP_LINEAR = 9987
CLAMP_TO_EDGE = 33071

UNLIT = "KHR_materials_unlit"


class GlbError(ValueError):
    """A GLB file violates the structure this module reads and writes."""


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


@dataclass
class _Mesh:
    name: str
    primitive: dict[str, Any]


@dataclass
class GlbBuilder:
    """Accumulates meshes, textures and buffers, then serialises one GLB."""

    generator: str = "View Finder World Engine"
    _bin: bytearray = field(default_factory=bytearray)
    _buffer_views: list[dict[str, Any]] = field(default_factory=list)
    _accessors: list[dict[str, Any]] = field(default_factory=list)
    _images: list[dict[str, Any]] = field(default_factory=list)
    _textures: list[dict[str, Any]] = field(default_factory=list)
    _materials: list[dict[str, Any]] = field(default_factory=list)
    _meshes: list[_Mesh] = field(default_factory=list)

    def _view(self, data: bytes, target: int | None) -> int:
        # Every view starts 4-byte aligned, which satisfies all component sizes.
        while len(self._bin) % 4:
            self._bin.append(0)
        view: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": len(self._bin),
            "byteLength": len(data),
        }
        if target is not None:
            view["target"] = target
        self._bin.extend(data)
        self._buffer_views.append(view)
        return len(self._buffer_views) - 1

    def _accessor(
        self,
        array: np.ndarray,
        component: int,
        kind: str,
        target: int,
        *,
        normalized: bool = False,
        with_bounds: bool = False,
    ) -> int:
        dtype = _COMPONENT_DTYPES[component]
        data = np.ascontiguousarray(array, dtype=dtype)
        width = _TYPE_WIDTH[kind]
        count = data.size // width
        if count == 0 or data.size % width:
            raise GlbError(f"accessor data of shape {array.shape} does not fit {kind}")
        accessor: dict[str, Any] = {
            "bufferView": self._view(data.tobytes(), target),
            "componentType": component,
            "count": int(count),
            "type": kind,
        }
        if normalized:
            accessor["normalized"] = True
        if with_bounds:
            reshaped = data.reshape(count, width)
            accessor["min"] = [float(v) for v in reshaped.min(axis=0)]
            accessor["max"] = [float(v) for v in reshaped.max(axis=0)]
        self._accessors.append(accessor)
        return len(self._accessors) - 1

    def add_texture(self, encoded: bytes, mime_type: str = "image/jpeg") -> int:
        view = self._view(encoded, None)
        self._images.append({"bufferView": view, "mimeType": mime_type})
        self._textures.append({"source": len(self._images) - 1, "sampler": 0})
        return len(self._textures) - 1

    def _material(self, name: str, texture: int | None) -> int:
        pbr: dict[str, Any] = {"metallicFactor": 0.0, "roughnessFactor": 1.0}
        if texture is not None:
            pbr["baseColorTexture"] = {"index": texture}
        self._materials.append(
            {
                "name": name,
                "pbrMetallicRoughness": pbr,
                "doubleSided": True,
                "extensions": {UNLIT: {}},
            }
        )
        return len(self._materials) - 1

    def add_mesh(
        self,
        name: str,
        positions: np.ndarray,
        indices: np.ndarray,
        *,
        uvs: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        texture: int | None = None,
    ) -> None:
        """Add one unlit triangle mesh.

        ``colors`` are uint8 RGBA per vertex. ``texture`` is an index returned by
        :meth:`add_texture` and requires ``uvs``.
        """
        positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        indices = np.asarray(indices).reshape(-1)
        if indices.size == 0 or indices.size % 3:
            raise GlbError(f"mesh {name!r} needs a non-empty multiple of 3 indices")
        vertex_count = len(positions)
        if int(indices.max()) >= vertex_count or int(indices.min()) < 0:
            raise GlbError(f"mesh {name!r} has indices outside its {vertex_count} vertices")
        if texture is not None and uvs is None:
            raise GlbError(f"mesh {name!r} has a texture but no UVs")

        attributes = {
            "POSITION": self._accessor(positions, FLOAT, "VEC3", ARRAY_BUFFER, with_bounds=True)
        }
        if uvs is not None:
            uvs = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
            if len(uvs) != vertex_count:
                raise GlbError(f"mesh {name!r} has {len(uvs)} UVs for {vertex_count} vertices")
            attributes["TEXCOORD_0"] = self._accessor(uvs, FLOAT, "VEC2", ARRAY_BUFFER)
        if colors is not None:
            colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 4)
            if len(colors) != vertex_count:
                raise GlbError(f"mesh {name!r} has {len(colors)} colours for {vertex_count} vertices")
            attributes["COLOR_0"] = self._accessor(
                colors, UNSIGNED_BYTE, "VEC4", ARRAY_BUFFER, normalized=True
            )
        index_component = UNSIGNED_SHORT if vertex_count <= 0xFFFF else UNSIGNED_INT
        index_accessor = self._accessor(indices, index_component, "SCALAR", ELEMENT_ARRAY_BUFFER)
        primitive = {
            "attributes": attributes,
            "indices": index_accessor,
            "material": self._material(name, texture),
            "mode": 4,
        }
        self._meshes.append(_Mesh(name=name, primitive=primitive))

    def build(self, extras: dict[str, Any] | None = None) -> bytes:
        if not self._meshes:
            raise GlbError("a GLB needs at least one mesh")
        document: dict[str, Any] = {
            "asset": {"version": "2.0", "generator": self.generator},
            "extensionsUsed": [UNLIT],
            "scene": 0,
            "scenes": [{"name": "world", "nodes": list(range(len(self._meshes)))}],
            "nodes": [{"name": m.name, "mesh": i} for i, m in enumerate(self._meshes)],
            "meshes": [{"name": m.name, "primitives": [m.primitive]} for m in self._meshes],
            "materials": self._materials,
            "accessors": self._accessors,
            "bufferViews": self._buffer_views,
            "buffers": [{"byteLength": len(_pad4(bytes(self._bin)))}],
        }
        if self._textures:
            document["images"] = self._images
            document["textures"] = self._textures
            document["samplers"] = [
                {
                    "magFilter": LINEAR,
                    "minFilter": LINEAR_MIPMAP_LINEAR,
                    "wrapS": CLAMP_TO_EDGE,
                    "wrapT": CLAMP_TO_EDGE,
                }
            ]
        if extras:
            document["scenes"][0]["extras"] = extras
        json_chunk = _pad4(json.dumps(document, separators=(",", ":")).encode("utf-8"), b" ")
        bin_chunk = _pad4(bytes(self._bin))
        total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
        return b"".join(
            [
                struct.pack("<III", GLB_MAGIC, 2, total),
                struct.pack("<II", len(json_chunk), CHUNK_JSON),
                json_chunk,
                struct.pack("<II", len(bin_chunk), CHUNK_BIN),
                bin_chunk,
            ]
        )


@dataclass(frozen=True)
class Glb:
    document: dict[str, Any]
    binary: bytes

    def accessor(self, index: int) -> np.ndarray:
        accessor = self.document["accessors"][index]
        view = self.document["bufferViews"][accessor["bufferView"]]
        dtype = _COMPONENT_DTYPES[accessor["componentType"]]
        width = _TYPE_WIDTH[accessor["type"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        count = accessor["count"]
        array = np.frombuffer(self.binary, dtype=dtype, count=count * width, offset=start)
        return array.reshape(count, width) if width > 1 else array

    def image_bytes(self, index: int) -> bytes:
        view = self.document["bufferViews"][self.document["images"][index]["bufferView"]]
        start = view.get("byteOffset", 0)
        return self.binary[start : start + view["byteLength"]]

    def mesh(self, name: str) -> dict[str, Any] | None:
        for mesh in self.document.get("meshes", []):
            if mesh.get("name") == name:
                return mesh
        return None

    @property
    def extras(self) -> dict[str, Any]:
        return self.document["scenes"][self.document.get("scene", 0)].get("extras", {})


def read_glb(data: bytes) -> Glb:
    """Parse and validate a GLB produced by :class:`GlbBuilder`."""
    if len(data) < 20:
        raise GlbError("file is too short to be a GLB")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2:
        raise GlbError("not a glTF 2.0 binary")
    if length != len(data):
        raise GlbError(f"header length {length} does not match file size {len(data)}")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    if json_type != CHUNK_JSON or json_length % 4:
        raise GlbError("first chunk must be 4-byte aligned JSON")
    document = json.loads(data[20 : 20 + json_length].decode("utf-8"))
    offset = 20 + json_length
    binary = b""
    if offset < len(data):
        bin_length, bin_type = struct.unpack_from("<II", data, offset)
        if bin_type != CHUNK_BIN or bin_length % 4:
            raise GlbError("second chunk must be 4-byte aligned BIN")
        binary = data[offset + 8 : offset + 8 + bin_length]
    glb = Glb(document=document, binary=binary)

    for index, view in enumerate(document.get("bufferViews", [])):
        end = view.get("byteOffset", 0) + view["byteLength"]
        if end > len(binary) or view.get("byteOffset", 0) % 4:
            raise GlbError(f"bufferView {index} is misaligned or out of range")
    for mesh in document.get("meshes", []):
        for primitive in mesh["primitives"]:
            vertices = document["accessors"][primitive["attributes"]["POSITION"]]["count"]
            for accessor_index in primitive["attributes"].values():
                if document["accessors"][accessor_index]["count"] != vertices:
                    raise GlbError(f"mesh {mesh.get('name')!r} has mismatched attribute counts")
            indices = glb.accessor(primitive["indices"])
            if indices.size % 3 or (indices.size and int(indices.max()) >= vertices):
                raise GlbError(f"mesh {mesh.get('name')!r} has invalid indices")
    return glb
