import struct

import numpy as np
import pytest

from viewfinder.world.gltf import GlbBuilder, GlbError, read_glb


def _quad(builder: GlbBuilder, *, vertices: int = 4, texture=None) -> None:
    positions = np.random.default_rng(0).normal(size=(vertices, 3)).astype(np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    uvs = np.zeros((vertices, 2), np.float32) if texture is not None else None
    builder.add_mesh("quad", positions, indices, uvs=uvs, texture=texture)


def test_round_trip_preserves_accessors_and_extras():
    builder = GlbBuilder()
    positions = np.arange(12, dtype=np.float32).reshape(4, 3)
    colors = np.array([[255, 0, 0, 255]] * 4, dtype=np.uint8)
    indices = np.array([0, 1, 2, 2, 1, 3], dtype=np.uint32)
    builder.add_mesh("tri", positions, indices, colors=colors)
    glb = read_glb(builder.build(extras={"viewfinder_world": {"answer": 42}}))

    primitive = glb.mesh("tri")["primitives"][0]
    np.testing.assert_array_equal(glb.accessor(primitive["attributes"]["POSITION"]), positions)
    np.testing.assert_array_equal(glb.accessor(primitive["attributes"]["COLOR_0"]), colors)
    np.testing.assert_array_equal(glb.accessor(primitive["indices"]), indices)
    assert glb.accessor(primitive["indices"]).dtype == np.uint16  # small mesh -> 16-bit
    assert glb.extras == {"viewfinder_world": {"answer": 42}}
    position_accessor = glb.document["accessors"][primitive["attributes"]["POSITION"]]
    assert position_accessor["min"] == [0.0, 1.0, 2.0] and position_accessor["max"] == [9.0, 10.0, 11.0]


def test_header_and_chunks_follow_glb_spec():
    builder = GlbBuilder()
    _quad(builder)
    data = builder.build()
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert (magic, version, length) == (0x46546C67, 2, len(data))
    json_length, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == 0x4E4F534A and json_length % 4 == 0
    bin_length, bin_type = struct.unpack_from("<II", data, 20 + json_length)
    assert bin_type == 0x004E4942 and bin_length % 4 == 0
    assert len(data) % 4 == 0


def test_large_meshes_switch_to_32_bit_indices():
    builder = GlbBuilder()
    count = 70_000
    positions = np.zeros((count, 3), np.float32)
    indices = np.array([0, 1, count - 1], dtype=np.uint32)
    builder.add_mesh("big", positions, indices)
    glb = read_glb(builder.build())
    assert glb.accessor(glb.mesh("big")["primitives"][0]["indices"]).dtype == np.uint32


def test_embedded_texture_and_unlit_material():
    builder = GlbBuilder()
    texture = builder.add_texture(b"\xff\xd8\xff fake jpeg")
    _quad(builder, texture=texture)
    glb = read_glb(builder.build())
    assert glb.image_bytes(0) == b"\xff\xd8\xff fake jpeg"
    material = glb.document["materials"][0]
    assert "KHR_materials_unlit" in material["extensions"]
    assert material["pbrMetallicRoughness"]["baseColorTexture"]["index"] == texture
    assert glb.document["samplers"][0]["wrapS"] == 33071  # clamp: NPOT-safe on WebGL1


def test_buffer_views_are_four_byte_aligned():
    builder = GlbBuilder()
    builder.add_texture(b"abc")  # odd length forces padding before the next view
    _quad(builder)
    glb = read_glb(builder.build())
    assert all(view["byteOffset"] % 4 == 0 for view in glb.document["bufferViews"])


@pytest.mark.parametrize(
    "indices, message",
    [(np.array([0, 1], np.uint32), "multiple of 3"), (np.array([0, 1, 9], np.uint32), "outside")],
)
def test_writer_rejects_bad_indices(indices, message):
    builder = GlbBuilder()
    with pytest.raises(GlbError, match=message):
        builder.add_mesh("bad", np.zeros((4, 3), np.float32), indices)


def test_texture_requires_uvs():
    builder = GlbBuilder()
    texture = builder.add_texture(b"x")
    with pytest.raises(GlbError, match="no UVs"):
        builder.add_mesh("m", np.zeros((3, 3)), np.array([0, 1, 2]), texture=texture)


def test_reader_rejects_truncated_and_foreign_files():
    builder = GlbBuilder()
    _quad(builder)
    data = builder.build()
    with pytest.raises(GlbError):
        read_glb(data[:-4])
    with pytest.raises(GlbError):
        read_glb(b"PK\x03\x04" + data[4:])
