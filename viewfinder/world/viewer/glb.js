/* GLB reader for the World Engine's own output (viewfinder/world/gltf.py).
 * Mirrors the Python reader: validates what the renderer relies on so a bad
 * file fails with a message instead of a black canvas. */
(function (root) {
  "use strict";

  var MAGIC = 0x46546c67, CHUNK_JSON = 0x4e4f534a, CHUNK_BIN = 0x004e4942;
  var COMPONENTS = { 5126: Float32Array, 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array };
  var WIDTH = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

  function fail(message) { throw new Error("Invalid world file: " + message); }

  function accessor(doc, bin, index) {
    var acc = doc.accessors[index];
    var view = doc.bufferViews[acc.bufferView];
    var Type = COMPONENTS[acc.componentType];
    if (!Type) fail("unsupported component type " + acc.componentType);
    var offset = (view.byteOffset || 0) + (acc.byteOffset || 0);
    var length = acc.count * WIDTH[acc.type];
    if (offset % Type.BYTES_PER_ELEMENT) fail("misaligned accessor " + index);
    if (offset + length * Type.BYTES_PER_ELEMENT > bin.byteLength) fail("accessor " + index + " out of range");
    return new Type(bin.buffer, bin.byteOffset + offset, length);
  }

  function parse(arrayBuffer) {
    if (arrayBuffer.byteLength < 20) fail("too short");
    var data = new DataView(arrayBuffer);
    if (data.getUint32(0, true) !== MAGIC || data.getUint32(4, true) !== 2) fail("not glTF 2.0 binary");
    var jsonLength = data.getUint32(12, true);
    if (data.getUint32(16, true) !== CHUNK_JSON) fail("missing JSON chunk");
    var text = new TextDecoder("utf-8").decode(new Uint8Array(arrayBuffer, 20, jsonLength));
    var doc = JSON.parse(text);
    var offset = 20 + jsonLength;
    var bin = new Uint8Array(0);
    if (offset + 8 <= arrayBuffer.byteLength) {
      var binLength = data.getUint32(offset, true);
      if (data.getUint32(offset + 4, true) !== CHUNK_BIN) fail("missing BIN chunk");
      bin = new Uint8Array(arrayBuffer, offset + 8, binLength);
    }

    var images = (doc.images || []).map(function (image) {
      var view = doc.bufferViews[image.bufferView];
      var bytes = new Uint8Array(bin.buffer, bin.byteOffset + (view.byteOffset || 0), view.byteLength);
      return new Blob([bytes], { type: image.mimeType || "image/jpeg" });
    });

    var meshes = (doc.meshes || []).map(function (mesh) {
      var primitive = mesh.primitives[0];
      var attributes = primitive.attributes;
      var positions = accessor(doc, bin, attributes.POSITION);
      var vertexCount = positions.length / 3;
      var material = doc.materials[primitive.material] || {};
      var pbr = material.pbrMetallicRoughness || {};
      var texture = pbr.baseColorTexture ? doc.textures[pbr.baseColorTexture.index].source : -1;
      // Keep-mask for pixel-accurate silhouettes (see gltf.py MASK_EXTRA).
      var maskRef = material.extras && material.extras.viewfinder_mask;
      var mask = maskRef !== undefined && maskRef !== null ? doc.textures[maskRef].source : -1;
      var indices = accessor(doc, bin, primitive.indices);
      for (var i = 0, max = 0; i < indices.length; i++) if (indices[i] > max) max = indices[i];
      if (indices.length % 3 || max >= vertexCount) fail("mesh " + mesh.name + " has invalid indices");
      return {
        name: mesh.name,
        positions: positions,
        uvs: attributes.TEXCOORD_0 !== undefined ? accessor(doc, bin, attributes.TEXCOORD_0) : null,
        colors: attributes.COLOR_0 !== undefined ? accessor(doc, bin, attributes.COLOR_0) : null,
        indices: indices,
        image: texture,
        mask: mask,
      };
    });

    var scene = (doc.scenes || [])[doc.scene || 0] || {};
    var manifest = (scene.extras || {}).viewfinder_world;
    if (!manifest) fail("no World Engine manifest");
    return { manifest: manifest, meshes: meshes, images: images };
  }

  root.WorldGlb = { parse: parse };
})(window);
