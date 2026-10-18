/* Dependency-free WebGL renderer for World Engine scenes.
 * One unlit program covers every layer: textured (photo, background, sky)
 * or vertex-coloured (dome, ground), each optionally fogged.
 * WebGL2 when available, else WebGL1 + OES_element_index_uint. */
(function (root) {
  "use strict";

  var VERTEX = [
    "attribute vec3 aPosition;",
    "attribute vec2 aUv;",
    "attribute vec4 aColor;",
    "uniform mat4 uViewProjection;",
    "uniform vec3 uEye;",
    "varying vec2 vUv;",
    "varying vec4 vColor;",
    "varying float vDistance;",
    "void main() {",
    "  vUv = aUv;",
    "  vColor = aColor;",
    "  vDistance = distance(aPosition, uEye);",
    "  gl_Position = uViewProjection * vec4(aPosition, 1.0);",
    "}",
  ].join("\n");

  var FRAGMENT = [
    "precision mediump float;",
    "uniform sampler2D uTexture;",
    "uniform sampler2D uMask;",
    "uniform float uTextured;",
    "uniform float uMasked;",
    "uniform float uFog;",
    "uniform float uFogDensity;",
    "uniform vec3 uFogColor;",
    "uniform float uFadeFrom;",
    "uniform float uFade;",
    "varying vec2 vUv;",
    "varying vec4 vColor;",
    "varying float vDistance;",
    "void main() {",
    // Silhouette cells discard the texels that belong to the far side, so
    // edges follow the photo's pixels rather than the mesh grid.
    "  if (uMasked > 0.5 && texture2D(uMask, vUv).r < 0.5) discard;",
    // Screen-door fade between stations (interleaved gradient noise):
    // partial transparency with no sorting.
    "  float noise = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));",
    "  if (noise < uFadeFrom || noise >= uFade) discard;",
    "  vec4 color = mix(vColor, texture2D(uTexture, vUv), uTextured);",
    "  float d = uFogDensity * vDistance;",
    "  float fog = uFog * (1.0 - exp(-d * d));",
    "  gl_FragColor = vec4(mix(color.rgb, uFogColor, fog), 1.0);",
    "}",
  ].join("\n");

  // ?webgl=1 forces the WebGL1 path that older phones fall back to, for
  // testing. Read at load, before the app rewrites the address bar.
  var FORCE_WEBGL1 = /[?&]webgl=1\b/.test(root.location.search);

  // Layers drawn behind everything else, without writing depth.
  var BACKDROP = { dome: 0, sky: 1 };
  // Each standing point the world was generated around (the start point,
  // and every station with a bubble in manifest.overlays) owns its meshes,
  // and only the current station's are drawn: one coherent surface wherever
  // you stand. Walking into another station's area (nearer to it by
  // STATION_HYSTERESIS m, so a boundary never flickers) dissolves into its
  // world over STATION_DISSOLVE ms. A dissolve tied to position instead
  // left both worlds screen-doored together wherever you stopped between.
  var STATION_HYSTERESIS = 0.25;
  var STATION_DISSOLVE = 450;
  function smoothstep(a, b, x) {
    var t = Math.min(Math.max((x - a) / (b - a), 0), 1);
    return t * t * (3 - 2 * t);
  }

  /* ---------------- tiny column-major mat4 helpers ---------------- */
  function perspective(fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  }
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function normalize(a) { var l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
  function lookAt(eye, target, up) {
    var z = normalize(sub(eye, target)), x = normalize(cross(up, z)), y = cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
  }
  function multiply(a, b) {
    var out = new Float32Array(16);
    for (var c = 0; c < 4; c++) for (var r = 0; r < 4; r++) {
      out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
    return out;
  }

  function Renderer(canvas, options) {
    this.canvas = canvas;
    this.maxPixelRatio = options.maxPixelRatio || 2;
    this.onLost = options.onLost || function () {};
    this.onRestored = options.onRestored || function () {};
    this.scene = null;
    this.gpu = null;
    this.lost = false;
    var self = this;
    canvas.addEventListener("webglcontextlost", function (event) {
      event.preventDefault(); // tells the browser we will restore
      self.lost = true;
      self.gpu = null;
      self.onLost();
    });
    canvas.addEventListener("webglcontextrestored", function () {
      self.lost = false;
      self._initContext();
      if (self.scene) self._upload(self.scene).then(function () { self.onRestored(); });
    });
    this._initContext();
  }

  Renderer.prototype._initContext = function () {
    var attributes = { antialias: true, alpha: false, powerPreference: "default", preserveDrawingBuffer: false };
    var gl = FORCE_WEBGL1 ? null : this.canvas.getContext("webgl2", attributes);
    this.webgl2 = !!gl;
    if (!gl) gl = this.canvas.getContext("webgl", attributes) || this.canvas.getContext("experimental-webgl", attributes);
    if (!gl) throw new Error("This browser cannot show 3D (WebGL is unavailable or disabled).");
    this.gl = gl;
    this.uint32 = this.webgl2 || !!gl.getExtension("OES_element_index_uint");
    // The panorama's texture is squeezed sideways toward the zenith; plain
    // mipmapping then picks a level by the squeezed axis and blurs the other
    // (treetops overhead came out soft). Anisotropic filtering samples along it.
    this.anisotropic = gl.getExtension("EXT_texture_filter_anisotropic") ||
      gl.getExtension("WEBKIT_EXT_texture_filter_anisotropic") || gl.getExtension("MOZ_EXT_texture_filter_anisotropic");
    this.program = this._program(VERTEX, FRAGMENT);
    var p = this.program;
    this.loc = {
      position: gl.getAttribLocation(p, "aPosition"),
      uv: gl.getAttribLocation(p, "aUv"),
      color: gl.getAttribLocation(p, "aColor"),
      viewProjection: gl.getUniformLocation(p, "uViewProjection"),
      eye: gl.getUniformLocation(p, "uEye"),
      texture: gl.getUniformLocation(p, "uTexture"),
      mask: gl.getUniformLocation(p, "uMask"),
      masked: gl.getUniformLocation(p, "uMasked"),
      textured: gl.getUniformLocation(p, "uTextured"),
      fog: gl.getUniformLocation(p, "uFog"),
      fogDensity: gl.getUniformLocation(p, "uFogDensity"),
      fogColor: gl.getUniformLocation(p, "uFogColor"),
      fadeFrom: gl.getUniformLocation(p, "uFadeFrom"),
      fade: gl.getUniformLocation(p, "uFade"),
    };
  };

  Renderer.prototype._program = function (vertexSource, fragmentSource) {
    var gl = this.gl;
    function compile(type, source) {
      var shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS) && !gl.isContextLost()) {
        throw new Error("Shader error: " + gl.getShaderInfoLog(shader));
      }
      return shader;
    }
    var program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS) && !gl.isContextLost()) {
      throw new Error("Shader link error: " + gl.getProgramInfoLog(program));
    }
    return program;
  };

  function decodeImage(blob) {
    return new Promise(function (resolve, reject) {
      var url = URL.createObjectURL(blob);
      var image = new Image();
      image.onload = function () { URL.revokeObjectURL(url); resolve(image); };
      image.onerror = function () { URL.revokeObjectURL(url); reject(new Error("Could not decode a world texture")); };
      image.src = url;
    });
  }

  Renderer.prototype._texture = function (image) {
    var gl = this.gl;
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); // glTF UV origin is the image's top-left
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    var pot = (image.width & (image.width - 1)) === 0 && (image.height & (image.height - 1)) === 0;
    if (this.webgl2 || pot) {
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      if (this.anisotropic) {
        var most = gl.getParameter(this.anisotropic.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
        gl.texParameterf(gl.TEXTURE_2D, this.anisotropic.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, most));
      }
    } else {
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); // WebGL1 cannot mip NPOT
    }
    return texture;
  };

  Renderer.prototype._buffer = function (target, data) {
    var gl = this.gl, buffer = gl.createBuffer();
    gl.bindBuffer(target, buffer);
    gl.bufferData(target, data, gl.STATIC_DRAW);
    return buffer;
  };

  /** Upload a parsed scene (from WorldGlb.parse). Keeps the CPU copy for context restore. */
  Renderer.prototype.load = function (scene) {
    this.scene = scene;
    return this._upload(scene);
  };

  Renderer.prototype._upload = function (scene) {
    var self = this, gl = this.gl;
    return Promise.all(scene.images.map(decodeImage)).then(function (images) {
      if (self.lost) return;
      var textures = images.map(function (image) { return self._texture(image); });
      var layers = scene.manifest.layers || {};
      var overlays = scene.manifest.overlays || {};
      var draws = scene.meshes.map(function (mesh) {
        // The dome and sky, and each station's own dome ("walk0_dome").
        var backdrop = mesh.name in BACKDROP || /_dome$/.test(mesh.name);
        var indices = mesh.indices;
        if (indices instanceof Uint32Array) {
          if (!self.uint32) throw new Error("This device cannot draw meshes this large; rebuild with the mobile profile.");
        }
        return {
          name: mesh.name,
          position: self._buffer(gl.ARRAY_BUFFER, mesh.positions),
          uv: mesh.uvs ? self._buffer(gl.ARRAY_BUFFER, mesh.uvs) : null,
          color: mesh.colors ? self._buffer(gl.ARRAY_BUFFER, mesh.colors) : null,
          index: self._buffer(gl.ELEMENT_ARRAY_BUFFER, indices),
          count: indices.length,
          indexType: indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,
          texture: mesh.image >= 0 ? textures[mesh.image] : null,
          mask: mesh.mask >= 0 ? textures[mesh.mask] : null,
          fog: layers[mesh.name] ? !!layers[mesh.name].fog : true,
          backdrop: backdrop,
          // The ground disc is coplanar with the photo's own ground, and the
          // generated surround overlaps the photo's edge at the same depth:
          // push both back so the photo always wins where it exists.
          offset: mesh.name === "ground" || mesh.name === "surround",
          // A station's bubble or dome, else the start point's (but the
          // sky, drawn from everywhere).
          station: overlays[mesh.name] ? overlays[mesh.name].station : mesh.name === "sky" ? null : [0, 0, 0],
          order: mesh.name in BACKDROP ? BACKDROP[mesh.name] : backdrop ? 0 : 10,
        };
      });
      draws.sort(function (a, b) { return a.order - b.order; });
      var stations = [[0, 0, 0]];
      draws.forEach(function (d) {
        if (d.station && !stations.some(function (s) {
          return Math.abs(s[0] - d.station[0]) + Math.abs(s[2] - d.station[2]) < 1e-6;
        })) stations.push(d.station);
      });
      draws.forEach(function (d) {
        d.stationIndex = d.station ? stations.findIndex(function (s) {
          return Math.abs(s[0] - d.station[0]) + Math.abs(s[2] - d.station[2]) < 1e-6;
        }) : -1;
      });
      self.gpu = { draws: draws, stations: stations };
      self.station = 0; // the start point
      self.previous = -1; // the station being dissolved away from, if any
      self.dissolveStart = 0;
      self.triangles = draws.reduce(function (sum, draw) { return sum + draw.count / 3; }, 0);
    });
  };

  Renderer.prototype.resize = function () {
    var ratio = Math.min(root.devicePixelRatio || 1, this.maxPixelRatio);
    var width = Math.max(1, Math.round(this.canvas.clientWidth * ratio));
    var height = Math.max(1, Math.round(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    return width / height;
  };

  /** camera: {eye:[3], target:[3], fovy (radians), near, far} */
  Renderer.prototype.draw = function (camera) {
    if (this.lost || !this.gpu) return;
    var gl = this.gl, loc = this.loc, env = this.scene.manifest.environment;
    var aspect = this.resize();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    var horizon = env.horizon;
    gl.clearColor(horizon[0], horizon[1], horizon[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE); // layers are double-sided

    var viewProjection = multiply(
      perspective(camera.fovy, aspect, camera.near, camera.far),
      lookAt(camera.eye, camera.target, [0, 1, 0])
    );
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(loc.viewProjection, false, viewProjection);
    gl.uniform3fv(loc.eye, camera.eye);
    gl.uniform1f(loc.fogDensity, env.fog.density);
    gl.uniform3fv(loc.fogColor, env.fog.color);
    gl.uniform1i(loc.texture, 0);
    gl.uniform1i(loc.mask, 1);
    gl.activeTexture(gl.TEXTURE0);

    var distances = this.gpu.stations.map(function (s) {
      return Math.hypot(camera.eye[0] - s[0], camera.eye[2] - s[2]);
    });
    var nearest = distances.indexOf(Math.min.apply(null, distances));
    var now = root.performance.now();
    if (nearest !== this.station && distances[this.station] - distances[nearest] > STATION_HYSTERESIS) {
      this.previous = this.station;
      this.station = nearest;
      this.dissolveStart = now;
    }
    // The two worlds take complementary halves of a screen-door pattern
    // during the dissolve, so every pixel shows one of them.
    var progress = this.previous >= 0 ? smoothstep(0, 1, (now - this.dissolveStart) / STATION_DISSOLVE) : 1;
    if (progress >= 1) this.previous = -1;
    var current = this.station, previous = this.previous;
    this.gpu.draws.forEach(function (draw) {
      var from = 0, fade = 1;
      if (draw.stationIndex === current) fade = progress;
      else if (draw.stationIndex === previous) from = progress;
      else if (draw.stationIndex >= 0) return;
      if (fade <= from) return;
      gl.uniform1f(loc.fadeFrom, from);
      gl.uniform1f(loc.fade, fade);
      gl.depthMask(!draw.backdrop);
      gl.bindBuffer(gl.ARRAY_BUFFER, draw.position);
      gl.enableVertexAttribArray(loc.position);
      gl.vertexAttribPointer(loc.position, 3, gl.FLOAT, false, 0, 0);
      if (draw.uv && loc.uv >= 0) {
        gl.bindBuffer(gl.ARRAY_BUFFER, draw.uv);
        gl.enableVertexAttribArray(loc.uv);
        gl.vertexAttribPointer(loc.uv, 2, gl.FLOAT, false, 0, 0);
      } else if (loc.uv >= 0) {
        gl.disableVertexAttribArray(loc.uv);
        gl.vertexAttrib2f(loc.uv, 0, 0);
      }
      if (draw.color && loc.color >= 0) {
        gl.bindBuffer(gl.ARRAY_BUFFER, draw.color);
        gl.enableVertexAttribArray(loc.color);
        gl.vertexAttribPointer(loc.color, 4, gl.UNSIGNED_BYTE, true, 0, 0);
      } else if (loc.color >= 0) {
        gl.disableVertexAttribArray(loc.color);
        gl.vertexAttrib4f(loc.color, 1, 1, 1, 1);
      }
      gl.uniform1f(loc.textured, draw.texture ? 1 : 0);
      gl.uniform1f(loc.masked, draw.mask ? 1 : 0);
      if (draw.mask) {
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, draw.mask);
        gl.activeTexture(gl.TEXTURE0);
      }
      gl.uniform1f(loc.fog, draw.fog ? 1 : 0);
      if (draw.texture) gl.bindTexture(gl.TEXTURE_2D, draw.texture);
      if (draw.offset) {
        gl.enable(gl.POLYGON_OFFSET_FILL);
        gl.polygonOffset(4, 64);
      }
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, draw.index);
      gl.drawElements(gl.TRIANGLES, draw.count, draw.indexType, 0);
      if (draw.offset) gl.disable(gl.POLYGON_OFFSET_FILL);
    });
    gl.depthMask(true);
    return this.previous >= 0; // still dissolving: draw again next frame
  };

  root.WorldRenderer = Renderer;
  root.WorldMath = { normalize: normalize, cross: cross, dot: dot, sub: sub };
})(window);
