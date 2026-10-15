/* World Engine viewer app: upload → job polling → render loop → toolbar.
 * Runs in two modes:
 *   server:     served by `view-finder world serve`, uploads photos over the API.
 *   standalone: a single exported HTML with the GLB inlined as base64
 *               (window.__VIEWFINDER_WORLD__); no network at all. */
(function (root) {
  "use strict";

  var doc = root.document;
  var $ = function (id) { return doc.getElementById(id); };
  var TOKEN_KEY = "viewfinder.world.token";
  // Slightly above each profile's working resolution: anything larger is
  // discarded by the engine anyway, so sending it only costs mobile data.
  var UPLOAD_SIDE = { mobile: 1280, desktop: 2048, high: 2560 };
  // Adaptive resolution: frame-time targets (ms) for stepping the pixel ratio.
  var SLOW_FRAME_MS = 28, FAST_FRAME_MS = 15, MIN_PIXEL_RATIO = 1;
  var STAGE_LABELS = {
    queued: "Waiting for the engine…",
    start: "Reading the photo…",
    ingest: "Estimating depth…",
    depth: "Finding sky and hidden surfaces…",
    generate: "Imagining the world all around the photo (a few minutes)…",
    explore: "Walking into the world and painting what it reveals…",
    layers: "Building the world around it…",
    environment: "Meshing…",
    mesh: "Packing textures…",
    export: "Almost there…",
    done: "Downloading world…",
  };

  var state = {
    standalone: typeof root.__VIEWFINDER_WORLD__ === "string",
    token: null,
    worldId: null,
    glbBytes: null,
    renderer: null,
    controls: null,
    dirty: false,
    frameRequested: false,
    swayStart: 0,
    lastTick: 0,
    frameSamples: [],
    pixelRatioCeiling: 2,
    debug: /[?&]debug=1\b/.test(root.location.search),
    stats: { frames: 0, fps: 0, pixelRatio: 0, triangles: 0, drawCalls: 0, lastFpsTime: 0, lastFpsFrames: 0 },
  };
  // Read by automated checks and the ?debug=1 overlay.
  root.__worldStats = state.stats;
  root.__worldViewer = state;

  /* ---------------- small storage helpers (may throw in private mode) ---------------- */
  function storeGet(key) { try { return root.localStorage.getItem(key); } catch (e) { return null; } }
  function storeSet(key, value) { try { root.localStorage.setItem(key, value); } catch (e) { /* ignore */ } }

  function isMobile() {
    var coarse = root.matchMedia && root.matchMedia("(pointer: coarse)").matches;
    var small = Math.min(root.screen.width, root.screen.height) < 768;
    // navigator.deviceMemory is secure-context only; undefined means "unknown" (PDD-REVIEW R3).
    var lowMemory = typeof navigator.deviceMemory === "number" && navigator.deviceMemory <= 4;
    return !!(coarse || small || lowMemory);
  }

  /* ---------------- UI state ---------------- */
  function show(section) {
    ["home", "busy"].forEach(function (id) { $(id).hidden = id !== section; });
    var viewing = section === "viewer";
    $("toolbar").hidden = !viewing;
    doc.body.classList.toggle("viewing", viewing);
  }
  function setBusy(text, fraction) {
    show("busy");
    $("busy-stage").textContent = text;
    $("busy-bar").style.width = Math.round(Math.max(0.03, Math.min(1, fraction)) * 100) + "%";
  }
  function showError(message) {
    show("home");
    var box = $("error");
    box.textContent = message;
    box.hidden = !message;
  }

  /* ---------------- API ---------------- */
  function api(path, options) {
    options = options || {};
    var headers = options.headers || {};
    if (state.token) headers["X-World-Token"] = state.token;
    options.headers = headers;
    return fetch(path, options).then(function (response) {
      if (response.ok) return response;
      return response.json().catch(function () { return {}; }).then(function (body) {
        var message = body.error || ("Request failed (" + response.status + ")");
        if (response.status === 401) message = "Access token missing or wrong. Reopen the link printed by `view-finder world serve`.";
        if (response.status === 503) message = body.error || "The engine is busy. Try again in a moment.";
        throw new Error(message);
      });
    });
  }

  /* Downscale + re-encode on the phone: ~10x smaller uploads, and HEIC becomes JPEG
   * wherever the browser can decode it. <img> applies EXIF orientation in every
   * target browser, so the canvas receives an upright picture (PDD-REVIEW R8). */
  function prepareUpload(file, quality) {
    var maxSide = UPLOAD_SIDE[quality] || UPLOAD_SIDE.desktop;
    return new Promise(function (resolve) {
      var url = URL.createObjectURL(file);
      var image = new Image();
      image.onload = function () {
        try {
          var scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight));
          var canvas = doc.createElement("canvas");
          canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
          canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
          canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
          URL.revokeObjectURL(url);
          canvas.toBlob(function (blob) { resolve(blob || file); }, "image/jpeg", 0.9);
        } catch (e) {
          URL.revokeObjectURL(url);
          resolve(file);
        }
      };
      image.onerror = function () { URL.revokeObjectURL(url); resolve(file); }; // server decodes it instead
      image.src = url;
    });
  }

  function sleep(ms) { return new Promise(function (resolve) { setTimeout(resolve, ms); }); }

  function pollJob(jobId) {
    return api("api/jobs/" + jobId).then(function (r) { return r.json(); }).then(function (job) {
      if (job.status === "error") throw new Error(job.error || "The world could not be built.");
      if (job.status === "done") return job.world_id;
      var label = STAGE_LABELS[job.stage || job.status] || "Working…";
      if (job.status === "queued" && job.position > 0) label = "Waiting for " + job.position + " photo(s) ahead…";
      setBusy(label, 0.1 + 0.8 * (job.progress || 0));
      return sleep(600).then(function () { return pollJob(jobId); });
    });
  }

  function uploadFile(file) {
    showError("");
    setBusy("Preparing photo…", 0.02);
    var quality = $("quality").value === "auto" ? (isMobile() ? "mobile" : "desktop") : $("quality").value;
    return prepareUpload(file, quality)
      .then(function (blob) {
        setBusy("Uploading…", 0.06);
        return api("api/worlds?quality=" + encodeURIComponent(quality), {
          method: "POST",
          headers: { "Content-Type": blob.type || "application/octet-stream" },
          body: blob,
        });
      })
      .then(function (r) { return r.json(); })
      .then(function (job) { return pollJob(job.job_id); })
      .then(openWorld)
      .catch(function (error) { showError(error.message); });
  }

  function openWorld(worldId) {
    setBusy(STAGE_LABELS.done, 0.92);
    return api("api/worlds/" + worldId + "/world.glb")
      .then(function (r) { return r.arrayBuffer(); })
      .then(function (bytes) {
        state.worldId = worldId;
        try {
          var params = new URLSearchParams(root.location.search);
          params.set("world", worldId); // keep other flags (debug, webgl)
          root.history.replaceState(null, "", "?" + params.toString());
        } catch (e) { /* ignore */ }
        return showWorld(bytes);
      });
  }

  /* ---------------- rendering ---------------- */
  function fovFor(manifest) {
    var camera = manifest.camera;
    var aspect = Math.max(0.2, state.renderer.canvas.clientWidth / Math.max(1, state.renderer.canvas.clientHeight));
    var vfov = camera.vfov_deg * Math.PI / 180;
    if (camera.projection === "cylindrical") {
      // A panorama is explored by turning, not seen whole: show a natural
      // field of view and let the user look along it.
      return Math.min(Math.max(vfov * 1.25, 50 * Math.PI / 180), 80 * Math.PI / 180);
    }
    var hfov = camera.hfov_deg * Math.PI / 180;
    var generated = manifest.surround && manifest.surround.generator && manifest.surround.generator !== "none";
    if (generated) {
      // A generated world continues past the photo on every side: frame the
      // photo, widening on narrow screens so its width fits, up to 95°.
      var fit = 2 * Math.atan(Math.tan(hfov / 2) / aspect);
      return Math.min(Math.max(vfov, fit), 95 * Math.PI / 180);
    }
    // Narrow (portrait) screens: widen vertically so more of the photo's
    // width fits, but never past the padded photo's own vertical extent;
    // beyond it there is only the plain dome and ground.
    var paddedHeight = (manifest.source.padded || [0, camera.height])[1];
    var paddedVfov = 2 * Math.atan(paddedHeight / 2 / camera.fy);
    var fitWidth = 2 * Math.atan(Math.tan(hfov / 2) / aspect);
    return Math.min(Math.max(vfov, fitWidth), paddedVfov * 0.96, 95 * Math.PI / 180);
  }

  function requestFrame() {
    if (state.frameRequested || doc.hidden) return;
    state.frameRequested = true;
    root.requestAnimationFrame(frame);
  }

  /* Adaptive resolution: while frames are being drawn back-to-back, step the
   * pixel ratio down if they are slow (weak phone GPUs) and back up when
   * there is headroom. Only continuous frames count; the gaps of render-on-
   * demand idling would read as slowness. */
  function adaptResolution(frameMs) {
    var samples = state.frameSamples;
    samples.push(frameMs);
    if (samples.length < 24) return;
    samples.sort(function (a, b) { return a - b; });
    var typical = samples[samples.length >> 1];
    state.frameSamples = [];
    state.stats.frameMs = Math.round(typical * 10) / 10;
    var renderer = state.renderer;
    // Slowness is judged against the device's own idle frame cadence, not
    // a fixed 60 Hz: iOS Low Power Mode caps pages at 30 fps and some
    // webviews throttle further, and neither means the GPU is struggling.
    var cadence = state.cadenceMs || 16.7;
    var slow = Math.max(SLOW_FRAME_MS, cadence * 1.5), fast = Math.max(FAST_FRAME_MS, cadence * 1.15);
    if (typical > slow && renderer.maxPixelRatio > MIN_PIXEL_RATIO) {
      renderer.maxPixelRatio = Math.max(MIN_PIXEL_RATIO, renderer.maxPixelRatio - 0.25);
    } else if (typical < fast && renderer.maxPixelRatio < state.pixelRatioCeiling) {
      renderer.maxPixelRatio = Math.min(state.pixelRatioCeiling, renderer.maxPixelRatio + 0.25);
    }
  }

  /* Median interval of ~20 empty animation frames: the display/browser cap. */
  function measureCadence() {
    var stamps = [];
    function tick(time) {
      stamps.push(time);
      if (stamps.length < 21) { root.requestAnimationFrame(tick); return; }
      var gaps = stamps.slice(1).map(function (t, i) { return t - stamps[i]; }).sort(function (a, b) { return a - b; });
      state.cadenceMs = gaps[gaps.length >> 1];
      state.stats.cadenceMs = Math.round(state.cadenceMs * 10) / 10;
    }
    root.requestAnimationFrame(tick);
  }

  function updateStats(time) {
    var s = state.stats, renderer = state.renderer;
    s.frames += 1;
    s.pixelRatio = renderer.canvas.width / Math.max(1, renderer.canvas.clientWidth);
    s.triangles = renderer.triangles;
    s.drawCalls = renderer.gpu ? renderer.gpu.draws.length : 0;
    if (time - s.lastFpsTime >= 500 || s.frames === 1) {
      s.fps = Math.round((s.frames - s.lastFpsFrames) * 1000 / Math.max(1, time - s.lastFpsTime));
      s.lastFpsTime = time;
      s.lastFpsFrames = s.frames;
      if (state.debug) {
        $("debug").hidden = false;
        // Frame time while moving; render-on-demand idles between interactions.
        $("debug").textContent = (s.frameMs ? s.frameMs + " ms/frame" : "–") +
          " (cap " + Math.round(1000 / (state.cadenceMs || 16.7)) + " fps) · " +
          s.pixelRatio.toFixed(2) + "x · " +
          Math.round(s.triangles / 1000) + "k tris · " + s.drawCalls + " draws · " +
          renderer.canvas.width + "×" + renderer.canvas.height + (renderer.webgl2 ? " · WebGL2" : " · WebGL1");
      }
    }
  }

  function frame(time) {
    state.frameRequested = false;
    if (!state.renderer || !state.controls) return;
    var dt = state.lastTick ? Math.min(0.1, (time - state.lastTick) / 1000) : 0;
    if (state.lastTick) adaptResolution(time - state.lastTick);
    state.lastTick = time;
    var animating = state.controls.sway((time - state.swayStart) / 1000);
    animating = state.controls.tick(dt) || animating;
    var manifest = state.renderer.scene.manifest;
    var view = state.controls.camera();
    var fovy = fovFor(manifest);
    state.controls.fovy = fovy; // drag rotation tracks the finger at this field of view
    var dissolving = state.renderer.draw({
      eye: view.eye,
      target: view.target,
      fovy: fovy,
      near: manifest.camera.near_clip,
      far: manifest.camera.far_clip,
    });
    state.stats.eye = view.eye.map(function (v) { return Math.round(v * 1000) / 1000; });
    state.stats.yaw = Math.round(view.yaw * 1000) / 1000;
    state.stats.pitch = Math.round(view.pitch * 1000) / 1000;
    updateStats(time);
    // Render on demand: keep going only while something is moving (NFR9).
    if (animating || dissolving || state.dirty) { state.dirty = false; requestFrame(); }
    else state.lastTick = 0;
  }

  function invalidate() { state.dirty = true; requestFrame(); }

  function showWorld(bytes) {
    var scene = root.WorldGlb.parse(bytes);
    state.glbBytes = bytes;
    var canvas = $("world");
    if (!state.renderer) {
      state.renderer = new root.WorldRenderer(canvas, {
        maxPixelRatio: state.pixelRatioCeiling = isMobile() ? 1.5 : 2,
        onLost: function () { setBusy("Restoring the view…", 0.5); },
        onRestored: function () { show("viewer"); invalidate(); },
      });
    }
    return state.renderer.load(scene).then(function () {
      if (!state.controls) {
        state.controls = new root.WorldControls(canvas, scene.manifest, invalidate);
        state.controls.onFirstInteraction = function () { $("hint").hidden = true; };
      } else {
        state.controls.setManifest(scene.manifest);
        state.controls.interacted = false;
      }
      state.swayStart = performance.now();
      show("viewer");
      $("hint").hidden = false;
      $("hint").textContent = isMobile()
        ? "Drag to look around · Pinch to move · Double-tap to reset"
        : "Drag to look around · Scroll to move · Right-drag to slide · WASD to walk";
      setTimeout(function () { $("hint").hidden = true; }, 6000);
      updateToolbar();
      invalidate();
    });
  }

  /* ---------------- toolbar ---------------- */
  function updateToolbar() {
    var c = state.controls;
    // The button names the *other* mode it switches to.
    $("btn-mode").textContent = c.mode === "look" ? "Orbit" : "Look";
    $("btn-mode").setAttribute("aria-label", c.mode === "look" ? "Switch to orbit mode" : "Switch to look-around mode");
    $("btn-roam").setAttribute("aria-pressed", c.freeRoam);
    $("btn-tilt").hidden = !root.WorldControls.tiltSupported();
    $("btn-new").hidden = state.standalone;
  }

  function downloadGlb() {
    var name = ((state.renderer.scene.manifest.name || "world") + "_world.glb").replace(/[^\w.-]+/g, "_");
    var url = URL.createObjectURL(new Blob([state.glbBytes], { type: "model/gltf-binary" }));
    var link = doc.createElement("a");
    link.href = url;
    link.download = name;
    doc.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
  }

  function bindUi() {
    $("photo").addEventListener("change", function (event) {
      var file = event.target.files && event.target.files[0];
      event.target.value = ""; // allow picking the same file again
      if (file) uploadFile(file);
    });
    $("btn-new").addEventListener("click", function () {
      try { root.history.replaceState(null, "", "./"); } catch (e) { /* ignore */ }
      show("home");
    });
    $("btn-reset").addEventListener("click", function () { state.controls.reset(); });
    $("btn-mode").addEventListener("click", function () {
      state.controls.setMode(state.controls.mode === "look" ? "orbit" : "look");
      updateToolbar();
    });
    $("btn-roam").addEventListener("click", function () {
      state.controls.freeRoam = !state.controls.freeRoam;
      if (!state.controls.freeRoam) state.controls.reset();
      updateToolbar();
    });
    $("btn-tilt").addEventListener("click", function () {
      var button = $("btn-tilt"), c = state.controls;
      if (c._tiltHandler) { c.disableTilt(); button.setAttribute("aria-pressed", "false"); return; }
      c.enableTilt().then(function () { button.setAttribute("aria-pressed", "true"); })
        .catch(function (error) { $("hint").hidden = false; $("hint").textContent = error.message; });
    });
    $("btn-download").addEventListener("click", downloadGlb);
    root.addEventListener("resize", invalidate);
    doc.addEventListener("visibilitychange", function () { if (!doc.hidden) invalidate(); });
  }

  function base64ToBuffer(text) {
    var binary = root.atob(text);
    var bytes = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes.buffer;
  }

  function start() {
    bindUi();
    measureCadence(); // runs while the world downloads
    if (state.standalone) {
      setBusy("Opening world…", 0.5);
      showWorld(base64ToBuffer(root.__VIEWFINDER_WORLD__)).catch(function (e) { showError(e.message); });
      return;
    }
    // Token: taken once from the printed link, then remembered and removed
    // from the address bar so it isn't left in history or screenshots (PDD-REVIEW R4).
    var params = new URLSearchParams(root.location.search);
    var fromUrl = params.get("t");
    if (fromUrl) {
      storeSet(TOKEN_KEY, fromUrl);
      params.delete("t");
      var rest = params.toString();
      try { root.history.replaceState(null, "", rest ? "?" + rest : "./"); } catch (e) { /* ignore */ }
    }
    state.token = fromUrl || storeGet(TOKEN_KEY);
    $("quality-auto").textContent = "Auto (" + (isMobile() ? "mobile" : "desktop") + ")";
    var worldId = params.get("world");
    if (worldId && /^[0-9a-f]{32}$/.test(worldId)) {
      openWorld(worldId).catch(function (e) { showError(e.message); });
    } else {
      show("home");
    }
  }

  if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", start);
  else start();
})(window);
