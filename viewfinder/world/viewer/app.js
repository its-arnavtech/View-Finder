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
  var MAX_UPLOAD_SIDE = 2048;
  var STAGE_LABELS = {
    queued: "Waiting for the engine…",
    start: "Reading the photo…",
    ingest: "Estimating depth…",
    depth: "Finding sky and hidden surfaces…",
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
  };

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
  function prepareUpload(file) {
    return new Promise(function (resolve) {
      var url = URL.createObjectURL(file);
      var image = new Image();
      image.onload = function () {
        try {
          var scale = Math.min(1, MAX_UPLOAD_SIDE / Math.max(image.naturalWidth, image.naturalHeight));
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
    return prepareUpload(file)
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
        try { root.history.replaceState(null, "", "?world=" + worldId); } catch (e) { /* ignore */ }
        return showWorld(bytes);
      });
  }

  /* ---------------- rendering ---------------- */
  function fovFor(manifest) {
    var camera = manifest.camera;
    var aspect = Math.max(0.2, state.renderer.canvas.clientWidth / Math.max(1, state.renderer.canvas.clientHeight));
    var vfov = camera.vfov_deg * Math.PI / 180;
    var hfov = camera.hfov_deg * Math.PI / 180;
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

  function frame(time) {
    state.frameRequested = false;
    if (!state.renderer || !state.controls) return;
    var dt = state.lastTick ? Math.min(0.1, (time - state.lastTick) / 1000) : 0;
    state.lastTick = time;
    var animating = state.controls.sway((time - state.swayStart) / 1000);
    animating = state.controls.tick(dt) || animating;
    var manifest = state.renderer.scene.manifest;
    var view = state.controls.camera();
    state.renderer.draw({
      eye: view.eye,
      target: view.target,
      fovy: fovFor(manifest),
      near: manifest.camera.near_clip,
      far: manifest.camera.far_clip,
    });
    // Render on demand: keep going only while something is moving (NFR9).
    if (animating || state.dirty) { state.dirty = false; requestFrame(); }
    else state.lastTick = 0;
  }

  function invalidate() { state.dirty = true; requestFrame(); }

  function showWorld(bytes) {
    var scene = root.WorldGlb.parse(bytes);
    state.glbBytes = bytes;
    var canvas = $("world");
    if (!state.renderer) {
      state.renderer = new root.WorldRenderer(canvas, {
        maxPixelRatio: isMobile() ? 1.5 : 2,
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
        ? "Drag to explore · Pinch to move · Double-tap to reset"
        : "Drag to explore · Scroll to move · Right-drag to slide · WASD to fly";
      setTimeout(function () { $("hint").hidden = true; }, 6000);
      updateToolbar();
      invalidate();
    });
  }

  /* ---------------- toolbar ---------------- */
  function updateToolbar() {
    var c = state.controls;
    $("btn-mode").textContent = c.mode === "orbit" ? "Orbit" : "Fly";
    $("btn-mode").setAttribute("aria-pressed", c.mode === "fly");
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
      state.controls.mode = state.controls.mode === "orbit" ? "fly" : "orbit";
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
