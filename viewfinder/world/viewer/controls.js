/* Camera + input for World Engine scenes.
 *
 * Two modes share one camera (position + yaw/pitch):
 *   look (default): 1-finger drag / mouse drag turns the view, a full 360°
 *                   around and up/down. The eye sits slightly in front of
 *                   a "neck" pivot, so turning also shifts it a little,
 *                   which gives real parallax the way turning your head does.
 *   orbit:          1-finger drag circles a pivot in front of the camera,
 *                   which shows the most depth.
 * In both: pinch / wheel = move forward/back, 2-finger drag / right-drag = strafe,
 * double-tap / R = reset, WASD+QE = move, arrows = turn.
 * Movement is clamped to the manifest's navigation box unless free roam is on. */
(function (root) {
  "use strict";

  var M = root.WorldMath;
  var PITCH_LIMIT = 1.45; // ~83 degrees
  var DOUBLE_TAP_MS = 320, DOUBLE_TAP_PX = 36;

  function forwardOf(yaw, pitch) {
    return [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch)];
  }
  function rightOf(yaw) { return [Math.cos(yaw), 0, Math.sin(yaw)]; }
  function add(a, b, s) { return [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function Controls(element, manifest, onChange) {
    this.element = element;
    this.onChange = onChange;
    this.mode = "look";
    this.freeRoam = false;
    this.pointers = {};
    this.keys = {};
    this.interacted = false;
    this.onFirstInteraction = function () {};
    this.setManifest(manifest);
    this._bind();
  }

  Controls.prototype.setManifest = function (manifest) {
    this.manifest = manifest;
    var nav = manifest.navigation;
    this.center = nav.center;
    this.half = nav.half_extents;
    this.pivotDefault = Math.max(0.5, (nav.near_field_depth || 2) * 1.5);
    this.speed = Math.max(0.02, this.half[2]);
    // Eye-to-neck distance for look mode, exaggerated from anatomy so the
    // parallax is noticeable, and kept inside the navigation box.
    this.neck = this.half[0] * 0.6;
    this.reset();
  };

  Controls.prototype.reset = function () {
    this.position = [0, 0, 0];
    this.yaw = 0;
    this.pitch = 0;
    this.tiltYaw = 0;
    this.tiltPitch = 0;
    this.pivotDistance = this.pivotDefault;
    this.swayOffset = [0, 0, 0];
    this.changed();
  };

  Controls.prototype.setMode = function (mode) {
    if (mode === this.mode) return;
    if (mode === "orbit") {
      // Start orbiting from exactly where the look camera's eye is.
      var eye = this.camera().eye;
      this.position = this._clampPosition(eye);
      this.pivotDistance = this.pivotDefault;
    }
    this.mode = mode;
    this.changed();
  };

  Controls.prototype.changed = function () { this.onChange(); };

  Controls.prototype.markInteraction = function () {
    if (!this.interacted) {
      this.interacted = true;
      this.swayOffset = [0, 0, 0];
      this.onFirstInteraction();
    }
  };

  Controls.prototype._clampPosition = function (p) {
    if (this.freeRoam) return p;
    var c = this.center, h = this.half;
    return [clamp(p[0], c[0] - h[0], c[0] + h[0]), clamp(p[1], c[1] - h[1], c[1] + h[1]), clamp(p[2], c[2] - h[2], c[2] + h[2])];
  };

  Controls.prototype.pivot = function () {
    return add(this.position, forwardOf(this.yaw, this.pitch), this.pivotDistance);
  };

  Controls.prototype._aimAt = function (target) {
    var d = M.sub(target, this.position);
    var length = Math.hypot(d[0], d[1], d[2]) || 1;
    this.pivotDistance = length;
    this.pitch = clamp(Math.asin(d[1] / length), -PITCH_LIMIT, PITCH_LIMIT);
    this.yaw = Math.atan2(d[0], -d[2]);
  };

  Controls.prototype.rotate = function (dx, dy, radiansPerPixel) {
    if (this.mode === "orbit") {
      var pivot = this.pivot();
      // Drag right: the camera swings left around the pivot, so the scene follows the finger.
      this.yaw -= dx * radiansPerPixel;
      this.pitch = clamp(this.pitch + dy * radiansPerPixel, -PITCH_LIMIT, PITCH_LIMIT);
      var wanted = add(pivot, forwardOf(this.yaw, this.pitch), -this.pivotDistance);
      this.position = this._clampPosition(wanted);
      this._aimAt(pivot); // at a bound, keep looking at the pivot instead of drifting
    } else {
      this.yaw -= dx * radiansPerPixel;
      this.pitch = clamp(this.pitch + dy * radiansPerPixel, -PITCH_LIMIT, PITCH_LIMIT);
    }
    this.changed();
  };

  Controls.prototype.move = function (forward, right, up) {
    var f = forwardOf(this.yaw, this.mode === "orbit" ? this.pitch : 0);
    var before = this.position;
    var p = add(add(add(before, f, forward), rightOf(this.yaw), right), [0, 1, 0], up);
    this.position = this._clampPosition(p);
    if (this.mode === "orbit") {
      var moved = M.dot(M.sub(this.position, before), forwardOf(this.yaw, this.pitch));
      this.pivotDistance = Math.max(0.15, this.pivotDistance - moved);
    }
    this.changed();
  };

  Controls.prototype.camera = function () {
    var yaw = this.yaw + this.tiltYaw;
    var pitch = clamp(this.pitch + this.tiltPitch, -PITCH_LIMIT, PITCH_LIMIT);
    var forward = forwardOf(yaw, pitch);
    var eye = this.position;
    if (this.mode === "look") {
      // Eye = neck pivot + rotated offset; zero offset when facing forward.
      eye = this._clampPosition(add(eye, M.sub(forward, [0, 0, -1]), this.neck));
    }
    eye = add(eye, this.swayOffset, 1);
    return { eye: eye, target: add(eye, forward, 1), yaw: yaw, pitch: pitch };
  };

  /* Gentle side-to-side sway until the first interaction, to show the photo is 3D. */
  Controls.prototype.sway = function (seconds) {
    if (this.interacted) return false;
    var t = seconds * 1.4;
    var fade = Math.min(1, seconds / 0.8);
    this.swayOffset = [Math.sin(t) * this.half[0] * 0.55 * fade, Math.sin(t * 2) * this.half[1] * 0.2 * fade, 0];
    return true;
  };

  /* Continuous keyboard movement; returns true while keys are held. */
  Controls.prototype.tick = function (dt) {
    var k = this.keys, s = this.speed * dt * 1.2, turn = dt * 1.2;
    var forward = (k.KeyW ? 1 : 0) - (k.KeyS ? 1 : 0);
    var right = (k.KeyD ? 1 : 0) - (k.KeyA ? 1 : 0);
    var up = (k.KeyE ? 1 : 0) - (k.KeyQ ? 1 : 0);
    var yaw = (k.ArrowRight ? 1 : 0) - (k.ArrowLeft ? 1 : 0);
    var pitch = (k.ArrowUp ? 1 : 0) - (k.ArrowDown ? 1 : 0);
    if (!forward && !right && !up && !yaw && !pitch) return false;
    if (forward || right || up) this.move(forward * s, right * s, up * s);
    if (yaw || pitch) {
      this.yaw += yaw * turn;
      this.pitch = clamp(this.pitch + pitch * turn, -PITCH_LIMIT, PITCH_LIMIT);
      this.changed();
    }
    return true;
  };

  /* Look mode: the scene follows the finger. Rotation per pixel is the field
   * of view per pixel, with a little gain so turning all the way round takes
   * a handful of swipes, not a dozen. A fixed rate overshoots wildly at
   * narrow fields of view (panoramas, telephoto shots).
   * Orbit mode: a drag across the full width = ~60 deg. */
  var LOOK_GAIN = 1.4;
  Controls.prototype._radiansPerPixel = function () {
    if (this.mode === "look" && this.fovy) {
      return LOOK_GAIN * this.fovy / Math.max(1, this.element.clientHeight);
    }
    return 1.05 / Math.max(320, Math.min(this.element.clientWidth, this.element.clientHeight * 1.6));
  };

  Controls.prototype._bind = function () {
    var self = this, el = this.element;
    var lastTap = { time: 0, x: 0, y: 0 };
    var pinch = null;

    function points() { return Object.keys(self.pointers).map(function (id) { return self.pointers[id]; }); }
    function pinchState() {
      var p = points();
      return { distance: Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y), x: (p[0].x + p[1].x) / 2, y: (p[0].y + p[1].y) / 2 };
    }

    el.addEventListener("pointerdown", function (event) {
      // Register first: capture can throw (InvalidPointerId) for a pointer
      // that has already ended, and a lost finger would silently turn a
      // pinch into a one-finger drag.
      self.pointers[event.pointerId] = { x: event.clientX, y: event.clientY, button: event.button, shift: event.shiftKey };
      try { el.setPointerCapture(event.pointerId); } catch (e) { /* capture is an optimisation */ }
      self.markInteraction();
      if (points().length === 2) pinch = pinchState();
    });
    el.addEventListener("pointermove", function (event) {
      var p = self.pointers[event.pointerId];
      if (!p) return;
      var dx = event.clientX - p.x, dy = event.clientY - p.y;
      p.x = event.clientX; p.y = event.clientY;
      var all = points();
      var unit = self.speed * 3 / Math.max(240, el.clientHeight);
      if (all.length >= 2 && pinch) {
        var now = pinchState();
        var zoom = (now.distance - pinch.distance) / Math.max(120, el.clientHeight * 0.4);
        self.move(zoom * self.speed * 2, -(now.x - pinch.x) * unit, (now.y - pinch.y) * unit);
        pinch = now;
      } else if (all.length === 1) {
        if (p.button === 2 || p.button === 1 || p.shift) self.move(0, -dx * unit, dy * unit);
        else self.rotate(dx, dy, self._radiansPerPixel());
      }
    });
    function release(event) {
      if (!self.pointers[event.pointerId]) return;
      delete self.pointers[event.pointerId];
      if (points().length < 2) pinch = null;
      if (event.type === "pointerup" && event.pointerType !== "mouse") {
        var now = performance.now();
        if (now - lastTap.time < DOUBLE_TAP_MS && Math.hypot(event.clientX - lastTap.x, event.clientY - lastTap.y) < DOUBLE_TAP_PX) {
          self.reset();
          lastTap.time = 0;
        } else {
          lastTap = { time: now, x: event.clientX, y: event.clientY };
        }
      }
    }
    el.addEventListener("pointerup", release);
    el.addEventListener("pointercancel", release);
    el.addEventListener("dblclick", function () { self.reset(); });
    el.addEventListener("contextmenu", function (event) { event.preventDefault(); });
    el.addEventListener("wheel", function (event) {
      event.preventDefault();
      self.markInteraction();
      var delta = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY;
      self.move(-delta * self.speed * 0.0025, 0, 0);
    }, { passive: false });

    root.addEventListener("keydown", function (event) {
      if (event.target && /input|select|textarea/i.test(event.target.tagName)) return;
      if (event.code === "KeyR") { self.reset(); return; }
      self.keys[event.code] = true;
      if (/^(Key[WASDQE]|Arrow)/.test(event.code)) { event.preventDefault(); self.markInteraction(); self.changed(); }
    });
    root.addEventListener("keyup", function (event) { delete self.keys[event.code]; });
    root.addEventListener("blur", function () { self.keys = {}; self.pointers = {}; pinch = null; });
  };

  /* ---------------- device orientation ("Tilt") ----------------
   * Only offered in a secure context: browsers withhold orientation from
   * plain-HTTP pages (PDD-REVIEW R2).
   *   look mode:  a "magic window". Turning the phone turns the view, so you
   *               look around the world by moving the phone.
   *   orbit mode: tilting slides the camera, like a 3D photo. */
  Controls.tiltSupported = function () {
    return !!(root.isSecureContext && "DeviceOrientationEvent" in root);
  };

  function wrapDegrees(d) { return ((d + 540) % 360) - 180; }

  Controls.prototype.enableTilt = function () {
    var self = this;
    function start() {
      var base = null;
      self._tiltHandler = function (event) {
        if (event.beta === null || event.gamma === null) return;
        var angle = (screen.orientation && screen.orientation.angle) || root.orientation || 0;
        // Pitch axis is beta in portrait and gamma in landscape.
        var pitchAxis = event.beta, sideAxis = event.gamma;
        if (angle === 90) { pitchAxis = -event.gamma; sideAxis = event.beta; }
        else if (angle === -90 || angle === 270) { pitchAxis = event.gamma; sideAxis = -event.beta; }
        var heading = event.alpha === null ? null : event.alpha;
        if (!base) base = { pitch: pitchAxis, side: sideAxis, heading: heading };
        self.markInteraction();
        if (self.mode === "look") {
          var dHeading = heading === null || base.heading === null ? 0 : wrapDegrees(heading - base.heading);
          self.tiltYaw = -dHeading * Math.PI / 180;
          self.tiltPitch = clamp((pitchAxis - base.pitch) * Math.PI / 180, -PITCH_LIMIT, PITCH_LIMIT);
          self.swayOffset = [0, 0, 0];
        } else {
          var nx = clamp((sideAxis - base.side) / 25, -1, 1), ny = clamp((pitchAxis - base.pitch) / 25, -1, 1);
          self.tiltYaw = self.tiltPitch = 0;
          self.swayOffset = [nx * self.half[0], -ny * self.half[1], 0];
        }
        self.changed();
      };
      root.addEventListener("deviceorientation", self._tiltHandler);
      return true;
    }
    var Orientation = root.DeviceOrientationEvent;
    if (Orientation && typeof Orientation.requestPermission === "function") {
      // iOS: must be called from the tap that enabled tilt.
      return Orientation.requestPermission().then(function (state) {
        if (state !== "granted") throw new Error("Motion access was not allowed.");
        return start();
      });
    }
    return Promise.resolve(start());
  };

  Controls.prototype.disableTilt = function () {
    if (this._tiltHandler) root.removeEventListener("deviceorientation", this._tiltHandler);
    this._tiltHandler = null;
    // Keep the direction the phone was pointing as the new drag baseline.
    this.yaw += this.tiltYaw;
    this.pitch = clamp(this.pitch + this.tiltPitch, -PITCH_LIMIT, PITCH_LIMIT);
    this.tiltYaw = this.tiltPitch = 0;
    this.swayOffset = [0, 0, 0];
    this.changed();
  };

  root.WorldControls = Controls;
})(window);
