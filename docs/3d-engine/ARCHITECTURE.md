# World Engine — Design and Architecture

The World Engine turns a single photograph into an explorable 3D scene: a
textured, layered mesh you can look around and move through, on a desktop
browser or a phone. It lives in `viewfinder/world/` and shares View Finder's
ground rules. It runs locally, it never downloads a model silently, and it
fails with an actionable message rather than quietly swapping in a weaker
component.

## 1. What "a 3D world from one image" can honestly mean

A single photo records one viewpoint. Anything the camera did not see is not
in the data: the back of a building, the space behind a person, everything
outside the frame. An engine can do three things with that:

1. **Recover the visible geometry.** A monocular depth network estimates how
   far away each pixel is. Projecting pixels out along their camera rays gives
   a 3D surface that looks exactly like the photo from the original viewpoint
   and shows real parallax when you move.
2. **Fill the occluded gaps plausibly.** When the viewer moves, a foreground
   object reveals the background behind it. A second, *background* layer is
   built under every occlusion edge with inpainted colour and extrapolated
   depth, so a hole shows a smeared continuation instead of a black void.
3. **Surround it with a world.** The photo is wrapped in an environment: a
   sky dome coloured from the photo's own sky, a ground plane fitted to the
   photo's floor, the frame padded outward, and distance fog that blends
   geometry into the dome. Turning the camera past the frame edge shows a
   continuation, not the end of the universe.

The result is a *2.5D layered world*, explorable within a navigation volume
around the original viewpoint. Generative "invent the other side" synthesis
(diffusion outpainting or novel-view models) is deliberately out of scope for
v1. It needs multi-gigabyte models, takes far longer than a phone user will
wait, and produces content the photo does not support. The architecture
leaves a seam for it (§9).

## 2. System context

```
 phone / desktop browser                         host PC (GPU optional)
┌──────────────────────────┐   HTTP (LAN)   ┌──────────────────────────────────┐
│ viewer/index.html        │ ─────────────▶ │ viewfinder.world.server          │
│  • pick/take photo       │  POST image    │  • token check, size caps        │
│  • downscale + JPEG      │                │  • 1-worker job queue (GPU)      │
│  • poll job              │ ◀───────────── │  • world store (disk, LRU)       │
│  • WebGL world viewer    │  GET world.glb │            │                     │
│  • touch / gyro controls │                │            ▼                     │
└──────────────────────────┘                │ viewfinder.world.pipeline        │
                                            │  ingest → depth → geometry →     │
 terminal                                   │  layers → environment → export   │
┌──────────────────────────┐                └──────────────────────────────────┘
│ view-finder world build  │ ─── same pipeline, writes .glb/.json/.html
└──────────────────────────┘
```

Heavy work (the depth network, inpainting, meshing) runs on the host. The phone
only decodes a JPEG, uploads it, and renders a pre-built mesh. This is the
central mobile decision: a phone cannot be relied on to run a PyTorch depth
model, but every phone browser from the last several years can render a
100k-triangle textured mesh in WebGL.

## 3. Pipeline

Every stage is a pure function over NumPy arrays, apart from depth estimation.
That makes each stage testable without a model and swappable later.

| # | Stage | Module | Input → Output |
|---|---|---|---|
| 1 | Ingest | `ingest.py` | bytes/path → oriented RGB `uint8` at working resolution, EXIF 35 mm focal length |
| 2 | Depth | `depth.py` | RGB → relative disparity `float32` (larger = nearer) |
| 3 | Camera | `camera.py` | size + focal hint → pinhole intrinsics |
| 4 | Conditioning | `geometry.py` | disparity → metric-like depth in `[near, far]`, edge-preserving smoothing |
| 5 | Segmentation | `layers.py` | depth + RGB → sky mask, occlusion-edge mask, background layer (colour + depth) |
| 6 | Extension | `environment.py` | RGB + depth → padded canvas (frame outpaint), ground plane, sky/fog colours |
| 7 | Meshing | `mesh.py` | depth + masks + intrinsics → layer meshes with discontinuities cut |
| 8 | Export | `gltf.py`, `export.py` | meshes + textures + manifest → GLB 2.0, manifest JSON, optional standalone HTML |

### 3.1 Ingest

- Decodes with Pillow and applies `ImageOps.exif_transpose`, because phone
  photos are almost always stored rotated with an orientation tag.
- Reads `FocalLengthIn35mmFilm` (EXIF `0xA405`) before resizing, for the camera
  stage.
- Resizes to the quality profile's working long side (§5). Every later cost
  scales with pixel count, and phone photos arrive at 12–48 MP.
- Caps decoded pixels (`Image.MAX_IMAGE_PIXELS`-style guard) so a hostile
  upload cannot exhaust memory.

### 3.2 Depth

```python
class DepthEstimator(Protocol):
    name: str
    def estimate(self, rgb: np.ndarray) -> np.ndarray: ...  # HxW float32 disparity
```

- **`depth-anything-v2-small`** (default). `depth-anything/Depth-Anything-V2-Small-hf`
  is Apache-2.0, about 25 M parameters (~100 MB), and loads through the
  `transformers` version already pinned by the project. It is loaded with
  `local_files_only=True`. If the model is not cached, the engine raises
  `DepthModelUnavailable`, whose message names `view-finder world setup`, the
  explicit one-time download. The larger V2 checkpoints are CC-BY-NC, so they
  are not the default.
- **`heuristic`**. A model-free prior: a ground-plane gradient (bottom near,
  top far), modulated by local contrast. Its depth is crude. It exists for
  tests, air-gapped smoke runs and machines without the model, and it is only
  ever used when explicitly selected.

The output is *relative* (affine-invariant) disparity. Absolute scale is
unknowable from one image, so the engine works in normalised scene units
(§3.4).

### 3.3 Camera

A pinhole model with the principal point at the image centre. The horizontal
field of view comes from the EXIF 35 mm focal length when present
(`hfov = 2·atan(36 / 2f)` across the long side). Otherwise it defaults to the
26 mm equivalent of a phone main camera (≈ 69° across the long side). A wrong
FOV only stretches depth uniformly, which looks natural, so a sensible default
is good enough.

**Panoramas.** Images at least 2.8:1 use a **cylindrical** model instead.
Columns are linear in azimuth and rows follow `tan(elevation)`, with a ~60°
vertical field and the horizontal span set by the aspect ratio, up to a full
360°. A wrapped panorama gets no side padding, because its edges meet. EXIF
focal lengths are ignored for panoramas, since they describe the source
frames, not the stitch. The working-size and mesh budgets are set by
**area** (a 4:3 frame), not long side, so panoramas keep a usable height.

### 3.4 Depth conditioning

1. Robustly normalise disparity to `[0, 1]` in **two passes** (review R1).
   First, a provisional normalisation over all pixels feeds sky detection
   (§3.5). Then the final normalisation uses the 2nd and 98th percentiles of
   **non-sky** pixels, so an infinite sky does not compress the whole scene's
   range. Depth Anything's output is 518 px on its short side, so it is first
   upsampled bicubically to the working resolution.
2. Map to depth with an inverse-depth interpolation, which keeps the
   network's depth *ordering* and its near-field precision:
   `z = 1 / (d·(1/near − 1/far) + 1/far)`, with `near = 1`, `far = profile.far`.
3. Smooth with a bilateral pass. This flattens texture noise but keeps object
   boundaries sharp, and the boundaries are what step 5 detects.
4. **Snap depth edges onto colour edges.** A guided filter (He et al. 2010),
   with the photo's luminance as the guide, moves the network's soft,
   slightly offset boundaries onto the photo's actual edges.
5. **Turn occlusion ramps into steps** (`geometry.sharpen_depth_edges`). The
   network draws a boundary as a ramp several pixels wide, and on the mesh
   grid each step of that ramp became its own edge cell. Seen from the side,
   that read as horizontal "Venetian blind" stripes. A pixel is *on a ramp*
   when its gradient is steeper than `jump / grid stride`, the same test the
   grid uses, expressed per pixel. A ground plane never qualifies. Each ramp
   pixel is assigned to the near or the far side **by colour**: whichever
   local mean colour (of the confidently near and the confidently far pixels
   around it) it is closer to. It then takes that side's disparity. The mesh
   and the silhouette mask (§3.7) both read this one map, so they agree on
   every edge.

### 3.5 Segmentation and layered depth

- **Sky.** Pixels whose normalised disparity is below a small threshold, that
  are connected to the top border and that have low local depth gradient,
  cleaned with morphology. If they cover less than 0.5 % of the frame there is
  no sky (indoor shots).
- **Occlusion edges.** A grid vertex is on an edge where its **disparity**
  (`1/z`) differs from a 4-neighbour by more than
  `profile.edge_disparity_jump`. Disparity rather than log-depth: the gap a
  moving viewer sees open up is proportional to the disparity jump, and a
  smooth ground plane has a *constant* disparity gradient right up to the
  horizon. A log-depth test cuts holes in grazing ground (found during the
  build; see [PDD-REVIEW §4](PDD-REVIEW.md#4-build-time-findings)). A 3×3
  median on the grid disparity runs first, removing single-vertex spikes
  such as grass blades and wires.
- **Background layer** (simplified "3D photo inpainting", Shih et al. 2020):
  - Region = the near side of every occlusion edge, dilated by a width
    proportional to image size. This is the region a foreground object hides.
  - Depth: grey erosion (a min filter) of disparity over the region.
    Background is by definition farther, so the farthest nearby surface is
    the best cheap estimate of what continues behind the occluder.
  - Colour: only true *occluder* pixels (nearer than something within the
    radius by more than the jump) are replaced, at full resolution. Donors
    are far-side pixels and pixels with no silhouette nearby; the near side
    of any jump (an occluder, including its rim) never donates.
    - With **LaMa** installed (§3.9), every non-donor pixel is masked and
      inpainted, so the model continues the far side's texture (rock,
      grass, a wall) and never sees the occluder it is filling behind.
    - Otherwise a **push-pull** fill: donor colours are averaged down an
      image pyramid and pulled back up, giving each hole the local
      background colour, soft and continuous.
    - History: Telea inpainting smeared both sides into flat grey and took
      3–6 s; a nearest-pixel (Voronoi) copy was fast but dragged straight
      streaks out of every silhouette, which read as tearing when moving.
  - Meshed only over the region, with the same culling at 2× the threshold,
    and pushed back 1 % to avoid z-fighting with the foreground layer.

### 3.6 Environment ("the world around the photo")

- **Frame extension.** The image and depth are padded by
  `profile.pad_fraction` on every side.
  - With **LaMa** installed (§3.9), the pad is outpainted: the scene's
    texture and structure (a tree trunk, paving, a wall's courses) carry on
    past the frame. Only the outer two-thirds of the pad fade to the
    surround.
  - Otherwise colour is the photo's edge pixels **stretched** outward
    (never mirrored: a mirrored border puts a symmetric copy of the edge
    beside it, and symmetry is the first thing the eye catches). Each pad is
    blurred *along its edge*, more the farther out it is: side pads
    vertically, top and bottom pads horizontally. That removes the stripes
    a stretch otherwise leaves.
  - Either way the pad fades toward the surround's colour in each pad
    pixel's direction, so the photo's edge dissolves into the surround
    instead of ending in a wall.
  Disparity is extrapolated linearly from each border's gradient, fading over
  the pad. A plane's disparity is affine in pixel coordinates, so floors and
  walls continue as themselves, while edge replication would build a
  vertical curtain.
- **Sky layer.** Sky pixels are not given (meaningless) depth. They are placed
  on a sphere of radius `R_sky` along their camera rays, which keeps the real
  clouds with no false parallax.
- **Environment panorama (the 360° surround).** An equirectangular texture
  (1024×512 on mobile, a power of two so WebGL1 can mipmap it) on the dome.
  It covers every direction the photo never saw, so looking around always
  shows a world:
  - Straight ahead it is the photo, softly blurred. It is mostly hidden
    behind the mesh and shows only through gaps.
  - Everywhere else it is built from **vertical colour profiles**: the
    per-row median colour of the photo's outer edge strips, its thirds and
    the whole frame. "What colour is the scene at this elevation on this
    side" is all a surround can honestly claim. Going around, it moves from
    the edge strip to that side's third to the whole frame, and is blurred
    vertically more the farther you turn, so a border feature (a cloud)
    diffuses instead of running round as a stripe.
  - Above and below the photo it converges **fully** to the zenith and
    ground tones, so looking straight up or down is a smooth pole.

  It copies no image structure, so turning around never shows mirrored
  shapes. An earlier version mirrored the photo outward, which produced
  "Rorschach" patterns and a radial starburst at the zenith. The ambient part
  is smooth by construction, so it is computed at a quarter of the
  resolution. The frame padding and the ground disc take their colours from
  this same panorama, so all three agree.
- **Ground plane.** RANSAC plane fit on 3D points from the bottom ~38 % of the
  non-sky image. Three gates must all pass: an inlier ratio of at least 45 %,
  a normal within 40° of camera-up, and **at most 5 % of the whole scene below
  the plane**. A real ground has nothing visible beneath it, while a
  dashboard or a table top does. When accepted, a disc is sunk 5 % of the
  camera height below the plane. It reaches only 2.5× the floor the photo
  actually shows, because a disc to the horizon built an infinite flat plain
  through a slot canyon's walls. It is coloured from the surround along each
  vertex's direction and left **unfogged** like the surround, so the two
  meet without a line. The viewer also draws it with a slope-scaled polygon
  offset, so the photo's own ground always wins where it exists. When rejected (close-ups,
  aerial shots, foreground surfaces) no ground is invented.
- **Fog.** Exponential fog toward the horizon colour, with density chosen so
  that geometry at `far` is about 60 % fogged. It hides the hard seam between
  the mesh and the dome.

### 3.7 Meshing

- A regular vertex grid over the (padded) depth map. The stride is chosen so
  the long side has `profile.grid_long` vertices, which decouples mesh
  complexity from photo resolution.
- Vertex = camera-ray unprojection `(x, y, z) = ((u−cx)/fx·z, −(v−cy)/fy·z, −z)`
  in glTF's right-handed, +Y-up, camera-looking-down-−Z convention. The
  original camera sits at the origin.
- Each grid cell yields two triangles. A triangle whose vertices straddle an
  occlusion edge (disparity spread above the threshold) is neither stretched
  across the gap (a "rubber sheet") nor dropped (a staircase of missing
  cells along every silhouette). It is **rebuilt flat at its own near depth**:
  its far-side corners get private vertices on the near corners' mean
  disparity, so the near object continues one cell past its edge. These
  edge triangles go in a separate `silhouettes` mesh, drawn with a
  per-pixel **keep-mask** that discards the texels on the far side of the
  cell's midpoint disparity. The background layer shows through exactly
  there, so silhouettes follow the photo's pixels, not the grid. A cell
  where three surfaces meet has no single silhouette to extend and is cut.
  The separate mesh matters on phones: a shader that can `discard` disables
  early depth testing on many mobile GPUs, so only the ~4 % of triangles
  that need it pay for it.
- Sky vertices are excluded from the foreground and silhouette meshes.
- Unused vertices are compacted away. Indices are `uint16` when a primitive has
  ≤ 65,535 vertices and `uint32` otherwise. The mobile profile is sized to stay
  in `uint16`.

### 3.8 Export

- **GLB 2.0**, written by a small dependency-free writer (`gltf.py`), with one
  node per layer: `dome`, `sky`, `ground`, `background`, `foreground`,
  `silhouettes`. Materials use `KHR_materials_unlit`, because the photo
  already contains its lighting and re-lighting it would double-shade.
  Textures are embedded JPEGs. The silhouette keep-mask is an 8-bit PNG
  (a few KB), referenced from the material's `extras.viewfinder_mask`.
  Other glTF viewers ignore it and show a soft fringe at silhouettes
  instead of a broken mesh.
  The file opens in Blender, Windows 3D Viewer, Android Scene Viewer and
  `<model-viewer>`, not only in the bundled viewer.
- **Manifest** (`*_world.json`, also embedded in the GLB scene `extras`):
  schema version, source dimensions, profile, intrinsics, start camera pose,
  navigation bounds, fog and sky colours, per-layer stats, depth model
  provenance, stage timings and warnings.
- **Standalone HTML** (optional): the viewer with the GLB inlined as base64.
  One file you can send to a phone and open with no server.

### 3.9 Inpainting (optional)

`inpaint.py` wraps **LaMa** (Suvorov et al., WACV 2022), using the OpenCV
Model Zoo's ONNX export (`opencv/inpainting_lama`, ~93 MB, Apache-2.0) run
by ONNX Runtime on the CPU. It is optional. `world setup` downloads it, it is
only ever loaded from the local cache, and without it (or without
`onnxruntime`) the rule-based fills above are used.

- **Where it is used:** two passes per world, the frame pad (§3.6) and the
  hidden background colour (§3.5). Both are thin bands next to real photo
  content, which is where LaMa is strong.
- **Where it is not, and why:** the 360° surround. Measured on the fresh
  evaluation set: with the photo covering ~15% of an equirectangular
  window, LaMa collapsed every scene (a pink granite plaza, an orange cave)
  into the same grey-green texture. Growing the window in steps so that it
  never invents more than ~60% of its input collapsed in the first step
  too. The profile-based surround keeps the scene's own colours, which
  matters more.
- **Mechanics:** the model's input is fixed at 512×512, so images are
  area-resized in and the result cubic-resized out. Only masked pixels are
  taken from it; known pixels are never altered.
- **Speed:** full graph optimisation made the session take ~100 s to build
  (Core Ultra 9 185H) for no faster inference, so the basic level is used
  (~12 s, once per process). ONNX Runtime's default of one thread per
  physical core put work on the hybrid CPU's efficiency cores, and every
  step waited for them (passes took 10–58 s). Capping it at six threads
  gives ~3 s per pass on battery.

## 4. Viewer (phone-first)

A static, dependency-free WebGL viewer in `viewfinder/world/viewer/`. Raw
WebGL rather than three.js keeps the project offline-capable (no CDN),
cuts ~600 KB from every phone page load, and the scene only needs one unlit
shader.

- **Loading.** Parses its own GLB subset (the writer and reader are a matched
  pair, and the reader validates everything it relies on).
- **Rendering.** WebGL2, falling back to WebGL1 with `OES_element_index_uint`.
  There are two programs: textured-unlit with fog, and vertex-colour with fog
  for the dome and ground. Non-power-of-two textures use clamp-to-edge, with
  mipmaps only under WebGL2.
- **Battery.** It renders on demand: frames are drawn only while the camera
  is moving, an animation runs, or the gyro is active. Rendering stops when
  the tab is hidden. `devicePixelRatio` is capped (1.5 on mobile, 2 on
  desktop).
- **Adaptive resolution.** While frames are drawn back to back, the pixel
  ratio steps down 0.25 when the median frame time exceeds 1.5× the device's
  own idle frame cadence, and steps back up with headroom. The cadence is
  measured at start-up, so a 30 fps cap (iOS Low Power Mode, throttled
  webviews) is not mistaken for a slow GPU.
- **Robustness.** It handles `webglcontextlost`/`restored` (common on iOS when
  backgrounding) by rebuilding GPU resources from the parsed GLB kept in
  memory.
- **Controls.**

  | Input | Action |
  |---|---|
  | 1-finger drag / mouse drag | **Look** (default): turn the view a full 360°. Rotation per pixel equals the field of view per pixel (× 1.4), so the scene tracks the finger at any zoom. The eye sits in front of a "neck" pivot, which adds natural parallax · **Orbit**: circle a pivot in front of the camera for maximum depth |
  | pinch / mouse wheel | move forward / back |
  | 2-finger drag / right- or shift-drag | strafe / rise |
  | double-tap / `R` | reset to the photo's viewpoint |
  | WASD + QE / arrows | fly (desktop) |
  | Tilt toggle | Look mode: a "magic window" (turn the phone to look around). Orbit mode: tilting slides the camera like a 3D photo. iOS asks permission on the tap |
  | Look / Orbit toggle | switches what a 1-finger drag does |

  Movement is softly clamped to the manifest's navigation bounds, the region
  where the reconstruction holds up. The bounds are
  `half_extents = (0.35n, 0.2n, 0.6n)`, where `n` is the 20th-percentile
  non-sky depth (review R5). Tilt is offered only when
  `window.isSecureContext`, because browsers withhold device orientation from
  plain-HTTP pages (review R2). *Free roam* removes the clamp. A short
  parallax "sway" intro plays until the first interaction, so a first-time
  user immediately sees that the photo is 3D.
- **Layout.** Uses `viewport-fit=cover`, safe-area insets, ≥ 44 px tap
  targets, `touch-action: none` on the canvas and `prefers-color-scheme`
  theming.
- **Upload.** `<input type="file" accept="image/*">` lets the phone offer both
  the camera and the gallery. The image is downscaled client-side to a 2048 px
  long side and re-encoded as JPEG before upload. That cuts mobile upload size
  by about 10× and incidentally converts iPhone HEIC to JPEG. If the client
  resize fails, the original bytes are sent and the server downsizes them.

## 5. Quality profiles

| Profile | Working long side | Grid long side | Texture long side | Approx. triangles | Index type |
|---|---|---|---|---|---|
| `mobile` | 1024 | 160 | 1536 | ~45k | `uint16` |
| `desktop` | 1536 | 320 | 2048 | ~180k | `uint32` |
| `high` | 2048 | 512 | 4096 | ~470k | `uint32` |

The viewer requests `mobile` when the device has a coarse pointer, a narrow
screen or `navigator.deviceMemory ≤ 4`, and `desktop` otherwise. The user can
override the choice. Textures stay within 4096 px, the WebGL texture limit
that phones reliably support.

## 6. Server

`view-finder world serve` runs a stdlib `ThreadingHTTPServer`. It needs no
extra dependency, and multipart parsing (`cgi`) no longer exists on Python
3.13, so uploads are the raw image body.

| Method & path | Purpose |
|---|---|
| `GET /` , `GET /static/<asset>` | viewer page and whitelisted assets |
| `GET /api/health` | depth model, device, queue depth |
| `POST /api/worlds?quality=…` | body = image bytes → `202 {job_id}` |
| `GET /api/jobs/<id>` | `queued / running(stage) / done(world_id) / error(message)` |
| `GET /api/worlds/<id>/world.glb` | the mesh (immutable cache headers, gzip) |
| `GET /api/worlds/<id>/manifest.json` | the manifest |

- **Asynchronous jobs.** Mobile browsers drop long-lived requests when the
  screen sleeps, so an upload returns immediately and the client polls.
- **One GPU worker.** A single worker owns the depth model, which loads once.
  The queue is bounded, and the server answers `503` when it is full.
- **Store.** `local-data/worlds/<uuid>/` keeps the newest N worlds (default
  30, set with `--max-worlds`; review R9) and evicts the oldest beyond that.
  `world.glb.gz` is precomputed once at store time (review R7).
- **Exposure.** It binds to `127.0.0.1` by default. `--host 0.0.0.0` is
  required for a phone on the LAN, and in that mode a random access token is
  generated and printed inside the URL, and every `/api/*` call must present
  it. IDs are validated as UUID hex before any path is built. Upload size is
  capped.

## 7. CLI

```
view-finder world build PHOTO [--quality mobile|desktop|high] [--out DIR]
                              [--html] [--depth-model depth-anything-v2-small|heuristic]
                              [--device auto|cpu|cuda]
view-finder world serve [--host 127.0.0.1] [--port 8765] [--store DIR] [...]
view-finder world setup         # one-time explicit download of the depth model
```

`build` writes `PHOTO_world.glb` and `PHOTO_world.json` beside the input (or
into `--out`), plus `PHOTO_world.html` with `--html`. Exit codes follow the
main CLI: 0 ok, 1 stage failure, 2 bad input.

## 8. Module layout

```
viewfinder/world/
  __init__.py        public API: build_world(), WorldResult
  profiles.py        QualityProfile + the three presets
  ingest.py          decode, orient, EXIF focal, resize, pixel cap
  depth.py           DepthEstimator protocol, Depth Anything adapter, heuristic
  camera.py          Intrinsics, FOV from EXIF
  geometry.py        disparity→depth, smoothing, unprojection helpers
  layers.py          sky mask, occlusion edges, background layer
  environment.py     padding/outpaint, ground plane, dome, fog, colours
  mesh.py            grid meshing, discontinuity culling, compaction
  gltf.py            GLB writer + reader
  export.py          WorldResult → files, standalone HTML
  pipeline.py        orchestration, stage timings, progress callback
  server.py          HTTP API, job queue, store, token
  cli.py             build / serve / setup
  viewer/            index.html, viewer.css, glb.js, renderer.js, controls.js, app.js
viewfinder/tests/test_world_*.py
```

## 9. Extension seams (not built in v1)

- **A generative surround.** LaMa fills thin bands well but collapses to a
  generic texture when asked to invent most of an image (§3.9), so the 360°
  surround stays profile-based. A diffusion outpainter would slot into
  `environment.environment_map`.
- **Smaller downloads.** Quantised positions (`KHR_mesh_quantization`) and
  meshopt compression slot into `gltf.py` and `glb.js` together.
- **Video input.** Several frames give multi-view geometry, a different
  reconstruction family (SfM/Gaussian splats), which plugs in behind
  `pipeline.py`.
- **Geolocation tie-in.** Label the world with View Finder's predicted place,
  or orient the sun from the location and the capture time.
