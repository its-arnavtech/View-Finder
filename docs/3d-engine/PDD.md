# World Engine — Product Development Document

| | |
|---|---|
| Product | View Finder World Engine (single image → explorable 3D world) |
| Branch | `3D-engine` |
| Status | v1 build |
| Design | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Date | 2026-10-14 |

## 1. Problem and opportunity

View Finder answers *where* a photo was taken. People also want to *be* there:
step into the scene, look around and see the depth. Today that needs
photogrammetry (dozens of photos) or a cloud service that uploads private
photos. One photo and the user's own computer should be enough to get an
explorable scene, and many users will want to view it on their phone.

## 2. Goals and non-goals

**Goals**

- G1. Turn any single still photo into an explorable 3D scene with real
  parallax, visible occlusion fill and a surrounding environment.
- G2. Work end to end from a phone: take or pick a photo, upload it to the
  host on the LAN, then explore the result with touch and tilt.
- G3. Be local-first, like the rest of View Finder: no cloud, no silent
  downloads, and fail clearly.
- G4. Produce standard output (GLB) usable outside the bundled viewer.
- G5. Build a fully tested pipeline whose stages are model-free except for
  depth.

**Non-goals (v1)**

- Generative content (diffusion outpainting or novel-view synthesis).
- Multi-image or video reconstruction.
- Metric-scale accuracy. Scenes are in normalised units.
- Performance tuning beyond the mobile profile budget. The user has
  explicitly deferred optimisation.
- Hosting on the public internet or user accounts.

## 3. Users and primary use cases

| ID | User | Scenario |
|---|---|---|
| U1 | Phone user on the same Wi-Fi as their PC | Opens the URL printed by `world serve`, takes a photo, explores it within seconds |
| U2 | Desktop CLI user | `view-finder world build photo.jpg --html`, then opens the HTML or imports the GLB into Blender |
| U3 | Sharer | Sends the standalone `*_world.html` to a phone, where it opens with no server |
| U4 | Developer | Calls `build_world()` from Python or swaps the depth estimator |

## 4. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR1 | Accept every still format the installed Pillow decodes, honouring EXIF orientation | Must |
| FR2 | Estimate per-pixel relative depth with Depth Anything V2 Small, cached-only loading | Must |
| FR3 | Provide an explicit model-free `heuristic` depth estimator | Must |
| FR4 | Derive the FOV from the EXIF 35 mm focal length, with a 26 mm-equivalent default | Should |
| FR5 | Build a foreground mesh with occlusion-edge triangles removed | Must |
| FR6 | Build a background layer with inpainted colour and extrapolated depth behind occlusion edges | Must |
| FR7 | Detect sky; render it on a far sphere plus a gradient dome for unseen directions | Must |
| FR8 | Extend the frame beyond the photo borders with a soft continuation | Should |
| FR9 | Fit a ground plane and extend it as a ground disc when the gates pass | Should |
| FR10 | Export GLB 2.0 (unlit, embedded JPEG) and a manifest JSON | Must |
| FR11 | Export a standalone single-file HTML viewer | Should |
| FR12 | Provide a WebGL viewer with touch, mouse, keyboard and gyro controls, reset and free roam | Must |
| FR13 | Run an HTTP server for phone upload, with async jobs and polling | Must |
| FR14 | Provide `view-finder world build / serve / setup` | Must |
| FR15 | Offer three quality profiles (`mobile`, `desktop`, `high`), with auto-selection in the viewer | Must |
| FR16 | Record the depth model, profile, timings and warnings in the manifest | Must |

## 5. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR1 Mobile render | The `mobile` profile renders smoothly on mid-range phones | ≤ 65,535 vertices per primitive (`uint16`), ≤ 1536 px textures, DPR ≤ 1.5 |
| NFR2 Mobile network | Small uploads and downloads | Upload ≤ ~1 MB after client resize; mobile GLB ≤ ~4 MB |
| NFR3 Latency | Upload to viewable world on the dev PC | ≤ 10 s on GPU, ≤ 30 s on CPU for the `mobile` profile |
| NFR4 Compatibility | Viewer runs on iOS Safari 15+, Android Chrome 100+ and current desktop browsers | WebGL1 + `OES_element_index_uint` minimum |
| NFR5 Privacy | No image leaves the host. Default bind is loopback. LAN mode needs a token | — |
| NFR6 Robustness | Bounded upload size, pixel cap, bounded queue, ID validation, context-loss recovery | — |
| NFR7 Offline | No CDN or network at view or build time once the model is cached | — |
| NFR8 Testability | Every stage except model inference is covered by model-free unit tests | — |
| NFR9 Battery | Render on demand and stop when the tab is hidden | — |

## 6. User experience

**Phone flow (U1)**

1. On the PC: `view-finder world serve --host 0.0.0.0`. The server prints
   `http://192.168.x.y:8765/?t=<token>`, where the token is 10 easy-to-type
   characters. The user opens it once on the phone. The viewer then remembers
   the token and removes it from the address bar (review R4).
2. Big "Take or choose a photo" button → native camera or gallery.
3. A progress card shows the stage: *Uploading → Reading depth → Building
   layers → Meshing → Downloading world*.
4. The world opens at the photo's viewpoint and gently sways once to show
   depth. A hint overlay names the gestures until the first touch.
5. A bottom toolbar holds New photo · Reset · Tilt · Free roam · Download .glb.
   Tilt appears only in a secure context (HTTPS via `--tls-cert/--tls-key`,
   or `localhost`). Browsers withhold device orientation from plain-HTTP LAN
   pages (review R2).

**Desktop flow (U2)**

```
view-finder world build images/IMG_6790.png --html
→ images/IMG_6790_world.glb / .json / .html
```

**Errors are plain language.** Examples: "Depth model not installed: run
`view-finder world setup` on the PC", "That file isn't an image", "Server is
busy, try again in a moment" (503), "Access token missing or wrong — reopen the
link printed by the server" (401).

## 7. Interfaces

### 7.1 Python API

```python
from viewfinder.world import build_world, get_profile
from viewfinder.world.depth import load_estimator

result = build_world(
    "photo.jpg",                      # path, bytes, or PIL.Image
    profile=get_profile("mobile"),
    estimator=load_estimator("depth-anything-v2-small", device="auto"),
    progress=lambda stage, fraction: ...,
)
result.glb        # bytes
result.manifest   # dict (JSON-serialisable)
```

### 7.2 HTTP API

See [ARCHITECTURE.md §6](ARCHITECTURE.md#6-server). Every `/api/*` response is
JSON except the GLB. The token is accepted as the `X-World-Token` header or
the `t` query parameter.

### 7.3 Manifest (schema v1, abbreviated)

```json
{
  "schema": "viewfinder.world/1",
  "source": {"width": 4032, "height": 3024, "working": [1024, 768]},
  "profile": "mobile",
  "depth": {"estimator": "depth-anything-v2-small", "model_id": "...", "device": "cuda"},
  "camera": {"hfov_deg": 69.4, "vfov_deg": 54.4, "focal_source": "exif|default",
              "position": [0, 0, 0], "target": [0, 0, -1]},
  "navigation": {"center": [0, 0, 0], "half_extents": [0.6, 0.35, 0.9]},
  "environment": {"sky_fraction": 0.31, "zenith": [..], "horizon": [..],
                   "fog": {"color": [..], "density": 0.03}, "ground": {"accepted": true, ...}},
  "layers": {"foreground": {"vertices": 0, "triangles": 0}, "...": {}},
  "timings_ms": {"ingest": 0, "depth": 0, "...": 0},
  "warnings": []
}
```

## 8. Delivery phases

Each phase ends with its acceptance criteria green before the next starts.

| Phase | Scope | Acceptance criteria |
|---|---|---|
| **P0 Foundations** | Package skeleton, `profiles`, `ingest`, `camera`, GLB writer + reader | GLB round-trips byte-exact accessors; the header/chunk layout matches spec 2.0; EXIF rotation and focal parsing are tested |
| **P1 Depth** | Estimator protocol, Depth Anything adapter (cached-only), heuristic, `world setup` | The adapter returns HxW float32 with *near > far* on a real photo; a missing model raises `DepthModelUnavailable` naming `world setup` |
| **P2 Geometry** | Conditioning, unprojection, grid mesh, discontinuity culling, compaction | A synthetic step-depth scene is cut exactly at the step; vertex counts match the profile grid; `uint16` holds in `mobile` |
| **P3 Layers & environment** | Sky mask, background layer, frame padding, ground RANSAC, dome, fog | A synthetic scene with a sky band and a box occluder yields the expected sky fraction, a background layer under the box edge, and an accepted ground plane with an up-facing normal |
| **P4 Export & CLI** | Manifest, `export.py`, standalone HTML, `world build`/`setup` | `world build` on a repo image writes valid GLB/JSON/HTML; the manifest validates against the schema keys; exit codes match the main CLI |
| **P5 Viewer** | WebGL renderer, GLB parse, controls, gyro, render-on-demand, mobile UI | It loads a built world in a real browser at 375×812 and at desktop size with no console errors; drag, pinch, reset and keyboard visibly move the camera |
| **P6 Server** | Upload API, job queue, store/LRU, token, gzip, health | An end-to-end HTTP test uploads an image and polls to `done`; the GLB downloads; 401 without the token on LAN; 413 when oversize; 400 on a non-image; IDs are validated |
| **P7 Validation & docs** | Full test suite, real-image runs, browser check at phone viewport, README section | All `test_world_*` pass; at least 3 real photos produce worlds that look right when inspected; README documents setup, build, serve and phone use |

## 9. Test and validation plan

- **Unit (model-free).** Each pure stage is tested on synthetic arrays with
  known answers: a step depth, a sky band, a tilted plane, a box occluder.
- **Contract.** GLB writer/reader round-trip; manifest key schema; profile
  budgets (`uint16` on mobile).
- **Integration.** `build_world` with the heuristic estimator on a real image
  (no model needed); the HTTP server is exercised in-process on an ephemeral
  port.
- **Model.** Depth Anything on a real photo when cached; skipped, never
  failed, when absent.
- **Browser.** Build a world, serve it, and load it in the built-in browser at
  phone and desktop viewports. Check the console for errors, check that
  frames render (non-blank canvas pixels) and check that interaction changes
  the frame.

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Monocular depth is wrong on unusual images | Warped scene | Navigation bounds keep the user near the true viewpoint; free roam is an opt-in |
| Rubber-sheet stretching at edges | Looks fake | Log-depth discontinuity culling plus the background layer |
| Sky gets finite depth (a "wall" of sky) | Breaks immersion | Sky detection → far sphere and dome |
| Phone GPU/memory limits | Crash or black canvas | The mobile profile budget, `uint16`, capped DPR and texture size, context-loss recovery |
| Long requests killed on mobile | Lost jobs | Async jobs with polling |
| LAN exposure of an upload endpoint | Abuse | Loopback by default, random token, size and pixel caps, bounded queue |
| Model licence | Legal | Default to V2-Small (Apache-2.0); document that larger V2 models are non-commercial |
| HEIC on iPhone | Upload rejected | Client-side canvas re-encode to JPEG; the server accepts HEIC if `pillow-heif` is installed |

## 11. Open questions (decided for v1)

1. *Server-side or on-device depth?* Server-side. On-device (ONNX/WebGPU) is a
   later optimisation, once performance work begins.
2. *three.js or raw WebGL?* Raw WebGL, which works offline, is lighter and
   needs only one shader family.
3. *Should geolocation be integrated?* Not in v1. It is listed as a seam.

## 12. PDD review log

See [PDD-REVIEW.md](PDD-REVIEW.md) for the self-review of this document, the
issues it found, and the feasibility spike that validated the plan before the
build.
