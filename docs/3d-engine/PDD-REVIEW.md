# World Engine — PDD Self-Review and Feasibility Validation

I reviewed the [PDD](PDD.md) and [architecture](ARCHITECTURE.md) before
writing any engine code. The review did two things: it read the documents
looking for gaps, contradictions and assumptions that would break on a real
phone, and it ran a feasibility spike against the one component the plan
cannot test without hardware, the depth model. Every finding below is either
resolved in the documents (✅) or explicitly deferred (⏭).

## 1. Feasibility spike — depth model on this machine

Spike: load `depth-anything/Depth-Anything-V2-Small-hf` with
`local_files_only=True` under the pinned `transformers==4.57.6` and
`torch 2.13.0+cu130`, then run it on three repository photos resized to a
1024 px long side.

| Photo | Output shape | Warm GPU | Top-band mean | Bottom-band mean | Range |
|---|---|---|---|---|---|
| `Alaska_00001…` (600×800) | 686×518 | 83 ms | 0.00 | 5.44 | 0.00–9.00 |
| `Alabama_00002…` (1024×683) | 518×770 | 90 ms | 0.20 | 8.33 | 0.00–10.31 |
| `IMG_6790.png` (1024×745) | 518×714 | 88 ms | 1.69 | 5.16 | 0.04–10.01 |

Model load took 2.3 s. CPU inference took 870 ms. The three model files
download in about 7 s.

**What this validated**

- The model loads and runs on the pinned stack with **no new dependencies**.
- The output is **disparity**: values are larger at the bottom of the frame,
  which is near. The architecture's "larger = nearer" contract holds.
- **Open sky is exactly 0.0.** Sky detection by low normalised disparity is
  reliable with this model, not a hopeful heuristic.
- Latency is not a risk. Depth is under 0.1 s on GPU and under 1 s on CPU, so
  NFR3's 10 s budget is dominated by meshing and encoding, not the network.

**What it changed**

- The model output is **518 px on its short side**, not the input size. ✅
  The depth stage now upsamples to the working resolution with bicubic
  interpolation itself and does not assume the processor returns
  input-resolution output.
- The first spike attempt hung. The cause was a stray `cat` in my own shell
  command, not the hub. Even so, ✅ `world setup` downloads only the three
  files the engine needs (`config.json`, `preprocessor_config.json`,
  `model.safetensors`) rather than snapshotting the whole repository.

## 2. Document review findings

| # | Finding | Severity | Resolution |
|---|---|---|---|
| R1 | **Circular dependency in depth conditioning.** §3.4 normalises disparity on *non-sky* pixels, but §3.5 detects sky *from normalised* disparity. | High | ✅ The pipeline is two-pass: provisional normalisation on all pixels → sky mask → final normalisation on non-sky pixels. |
| R2 | **Tilt (gyro) cannot work over the LAN URL.** `DeviceOrientationEvent` is restricted to secure contexts on iOS Safari and Chrome, and `http://192.168.x.y` is not one. The PDD promised tilt on phones over plain HTTP. | High | ✅ The viewer hides Tilt when `!window.isSecureContext` or the API is missing. `world serve` gains `--tls-cert/--tls-key` for users who want HTTPS. Tilt works in the standalone HTML opened from `file://` on desktop and over HTTPS. The PDD now states the limitation. |
| R3 | **Other secure-context APIs.** `navigator.deviceMemory` and `crypto.randomUUID` are secure-context only, so code using them throws or returns `undefined` on the LAN URL. | Medium | ✅ The viewer feature-detects everything and uses no secure-only API on the critical path. Profile auto-selection treats a missing `deviceMemory` as unknown. |
| R4 | **Typing the token on a phone.** The PDD said "QR-free short URL", which specifies nothing, and a long token is miserable to type on a phone keyboard. | Medium | ✅ The token is 10 lowercase characters without lookalikes (~50 bits). The viewer moves it from the URL to `localStorage` and strips it with `history.replaceState`, so it is typed once. Comparison is constant-time. |
| R5 | **Navigation bounds were never defined.** The manifest had `half_extents` but no rule for computing them. | Medium | ✅ The rule is derived from the near-field depth: `n = p20(non-sky depth)`, `half_extents = (0.35n, 0.2n, 0.6n)` (x, y, z), floor 0.05. Moving further than a fraction of the nearest surface's distance is where single-view geometry visibly breaks. |
| R6 | **The background layer has its own discontinuities.** Max-filtered depth is mostly smooth, but where two occluders meet it still steps, and the layer would rubber-sheet there. | Low | ✅ The same culling is applied to the background layer at 2× the threshold. |
| R7 | **Gzip on every GLB request** costs CPU per phone download. | Low | ✅ `world.glb.gz` is precomputed once when a world is stored and served when the client accepts gzip. |
| R8 | **Client orientation handling was underspecified.** `createImageBitmap(..., {imageOrientation})` support differs across Safari versions. | Medium | ✅ The upload path decodes through an `<img>` element, which every target browser auto-orients by EXIF, then draws to a canvas. |
| R9 | **Store size.** 50 worlds × up to ~30 MB (`high`) is 1.5 GB. | Low | ✅ The default cap is 30 worlds, and `--max-worlds` is documented. |
| R10 | **Standalone HTML size.** Base64 inflates by 33%. A `high` world produces an HTML file too large for a phone. | Low | ✅ Export warns when the HTML exceeds 25 MB and recommends `--quality mobile` for sharing. |
| R11 | **Heuristic estimator and sky.** The model-free prior never produces a sky, so FR7 cannot hold under it. | Low | ✅ This is documented: under `heuristic`, the dome colour comes from the image's top band and no sky layer is built. The manifest records `sky_fraction: 0`. |
| R12 | **Packaging.** Viewer assets are not Python files, so an installed wheel would ship without them. | Medium | ✅ `pyproject.toml` declares `viewfinder.world/viewer/*` as package data. |
| R13 | **Mobile `uint16` claim.** It needed checking against the padded grid. | — | ✅ Checked. The mobile grid is at most 160×160 = 25,600 vertices per primitive, comfortably within `uint16`. The desktop maximum of 320×240 = 76,800 needs `uint32`, as the architecture says. |
| R14 | **NFR3 only covered GPU hosts.** | Low | ✅ A CPU target was added: ≤ 30 s for `mobile`. The spike's 0.9 s CPU depth makes this comfortable. |
| R15 | **Model licence.** It was only mentioned in passing. | Low | ✅ The README section states that V2-Small is Apache-2.0 and that `--depth-model-id` with Base/Large checkpoints is CC-BY-NC. |
| R16 | **Generative fills** would make far views look much better. | — | ⏭ Deferred by design (non-goal). The seams are named in architecture §9. |
| R17 | **Mesh compression** (quantisation/meshopt) would roughly halve mobile downloads. | — | ⏭ Deferred. The user has scheduled performance work for later. |

## 3. Verdict (pre-build)

The plan is feasible on the existing stack with no new dependencies. Two of
the findings, R1 and R2, would have shipped as visible bugs: a scene scaled by
its own sky, and a Tilt button that silently does nothing on every phone. Both
are fixed in the plan before the build. The build proceeds phase by phase as
laid out in PDD §8.

## 4. Build-time findings

The review continued through the build. Each of these was found either by
looking at real photos in a phone-sized browser or by a failing test. Each
one is fixed, and the ones that could regress have a test.

| # | Found by | Finding | Fix |
|---|---|---|---|
| B1 | Design check before meshing | A log-depth discontinuity test (the original §3.5) cuts holes in smooth ground near the horizon, where log-depth changes fastest per grid step. | The edge test uses the **disparity** jump, which is proportional to the parallax gap a viewer actually sees and is constant along a plane. Test: `test_grazing_ground_plane_is_never_cut`. |
| B2 | Browser, Alaska photo at 375×812 | The near-field grass was shredded into slivers with flat grey patches between them. Telea inpainting over a wide mask averaged both sides into grey, and was the 3–6 s hotspot. | A 3×3 grid median removes single-vertex depth spikes. Colour is flooded from farther neighbours only, at half resolution. The layers stage fell from 3.6–6.7 s to 0.6–2.1 s. |
| B3 | Unit test | The first flood fill accepted colour from *near* neighbours, so an occluder bled its own colour into the region it hides, which is exactly what the layer must not show. | Sources must be farther than the pixel being filled by half the jump. Test: `test_background_colour_flows_from_the_far_side`. |
| B4 | Browser, orbit drag | Orbiting toward the frame edge showed a hard, blurred vertical wall against the dome. Drag sensitivity was also ~3× too high (a 400 px drag ≈ 135°). | The pad fades toward the dome's colour along each pixel's ray. A full-width drag is now ~60° (orbit) or ~90° (fly). The portrait FOV is capped at the padded photo's vertical extent. |
| B5 | Browser, dashcam photo | A flat dark slab cut across the road. The fitted ground was correct, but the disc sat only 2 % of camera height below it while the photo's road depth noise is ~±2.5 %, so far away the disc won the depth test. | The disc is sunk 5 %, and the viewer applies a slope-scaled `polygonOffset` to it. The diagnosis also added a gate: a plane with more than 5 % of the scene below it is a foreground surface (dashboard, table), not ground. Test: `test_a_foreground_surface_with_scene_below_it_is_not_ground`. |
| B6 | Unit test | The server answered 413 before reading the body. Windows clients (and some browsers) then report a connection reset instead of the 413. | Bodies up to 4× the limit are drained before the reply. Larger ones are refused unread. |
| B7 | Smoke run | A local variable named `dome_colors` shadowed the new helper function, so every server build failed. | Renamed it. The end-to-end server test now covers this path. |

## 5. Final validation

- **Tests.** The World Engine tests are model-free except one real-model test
  (skipped when the model is not cached), and they ran with the model
  present. The full repository suite passes with no regressions. Counts are
  in the commit message.
- **Real photos.** Three repository photos were built on the real model:
  mountains and sky (Alaska), a race track (Alabama) and a dashcam road at
  dusk (`IMG_6790`). They were inspected in the browser at 375×812 and at
  desktop size. Mobile worlds are 0.8–0.9 MB (~0.63 MB gzipped on the wire)
  and build in 0.8–2.4 s on the GPU.
- **Phone flow end to end, in-browser.** File input → client resize
  (3000×2000 → 133 KB JPEG) → `POST` → polling → GLB → viewer took 3.6 s,
  with no console errors.
- **Standalone HTML** opens and renders with no server and no network.
- **Not verified on physical phones.** Gyro and real iOS/Android GPUs remain
  to be checked. Tilt cannot be exercised in the desktop browser pane.

## 6. Robustness pass: any image, look-around, phones

Requirement restated by the user: *"it should create a 3D world for any
given image; users must be able to look around with mouse/touch; it must be
optimised for phones."*

### Stress results (`benchmark-harness/world_stress.py`, real depth model)

| Profile | Inputs | Worlds | Expected rejections | Failures | Build median / p95 / max | GLB median (max) |
|---|---|---|---|---|---|---|
| mobile | 120 travel + 40 street + 19 edge = 179 | 177 | 2 | **0** | 0.65 / 0.76 / 1.16 s | 0.93 MB (1.34 MB) |
| desktop | 25 + 10 + 17 edge = 52 | 52 | 0 | **0** | 1.5 s median | 3.4 MB |
| high | 25 + 10 + 17 edge = 52 | 52 | 0 | **0** | 1.7 s median | 10.1 MB |

The mobile run had a median foreground coverage of 0.967 (share of the
surface kept after occlusion cutting); the only world below 0.6 was the
48×36 thumbnail. A ground plane was accepted for 50 % of photos and safely
declined for the rest.

### Findings

| # | Found by | Finding | Fix |
|---|---|---|---|
| C1 | Code review before the stress run | An all-sky photo leaves no foreground surface, and the pipeline raised. That breaks "any image". | A world with only the sky layer is valid. If neither layer exists, the photo is shown as one uncut surface. Every input that decodes yields a world. |
| C2 | Edge-case generator | A 16-bit PNG converted to RGB clips at 255 in Pillow, so it came out almost white. | High-bit modes (`I;16*`, `I`, `F`) are percentile-stretched to 8 bits first. |
| C3 | Browser, Bermuda photo | Looking away from the photo showed a flat gradient dome. There was no "world" to look around in. | A 360° environment panorama is synthesised from the photo (architecture §3.6), and the padding and ground disc are coloured from it. |
| C4 | Profiling the stress run | The new layers stage took a median of 2.7 s. The background colour flood (1.2 s) and two large-sigma blurs (0.6 s, 0.4 s) dominated. | A labelled distance transform replaces the flood (1175 → 23 ms), and large blurs run at reduced resolution. The median build is now **0.65 s, 4.2× faster**. |
| C5 | Browser, phone viewport | The default drag orbited a point, which is not "look around". | **Look** (360° turn with a neck pivot for parallax) is now the default and Orbit is a toggle. Tilt in Look mode is a magic window. |
| C6 | Browser, 8:1 panorama | Every image was treated as a 26 mm pinhole, so the panorama was squeezed into a 10°-tall sliver, and one swipe overshot the whole photo. | A cylindrical camera for images ≥ 2.8:1 (up to a 360° wrap), area-based working and mesh budgets, and drag rotation proportional to the field of view so the scene tracks the finger. |
| C7 | Browser, street-view photo | Turning sideways showed a seam between the photo's ground and the flat-coloured ground disc. | The disc's vertex colours are sampled from the environment panorama. |
| C8 | Unit test | `cv2.remap(BORDER_WRAP)` wraps vertically too. Looking straight down blended in the zenith colour across the pole. | Rows are clamped and only azimuth wraps. |
| C9 | Browser, measuring frame rate | The pane's own `requestAnimationFrame` runs at 16 fps with an empty loop. Adaptive resolution read that as a slow GPU. iOS Low Power Mode (30 fps cap) would trip it the same way. | The device's idle cadence is measured at start-up, and slowness is judged relative to it. |
| C10 | Synthetic touch test | `setPointerCapture` threw for a pointer ID the browser considered inactive, *before* the pointer was registered. The second finger was lost, so pinch became a one-finger drag. | The pointer is registered first and capture is guarded. Pinch now moves +0.68 / −0.84 along the view. |
| C11 | Browser, `?webgl=1` | The WebGL1 test flag (and `?debug=1`) were dropped when the app rewrote the URL to `?world=<id>`. | Flags are read at load, and the rewrite preserves other parameters. The WebGL1 path was verified rendering a 171k-triangle `uint32` world. |

### Interaction verified in the browser (375×812 phone viewport)

These were dispatched as real multi-pointer touch events, with assertions
on camera state:

- One-finger drag turns right/left and up/down.
- Two-finger pinch moves forward and back along the view.
- Two-finger drag slides the camera.
- Double-tap resets.
- The hint dismisses on first touch, and `touch-action: none` is set on the
  canvas.

Nine different worlds (portrait, city, street view, sky-only, synthetic and
real-photo panoramas, landscape, a desktop-profile world on WebGL1) loaded
with no console errors.


## 7. Visual quality pass: the slot-canyon report

The user explored a photo of The Narrows (Zion) in the browser and sent five
views of what they saw when looking around. Their verdict was "it's bad".
Every defect was real and reproduced.

### Method

The browser pane could not screenshot reliably at the time, and visual
judgements need to be repeatable, so a small offline renderer was built:
`benchmark-harness/world_preview.py`. It splats a world's GLB with a z-buffer
from six fixed poses (front, look left, look right, look up, a step to the
navigation limit, free roam). It uses the viewer's own camera model and
field-of-view rule, honours fog and the silhouette mask, and can hide single
layers to attribute an artifact to the layer that draws it. Every fix below
was judged on:
- before/after contact sheets of six photos (canyon, mountains, city,
  lighthouse portrait, street view, dashcam), with "before" rendered from
  the committed engine in a separate git worktree;
- the 179-input stress harness;
- a final check in the real WebGL viewer, on WebGL2 and on WebGL1.

### Findings

| # | Symptom (user's view) | Cause | Fix |
|---|---|---|---|
| V1 | Symmetric "Rorschach" chevrons on both canyon walls; a ghost second lighthouse; a mirrored crowd | Frame padding and the 360° surround both **mirrored** the photo outward | Padding stretches edge pixels, never mirrors. The surround is built from vertical colour profiles, with no copied image structure (architecture §3.6). Test: `test_environment_has_no_mirrored_copies_of_the_photo`. |
| V2 | Radial "starburst" when looking up | The surround stretched the photo's top row up to the zenith | Above and below the photo the surround hands over to the whole-frame profile and converges fully to the zenith/ground tone. Test: `test_looking_up_is_a_smooth_pole_not_rays`. |
| V3 | Flat floor with a hard horizon line | The ground disc reached 96 units, through the canyon walls, and was fogged while the surround was not | Disc radius is 2.5× the visible floor's reach, coloured from the surround, unfogged. |
| V4 | Horizontal colour streaks dragged out of every silhouette | The hidden-background fill copied the *nearest* donor pixel (Voronoi), which extrudes straight streaks | Push-pull pyramid fill at full resolution. Donors exclude the near side of every jump. Test: `test_push_pull_fill_is_smooth_and_stays_within_the_sources`. |
| V5 | Sawtooth silhouettes | Edge triangles were dropped whole, a staircase of grid cells | Edge cells are rebuilt flat at their near depth in a separate `silhouettes` mesh, with a per-pixel keep-mask discarding far-side texels. Silhouettes follow pixels, not the grid. Test: `test_edge_cells_are_extended_at_the_near_depth_and_kept_separate`. |
| V6 | Found while fixing V5: "Venetian blind" stripes along the canyon wall's edge | The network draws an occlusion as a ramp several pixels wide; every grid step of it became an edge cell at a slightly different depth | Ramps are snapped into steps at pixel level, and mesh and mask read the same map. Test: `test_soft_occlusion_ramps_become_steps_but_slopes_survive`. |
| V7 | Found while fixing V6: shards of sand glued to near rocks; a smeared hiker | Snapping by depth alone put the cut a pixel or two off the colour edge; an intermediate grid-level snap also flattened slopes | Ramp pixels are assigned near/far **by colour** (local means of confident near/far pixels), with a guided filter beforehand to align depth to colour. |
| V8 | Found on the six-photo comparison, not the canyon: heavy horizontal streaks through Barcelona's crowd and palms | Edge cells borrowed a 3×3-max neighbourhood depth that, in clutter, belonged to a *different* nearer object, so the "flat" extension became a sheet between two objects; a 1.2 % snapping window mixed objects too | Each edge triangle extends at *its own* near corners' mean depth on private vertices; cells where three surfaces meet are cut; the ramp window is 0.6 %. |
| V9 | Found on the comparison: a cloud touching Bermuda's left edge ran round the surround as a white stripe | Per-row profiles carry any border feature sideways | Profiles blur vertically with angular distance and hand over to the whole-frame profile. It is now a soft cloud bank, not a stripe. |
| V10 | Build median rose from 0.65 s to 2.9 s | Gaussian blurs with sigma up to ~130 px over the whole padded canvas (`_progressive_blur`, 1.8 s of 2.85 s) | Blur only the pad strips, with three box passes per Gaussian (constant cost). Mobile builds are now 0.5–1.5 s. |
| V11 | Flaky test (`test_upload_validation`, ~1 in 5 runs) | Real bug: POST rejections (404, **401 wrong token**, 400, 411) replied without reading the body, and Windows resets such connections, so a phone with a stale token could see a network failure instead of "token missing" | Every POST rejection drains the body first. Test: `test_rejected_uploads_still_deliver_their_error` (stable over repeated runs). |

### Results

- **Contact sheets (six photos, six poses each).** Every mirrored shape,
  zenith starburst and ground-disc horizon line from "before" is gone, and
  silhouettes no longer tear in the standard views. Remaining, at the
  navigation limit and in free roam: small notches along thin, noisy edges
  (dashcam tree trunks; these were also present before) and soft fills where
  hidden areas open up. Short streaks remain just above a photo's top edge
  when looking up.
- **Stress harness (mobile).** 179 inputs give 177 valid worlds, 2 correct
  rejections and 0 failures. GLB median is 1.11 MB (max 1.64 MB, budget
  4 MB). Minimum foreground coverage rose from 0.47 to 0.96: no more
  torn-up worlds.
- **Real viewer.** The canyon world draws in 6 calls (201k triangles at
  desktop quality) with `silhouettes` masked, and no console or GL errors on
  WebGL2 and on WebGL1 (`?webgl=1`).
- **Not done.** The single-photo limit stands. Content behind objects and
  outside the frame is a plausible blur, not a reconstruction. Generative
  outpainting/inpainting is the next quality step (architecture §9), and it
  needs a much larger model.

## 8. Quality pass 3: learned inpainting (LaMa)

Eight photos the engine had never been tuned on (a girl in a chicken coop,
a German street, Los Angeles, a Virginia cave, bronze bulls in Madrid,
New York, a Taipei stone lion, a Virgin Islands beach) were rendered from
the eight preview poses. The two weakest things on the sheets were the
blurred surround when looking left or right, and smeared fills where hidden
areas open up (the bulls, the lion, the cave). Both are invented content,
so this pass tried a learned inpainter: LaMa (OpenCV Model Zoo ONNX export,
Apache-2.0, ~93 MB), run by ONNX Runtime on the CPU (architecture §3.9).

### Findings

| # | Experiment | Result | Decision |
|---|---|---|---|
| L1 | LaMa outpainting the **360° surround**, photo projected into a 200° equirectangular window | Every scene collapsed into the same grey-green texture; even the pink granite plaza and the orange cave lost their colours | Not used. |
| L2 | The surround grown in steps of 1.6×, so that LaMa never invents more than ~60% of its input | Collapsed in the first step: generic mossy noise round the coop | Not used. The surround stays profile-based, which at least keeps each scene's colours. |
| L3 | LaMa for the **frame pad** (the 12% border the mesh extends past the photo) | Texture and structure carry on past the frame: granite and a bull's tail (Madrid), the lion's tree, paving and wall (Taipei), rock all round (cave). The stretched-and-blurred pad was a smear by comparison. | Used. Only the outer two-thirds of the pad fade to the surround. |
| L4 | LaMa for the **hidden background**, masking every non-donor pixel | Coherent far-side texture instead of push-pull's smears and bright blobs (cave), grass and wall behind the lion. Where most of an occluder stays visible to the model, it continues the occluder a little ("ghost" bull texture). | Used. Push-pull remains the fallback. |
| L4b | Growing the mask over each whole occluder (geodesic, through pixels at least as near as the edge's mid-level) | Masked 52–56% of the canvas instead of 45–46%, with no visible change in the band the layer actually draws; the "ghost" fragments are occluder pixels the layer never replaces or shows | Not used. |
| L5 | Session start took ~100 s | Full graph optimisation of this export, for no faster inference | Basic optimisation level: ~12 s, once per process. |
| L6 | Passes took 10–58 s instead of ~3 s | ONNX Runtime's default of one thread per physical core put work on the Core Ultra's efficiency cores, and every step waited for them (worse on battery) | At most six intra-op threads: ~3 s per pass on battery. |

### Results

- **Contact sheets (eight fresh photos, eight poses).** Wherever the view
  reaches the frame's edge, the scene now carries on: the coop fence, the
  German courtyard wall (before: a bright stretched strip), Los Angeles'
  paving (before: a beige smear across the lower left), the cave's rock.
  Stepping sideways behind the girl in the coop shows dirt texture where
  there were pink smears. No regressions in any pose. Turning far left or
  right still ends in the soft profile surround (L1, L2).
- **Stress harness with LaMa (mobile, 18 photos + 19 edge cases).** 35
  worlds, 2 correct rejections, 0 failures, including the 8:1 panorama,
  48×36, 16-bit, CMYK and all-black inputs. The cost is time: a median
  mobile build of 8.5 s (laptop on battery, sharing the CPU with the
  renderer), up from 0.5–1.5 s. `--inpaint none` restores the old speed.
- **Tests.** 195 pass; the LaMa paths are covered by a stand-in inpainter,
  plus one real-model test.

## 9. v2 and v3: a generated world, then a walkable one

The user's verdicts drove both: v1 "made the image a little 3D on the
front" (no world); v2 "only an image made 3D, no world behind it" (a
generated 360° panorama, but a bubble you could not walk out of). Design:
[GENERATIVE.md](GENERATIVE.md) and [EXPLORABLE-PLAN.md](EXPLORABLE-PLAN.md).
Every finding below was measured on the user's lake photo (515×388) with the
walkthrough renderer (`benchmark-harness/world_walk.py`), then on a canyon, a
street and an indoor corridor.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| G1 | Generated views black | The hole was blank, and at strength 0.99 the model starts from a trace of its input | Seed holes with a push-pull continuation of the known pixels plus noise |
| G2 | The photo framed as a picture in a wooden room | Pure noise (strength 1.0) | Same seeding |
| G3 | Panel seams; haze growing away from the photo | Each view's blended band was discarded; smooth seeds were copied | Blend a ~5° band back with a distance ramp; colour-match each view; noise in the seed; DPM++ 2M Karras |
| G4 | Upside-down cliff over the zenith, then a rock ceiling | One straight-up view saw treetops all round its edge | Four views tilted 50° up instead; outdoors "up" negatives exclude land |
| G5 | Comb-like streaks near the nadir | Bilinear sampling mixed unfilled black into the known boundary | Known only where every interpolated neighbour is known, eroded a pixel |
| G6 | A hard seam at the photo's frame | Generated content met the photo edge | A 2.5° ring inside the frame is repainted and faded; the photo's meshes stop at it |
| G7 | ~13 GB RAM, generation paging to a crawl | Model CPU offload shuttled the 5 GB UNet every view | UNet and VAE stay on the GPU; text encoders visit it to encode a prompt (3 per world, cached) |
| W1 | You could not walk anywhere | Navigation clamped to ±0.3 × the near-field depth around one viewpoint | Walking: stations 3, 6.5, 10 m out; the world explored from each |
| W2 | The lake tore open as you walked | Depth networks read reflections in water (and reflected sky) as far away | Ground prior: nothing below the horizon is farther than the ground (panorama and photo) |
| W3 | Grey smears 6 m in | A 515-px photo magnified across the screen | Texel density from the GPU render (`fwidth`); badly magnified surfaces within 5 m are repainted |
| W4 | Patchwork of reflections on the water | The same water repainted from several stations | A surface is regenerated at most once |
| W5 | Pale sky-coloured patches on the water | Repainted from afar at a grazing angle, where water reflects the sky | Repaint only within 5 m of the standing point |
| W6 | Bright streaks along every ridge | Sliver fills along skylines took a rock's depth | Fills narrower than 5 px dropped; a stricter sky threshold for new holes |
| W7 | The start view no longer exactly the photo | Repaints (preferred in depth ties) drew over it from everywhere | Repaints are *refinements*: drawn only within 1.5–3 m of where they were painted, screen-door faded (viewer and GPU renderer share the rule) |
| W8 | Canyon walls smeared along the slot | The density test took the larger texture axis; a rubber sheet is stretched one way | The smaller axis |
| W9 | The start view dithered | Refinements 3 m away were half faded in at the start point | The start point competes in the nearest-station rule; refinements fade out 2.5–3.5 m from their station |
| W10 | Two stations' refinements interleaved as comb patterns | Each faded by its own distance, so neighbours overlapped | Only the nearest station's refinements show, cross-faded over 0.3 m at the boundary |
| W11 | Unexplored stations reachable, and smeared | 16 views ran out before the far stations | Views capped per station; a station is walkable only if under 12% of its views is still hole or blur afterwards |
| W12 | A canyon broke into shards at the walking points | One scale and shift per view put each new patch at its own depth | A smooth correction field, exact where geometry exists, spread into the holes |
| W13 | Stations reported clean while visibly smeared | The cleanliness check rendered at a quarter of the resolution, and texel density scales with it | Checked at the generation resolution |
| W14 | Stretched fills were never improved | "Regenerate once" also excluded fills, which look stretched from other stations | Only refinements are excluded (they show near their own station only) |
| W15 | Smeared canyon walls passed the blur test | Even 2–3x magnification of an old texture smears at eye level; the threshold was 4x | Repaint anything within 5 m spreading a texel over more than 2 pixels |
| W16 | Indoors and in the canyon, walls shattered into shards between patches | Each generated view brings its own monocular depth noise; up close, a corridor's few large surfaces show every mismatch | Plane prior (`planes.py`): the ground, a ceiling and up to four walls are fitted to the panorama's geometry; depth near a plane and facing like it is snapped onto it in the panorama, the photo and every walking view. Snapping slides points along their rays, so the start view is unchanged. On a synthetic corridor with 2% depth noise: 74% of points land exactly on their wall, median error 1.35% → 0.01% |
| W17 | Looking at a corridor wall from 1.2 m showed a whole new corridor, vanishing point and all | Blurred surfaces were regenerated at full strength; with only a smear for context the model painted the prompt ("a corridor") onto the wall | Holes are generated outright; blurred surfaces are refined image-to-image at 0.45 strength (SDEdit), which keeps the wall, its doors and windows and adds detail |
| W18 | Walking into the corridor, every view past 3 m was a shredded mosaic | About ten patches painted from different stations, plus the photo's mesh, covered the same walls, each with its own depth a few centimetres off; the depth test interleaved them in slivers (a per-patch colour render showed it) | Drawing each station's patches only near it removed the mosaic but not the shredding (W19) |
| W19 | Still shredded with one station's patches at a time | The start bubble itself (the photo's layers, the panorama mesh) tears and melts seen from 3–6 m away; those surfaces are sharp, so no hole or blur test ever sent them to the model | A 360° bubble per station: the nearest bubble rendered from there, holes generated and everything within 50 m repainted (far scenery kept), depth estimated for the result, one continuous mesh; the viewer draws only the nearest station's bubble (EXPLORABLE-PLAN, round 2) |
| W20 | Walking in, a Gothic stone corridor became a modern hotel hallway | The repaint (0.8) followed the prompt, "a corridor", more than the picture | CLIP also picks the style and materials of built places for every prompt ("with gothic stone arches and vaulted ceilings"); repaint at 0.7 |
| W21 | The ground around a station looked motion-blurred | Seen from a few metres away, the start bubble's ground is magnified 5–10x: long radial streaks the repaint kept as structure | Surfaces magnified more than ~3x are blurred down to their colour layout, plus noise, before the repaint |
| W22 | Bright blobs at depth edges in station bubbles | Each cut at a depth edge showed the start point's dome, the wrong scenery from there | Each station has its own dome, its own panorama at infinity |
| W23 | A dotted line like a cable along every ridge, seen from the stations | From a station, the start point's dome (its panorama at infinity) peeks out beside the parallax-shifted 3D ridge, and its land mask, eroded a pixel, called the outermost ridge pixels sky; the repaint kept that thin dark sliver and made it a wire | Exploration widens the dome's (and the photo sky layer's) land mask, so the whole parallax gap is a hole and is generated; the repaint is also written over a rim of sky beyond near silhouettes, so moved edges land whole |
| W24 | A station taking 20 minutes instead of 4; then CUDA out of memory | Generation peaked at 7.8 of 8 GB and spilled into system memory; with RAM short as well, an allocation failed and wedged the server | The VAE decodes in 512-pixel tiles (peak 6.3 GB); the cache is freed after every view and job, and an out-of-memory error retries once |
| W25 | Two worlds screen-doored together between stations | The dissolve followed position, so stopping in the band froze it half way | The viewer switches station once another is 0.25 m nearer and dissolves over 450 ms |
| W26 | Trees beside a walking spot looked torn and wind-swept, with patches of shifted cloud around them | The depth network gives the sky around and through a crown (and beside posts, roofs, rocks) the object's own depth, one flat plateau no depth test sees; meshed, that halo of sky is glued to the tree and detaches as soon as you move, and the repaint turned the tears into streaks | `panorama.separate_sky`, on the start panorama and every station: confident sky becomes a colour palette, and land above the horizon near the sky that matches it is sky again. Indoors there is no confident sky and nothing changes |
| W27 | A tree next to a side station was still motion-blurred | That station started from the 3 m station's bubble, a repaint of a repaint (the two candidate bubbles were equally near); softness compounds with every hop and the model reads it as motion blur | A station starts from the bubble with the fewest repaints behind it unless another is a metre nearer; motion blur, streaks and bokeh are in the negative prompt |
| W28 | Overlooks: a valley seen from a hillside became a grey plain a few metres away | The ground prior (nothing below the horizon beyond a level ground at eye height) is needed for water reflections, but wrong from any elevated viewpoint (hills, rooftops, aircraft windows) | CLIP also tells where the photo was taken from (on the ground, from a height, from the air); from a height the prior is off for the panorama, the photo and the stations. And a standing point now needs ground under it: where the panorama sees much farther than a level ground at that distance would be (a drop), none is placed |
| W29 | A trunk a metre from a station, wiped down to its colours, came back as a blur | One repaint strength per view (0.7): the model kept the wipe's blur | Every pixel gets its own strength (differential diffusion as a step callback on the SDXL pipeline): holes 0.98, wiped smears 0.9, near surfaces 0.7 plus 0.05 per repaint already behind the station's source, far scenery and sky untouched |
| W30 | Treetops overhead streaked at every outdoor station | A station's upward views used the panorama's "looking up" prompt (open sky; trees, branches, ground in the negative) while repainting treetops and cliffs that were already there | Outdoors, a station's upward views use the scene's own prompt (indoors the ceiling prompt stays) |
| W31 | Forest canopy a few metres from a station streaked, though the blur test called it sharp | The test measured stretch per texture axis (fwidth); trees at different depths stretch the panorama's surface diagonally, and per axis that still read as about one texel per pixel | Stretch is the smaller singular value of the screen-to-texel Jacobian (any direction); repaint strength rises continuously from 0.7 at a texel per pixel to 0.9 at a third of one |
| W32 | The spot 6.5 m in, two repaints from the photo, stayed soft or odd (a leaning trunk became a pale slab) | Each spot is repainted from a distorted render of its neighbour's world, and how much structure to keep is a guess: keep too much and the distortions stay, too little and the model invents | Depth guidance (SDXL depth ControlNet, `diffusers/controlnet-depth-sdxl-1.0-mid`, 0.55 GB, optional): every station view is generated along the world's own geometry rendered from the spot (holes filled from the surfaces around them, sky far), so near surfaces can be repainted harder (0.8, plus 0.05 per hop, at most 0.9). Verified on the pine lake: the spot 6.5 m in is a coherent shore (no slab, no turquoise water); 7.1 GB VRAM, about 5 min a station |
| W33 | Ripple rings and banded rock on the ground round the walking spots (first guided build) | The guidance map held inverse distance, which draws level ground as circles round the station; and faint arcs already on the ground (a photo's waves laid on the ground, depth bumps at each wave) were sharpened at 0.9 and inherited by the next spot | Depth along each view's axis (`explore.view_inverse_depth`); near-level ground drawn exactly level in the guidance map (`GROUND_SNAP` 15%); guided, smears are wiped to their colours alone (`GUIDED_WIPE_SCALE`). Checked view by view against the old guidance and none, then rebuilt: no rings or bands at any spot |
| W34 | The integrated GPU at full load during a build; the desktop stuttered | The offscreen renderer ran on the integrated GPU (OpenGL picks it by default on an Optimus laptop), which also draws the desktop | `render.py` asks NVIDIA Optimus for the discrete GPU (`VIEWFINDER_RENDER_GPU`, default `discrete`; `integrated` moves it back). It fits beside the generator: 7.1 of 8 GB |
| W35 | Generated hills behind the start point washed out to a white haze, cut off in steps against the sky | The viewer fogged everything past ~100 m toward a pale horizon colour (meant to hide the single-viewpoint mesh's edge against the dome); the dome itself is unfogged, so the haze ended along the mesh's cells | Fog only for a photo in fog, and for the single-viewpoint world; a generated world on a clear day has none |
| W36 | What was generated behind you did not belong to the photo's place: a lake among the low granite islands of Ontario got alpine peaks, a mountain meadow and a giant pine sprig | CLIP called it "a mountain lake" (0.63 against 0.27 for "a calm lake among forested hills"), and the prompt carried the cliché into every unseen direction | GeoCLIP places the photo first (`locate.py`); where it is sure (probability within 500 km ≥ 0.6: every correctly placed outdoor photo of eleven regions scored ≥ 0.65, every misplaced one ≤ 0.44), the prompt names the region and the kind of country there, and location breaks close calls between places the photo plausibly shows |
