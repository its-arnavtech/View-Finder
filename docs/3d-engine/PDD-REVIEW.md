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
