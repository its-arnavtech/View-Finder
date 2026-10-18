# World Engine: step into a photo

The World Engine turns one still photo into a 3D world you can walk into, in
a browser on a phone or a desktop. With an NVIDIA GPU it **generates the
world around the photo**: the full 360° from where it was taken, then what
you see as you walk in. A room continues as an empty room, a lake shore as a
lake shore. Without a GPU it falls back to a single-viewpoint 3D photo.

How a world is made:

1. Depth for the photo (Depth Anything V2), and what kind of place it is
   (CLIP: setting, lighting, season). Outdoors, **GeoCLIP first places the
   photo on the globe**; where it is sure, the prompts name the region and
   the kind of country there ("boreal forest and lakes with granite
   outcrops, in Ontario, Canada"), so what is generated behind you belongs
   to the place the photo was taken, not to a cliché of it.
2. The 360° panorama around the photo, generated view by view by Stable
   Diffusion XL inpainting, the photo kept pixel-exact inside it.
3. Depth for the whole panorama; a ground prior (nothing below the horizon is
   farther than the ground), which turns reflective water into a surface.
4. Real scale: the eye 1.6 m above the ground.
5. **Walking in**: every standing point (3 and 6.5 m into the photo, then
   to either side) gets a 360° world of its own. The nearest one is rendered
   from there on the GPU; the model fills what is missing and repaints
   everything near (far scenery and sky are kept); depth is estimated for
   the result and it becomes one continuous mesh. The viewer shows the
   nearest standing point's world, so wherever you stand you see one
   coherent surface made for that spot.
6. A standard `.glb` with a dependency-free WebGL viewer.

Like the rest of View Finder it runs locally. Photos never leave your
computer, and the models are downloaded once, explicitly. The engine's guess
at where a photo was taken (region and confidence, coordinates to ~10 km) is
recorded in the world's manifest, which travels inside the `.glb`; set
`VIEWFINDER_WORLD_LOCATE=0` to build without it.

- Design: [ARCHITECTURE.md](ARCHITECTURE.md), the generated world in
  [GENERATIVE.md](GENERATIVE.md) and the walkable world in
  [EXPLORABLE-PLAN.md](EXPLORABLE-PLAN.md)
- Product spec and phases: [PDD.md](PDD.md)
- Self-review, feasibility spike and build findings: [PDD-REVIEW.md](PDD-REVIEW.md)

## Setup

One time, with network access, install the extras and download the models:
Depth Anything V2 Small (~100 MB), LaMa (~93 MB), CLIP ViT-L/14 (~1.7 GB),
SDXL inpainting (~7 GB) and an SDXL depth ControlNet (~0.55 GB), which keeps
the places you walk to on the world's own geometry:

```bash
python -m pip install -r viewfinder/requirements-world.txt
```
```bash
view-finder world setup
```

The core runs on the pinned `torch`/`transformers`/`opencv`/`Pillow` stack in
`viewfinder/requirements.txt`. The extras (ONNX Runtime, diffusers, moderngl)
are what make a *generated* world; without them, or without an NVIDIA GPU,
the engine builds the single-viewpoint 3D photo instead, and the server says
which it is doing when it starts.

**Time:** a generated desktop world takes about 25 minutes on an RTX 4060
Laptop (8 GB) plugged in: ~5 for the 360° panorama, then ~5 for each of
the four standing points you can walk to (~3.5 without the depth guidance). A mobile world builds one standing
point. On battery it is several times slower. Re-uploading the same photo reuses its panorama.
Close memory-heavy apps: on Windows, a GPU that runs out of memory slows to a
crawl instead of failing.

**Laptops with two GPUs:** everything runs on the NVIDIA card, including the
offscreen renderer that looks at the world while it is built (it fits
alongside the generator, ~7.1 of 8 GB). `VIEWFINDER_RENDER_GPU=integrated`
moves the renderer to the integrated GPU instead; that GPU also draws the
desktop, which then stutters while a world builds.

## On your phone

On the PC, start the server so devices on your Wi-Fi can reach it:

```bash
view-finder world serve --host 0.0.0.0
```

The server prints a link like `http://192.168.1.20:8765/?t=k3m9pq2xwz`. Open
it on the phone (same Wi-Fi), tap **Take or choose a photo**, and the world
opens a few seconds later. The phone remembers the access token after the
first visit.

- Uploads are resized on the phone first (≤ 2048 px JPEG), so they are
  small, and iPhone HEIC photos convert automatically.
- **Quality: Auto** picks `mobile` on phones and `desktop` on computers.
- Off localhost, every API call needs the token, which is generated on each
  start. `--token` fixes it, and `--no-token` disables it (not recommended).
- **Tilt** (gyro parallax) needs HTTPS, because browsers withhold motion
  sensors from plain-HTTP pages. Pass a certificate with
  `--tls-cert cert.pem --tls-key key.pem` if you want it. Everything else
  works over plain HTTP.
- Worlds are kept in `local-data/worlds/` (newest 30; `--max-worlds`).
  `?world=<id>` links reopen them.

Without `--host`, the server listens only on `127.0.0.1`, which reaches this
computer only.

## From the command line

```bash
view-finder world build photo.jpg                 # photo_world.glb + photo_world.json
view-finder world build photo.jpg --html          # + photo_world.html (one file, no server)
view-finder world build photo.jpg --quality mobile --out worlds/
view-finder world build photo.jpg --depth-model heuristic   # no model needed; crude depth
view-finder world build photo.jpg --inpaint none  # skip LaMa (faster; blurred fills)
```

`--inpaint` (also on `serve`) is `auto` by default: LaMa when it is
installed, the rule-based fills otherwise. `lama` makes a missing model an
error. LaMa adds two passes per world, about 1–3 s each on a desktop CPU
(several times that on a laptop running on battery).

The standalone `*_world.html` embeds everything, so you can send it to a
phone and open it with no server. Use `--quality mobile` for files you plan
to share. The `.glb` also opens in Blender, Windows 3D Viewer and Android
Scene Viewer.

| Quality | Use | Typical size |
|---|---|---|
| `mobile` | phones, sharing | ~1 MB |
| `desktop` | computers (default for `build`) | ~3–4 MB |
| `high` | large screens | ~10+ MB |

Exit codes: `0` ok, `1` engine or model failure, `2` bad input.

## Controls

| | Phone | Desktop |
|---|---|---|
| **Look around** (default) | drag | drag, arrow keys |
| Walk forward/back | pinch | `W`/`S`, scroll |
| Step sideways | two-finger drag | `A`/`D`, right-drag or Shift-drag |
| Reset view | double-tap / **Reset** | double-click, `R` |
| Look by moving the phone | **Tilt** (HTTPS only) | — |

Dragging turns the view a full 360° and up/down, and the scene follows your
finger at any zoom. The eye sits slightly in front of a "neck" pivot, so
turning shows a little real parallax, the way turning your head does.

In a **generated world** you walk at eye height (1.6 m) at walking pace,
anywhere within reach of the standing points the world was generated
around: typically 6–9 m into the scene and a few metres to either side. As
you cross from one standing point's area to the next, the scene dissolves
into the next one's version of it over about a metre: the layout and far
scenery stay, the near detail is that point's own.

In a single-viewpoint world (no GPU, or `--generate off`) movement stays in
the small region where one photo's geometry holds up.

**Orbit** (toolbar) switches a drag to circling a point in front of the
camera. In Orbit mode, Tilt slides the camera like a 3D photo. **Free roam**
lifts every movement limit (and shows where the world ends).

### Panoramas

Images at least 2.8× wider than tall are treated as panoramas. Their columns
map linearly to angle around you, up to a full 360° wrap. You look around
them instead of seeing them squeezed into one frame.

### Phone performance

- A mobile world is about 1 MB (about 0.65 MB on the wire, gzipped): 4–5
  draw calls, ~45k triangles, 16-bit indices, and about 11 MB of GPU texture
  memory.
- The viewer renders only while something moves, and stops when the tab is
  hidden.
- The pixel ratio is capped (1.5× on phones) and adapts. It steps down if
  frames run slower than the device's own frame cap, and back up when there
  is headroom. It measures that cap itself, so iOS Low Power Mode's 30 fps
  is not mistaken for a slow GPU.
- Uploads are resized on the phone to just above the chosen quality's
  working size (1280 px for mobile).
- It works on WebGL2 and on WebGL1 (older phones).

Diagnostics: append `?debug=1` for a frame-time/triangle/pixel-ratio overlay,
or `?webgl=1` to force the WebGL1 path.

## Python

```python
from viewfinder.world import build_world
from viewfinder.world.depth import load_estimator

result = build_world("photo.jpg", profile="mobile", estimator=load_estimator())
open("photo.glb", "wb").write(result.glb)
print(result.manifest["layers"])
```

## What to expect

From where it was taken, the world looks exactly like the photo. Everything
else (the rest of the 360°, what is behind things, what you walk up to) is
**invented** by the image model, consistent with the photo but not a record
of the real place: the far shore of a lake you have never seen is a
plausible far shore, not the real one. Generated areas are kept empty of
people.

Generated worlds are in metres, scaled so the eye is 1.6 m above the
ground. Water, glass and very fine structure (wires, leaves against the sky)
are where depth is least reliable; the ground prior keeps water a surface.
Photos taken from a height (a hillside, a rooftop) are recognised and keep
the land below far away, and a standing point is only placed where there
is ground to stand on.
Walk past the explored area in Free roam and you reach the edge of the
world, where surfaces stretch.

Without the generator (no NVIDIA GPU), a world is a single-viewpoint 3D
photo: real depth as you move a little, a plausible continuation behind
objects (LaMa), and a soft surround built from the photo's own colours.

## Tests

```bash
python -m pytest viewfinder/tests/ -k world
```

All tests except two run without any model, using synthetic scenes with
known geometry and a stand-in inpainter. The real depth-model and LaMa tests
are skipped when those models are not installed.

### Stress test: "any image in, a world out"

```bash
python benchmark-harness/world_stress.py --photos 120 --street 40 --quality mobile
```

This builds worlds from a seeded random sample of the repository's travel
photos and street-level benchmark images, plus 19 generated edge cases:

- **Shape:** 8:1 panorama, 1:6 portrait, 48×36 and 7200×4800 images.
- **Pixel modes:** grayscale, CMYK, 16-bit, half-transparent.
- **Formats:** animated GIF, TIFF, WebP, BMP.
- **Content:** all-black, uniform grey, pure noise, a sky-only crop, an
  EXIF-rotated photo.
- **Should be refused:** a 12×12 image and a corrupt file.

Every world must pass `viewfinder.world.validate.validate_world`: the GLB
parses, there is a dome and a photo layer, positions are finite, mobile
meshes fit 16-bit indices, textures and file size are within budget, and
the FOV and navigation are sane. Results go to
`local-data/world-stress/{results.csv,summary.json}`, and the script exits 1
on any failure.

`--inpaint lama` runs the stress test with LaMa too (slow: it adds a few
seconds per photo).

### Visual check: how a world looks from eight poses

```bash
python benchmark-harness/world_preview.py photo.jpg [more.jpg ...] --tag after
```

This renders each world offline from eight fixed poses (front, look left,
look right, look up, a step to the navigation limit, walking in while
looking left and down, free roam) into a contact sheet under
`local-data/world-eval/`. It uses the viewer's camera model, fog and
silhouette mask, so a pose there matches the browser. Use it to compare an
engine change before and after across a set of photos. It builds with LaMa
by default; `--inpaint none` shows the fallback.
