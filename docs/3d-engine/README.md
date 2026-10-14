# World Engine: step into a photo

The World Engine turns one still photo into an explorable 3D world, which you
can view in a browser on a phone or a desktop. It estimates per-pixel depth,
cuts the scene into layers (foreground, the hidden background behind it, the
sky), surrounds it with a sky dome and ground, and exports a standard
`.glb` with a dependency-free WebGL viewer.

Like the rest of View Finder it runs locally. Photos never leave your
computer, and the depth model is downloaded once, explicitly.

- Design: [ARCHITECTURE.md](ARCHITECTURE.md)
- Product spec and phases: [PDD.md](PDD.md)
- Self-review, feasibility spike and build findings: [PDD-REVIEW.md](PDD-REVIEW.md)

## Setup

One time, with network access, download the depth model (Depth Anything V2
Small, ~100 MB, Apache-2.0):

```bash
view-finder world setup
```

No new Python packages are needed. The engine runs on the pinned
`torch`/`transformers`/`opencv`/`Pillow` stack in `viewfinder/requirements.txt`.

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
```

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
| Move forward/back | pinch | scroll, `W`/`S` |
| Slide | two-finger drag | right-drag or Shift-drag, `A`/`D`, `Q`/`E` |
| Reset view | double-tap / **Reset** | double-click, `R` |
| Look by moving the phone | **Tilt** (HTTPS only) | — |

Dragging turns the view a full 360° and up/down, and the scene follows your
finger at any zoom. The eye sits slightly in front of a "neck" pivot, so
turning also shows a little real parallax, the way turning your head does.
When you turn away from the photo you see a surround synthesised from it,
deliberately blurred so it never pretends to be real content.

**Orbit** (toolbar) switches a drag to circling a point in front of the
camera, which shows the most depth. In Orbit mode, Tilt slides the camera
like a 3D photo. Movement stays inside the region where single-photo
geometry holds up; **Free roam** lifts that limit.

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

One photo only records one viewpoint. The world looks exactly like the photo
from where it was taken and shows real depth as you move around it. Behind
foreground objects you see a plausible smeared continuation, not the true
hidden content. Beyond the frame the photo fades into a blurred 360°
surround built from the photo itself (mirrored outward, fading to its sky
and ground tones). Move far enough and the missing sides of
objects show. That is why movement is limited by default.

Depth is relative, so scenes are in normalised units, not metres. Glass,
mirrors, water reflections and very fine structure (wires, leaves against
sky) are where monocular depth is least reliable.

## Tests

```bash
python -m pytest viewfinder/tests/ -k world
```

All tests except one run without the depth model, using synthetic scenes
with known geometry. The real-model test is skipped when the model is not
installed.

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
