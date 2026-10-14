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
| Explore (Orbit mode) | drag | drag |
| Move forward/back | pinch | scroll, `W`/`S` |
| Slide | two-finger drag | right-drag or Shift-drag, `A`/`D`, `Q`/`E` |
| Turn (Fly mode) | drag | drag, arrow keys |
| Reset view | double-tap / **Reset** | double-click, `R` |

**Orbit** circles a point in front of the camera, which shows the most
depth. **Fly** turns the head in place to look around. Movement stays inside
the region where single-photo geometry holds up; **Free roam** lifts that
limit.

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
hidden content. Beyond the frame the photo fades into a sky and ground
coloured from the photo itself. Move far enough and the missing sides of
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
