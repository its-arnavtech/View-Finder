# World Engine v2: a generated world around the photo

## Why

v1 turned a photo into a layered 3D relief: correct in front, and a soft
colour blur everywhere else. Looking round a lake photo in Free roam, the
user found no world at all: "you've just made the image a little 3D on the
front". Rule-based fills cannot invent the rest of a place, and LaMa
collapses when asked to invent most of an image (PDD-REVIEW §8). The
missing piece is a **generative image model** that continues the scene in
every direction.

The requirements, from the user:

- **Any photo**: nature, rooms, streets, airplane cabins. Never tuned or
  trained per picture.
- **AI-powered**: invents what the camera never saw, plausibly.
- **Empty**: rooms and crowded places are continued without people.
- **Local first** (RTX 4060 Laptop, 8 GB), with a **cloud backend** able to
  plug in later.

## Architecture

```
photo ─┬─ depth (Depth Anything V2) ─────────────────────────┐
       ├─ scene (CLIP zero-shot) → prompts                   │
       └─ equirect canvas (photo at its true FOV)            │
              │                                              │
              ▼                                              ▼
   generate.complete_panorama: 8 perspective views   panorama.stitch_disparity:
   (6 round the horizon, then up, then down), each   depth per view, each aligned
   inpainted by the backend with what is already     (scale + shift) to what is
   known as context, pasted back                     known, photo's depth first
              │                                              │
              └──────────────┬───────────────────────────────┘
                             ▼
     surround.surround_layer: closed 360° mesh (inverse distance per
     vertex, cut at depth jumps), minus the photo's own frame
                             │
     photo layers from v1 (foreground, silhouettes, background, sky)
     meet it at the photo's edge, at the same depth
```

| Module | Role |
|---|---|
| `scene.py` | CLIP ViT-L/14 picks place, setting (nature / urban / indoor / vehicle) and lighting from fixed vocabularies. Builds per-direction prompts ("the ceiling of …" when looking up indoors). The negative prompt always excludes people, text and frames. |
| `panorama.py` | Equirectangular geometry: photo → canvas, perspective views in and out, depth stitching. |
| `generate.py` | `ViewBackend` protocol (`inpaint(image, mask, prompt, negative, seed)`) and the local SDXL backend; the view plan and loop. |
| `surround.py` | Builds the `Surround` (panorama, depth, scene) and its mesh layer. |
| `pipeline.py` | With a generator: no frame padding, the surround replaces the profile environment and the ground disc. |

### Why views, not one equirectangular image

Image models are trained on perspective photos. A 100° view of the
equirectangular canvas looks like a wide-angle photo, so the model behaves.
1024-px views (SDXL's native size) give ~10 px/degree, which a 4096×2048
canvas holds.

### Two findings from the first runs

- **Seed the hole, don't blank it.** A black hole stayed black (at strength
  0.99 the model starts from a trace of its input), and pure noise
  (strength 1.0) read the photo as a framed picture and built a wooden room
  around it. A soft push-pull continuation of the known pixels as the
  starting image gives a natural continuation: the lake's treeline, shore
  and sky carry on.
- **Fit depth in each view's own space.** A monocular network gives inverse
  depth along *its* axis; two views see the same point at different angles,
  so their values differ by the ray's cosine. Each view is aligned in its
  own inverse-depth space and converted to inverse distance.

## Models and licences

| Model | Size | Licence | Use |
|---|---|---|---|
| Depth Anything V2 Small | ~100 MB | Apache-2.0 | depth (v1) |
| LaMa (OpenCV ONNX) | ~93 MB | Apache-2.0 | hidden-background fill (v1) |
| CLIP ViT-L/14 (OpenAI) | ~1.7 GB | MIT | scene recognition |
| SDXL inpainting 0.1 + fp16-fix VAE | ~7 GB | CreativeML Open RAIL++-M / MIT | the surround |

All are loaded only from the local Hugging Face cache; `view-finder world
setup` downloads them.

## Hardware and time

- Needs an NVIDIA GPU. SDXL in fp16 with model CPU offload keeps the UNet
  (~5 GB) on the GPU and fits 8 GB.
- Measured on the RTX 4060 Laptop: ~30 s per view, so 8 views ≈ 4 minutes.
  Slower on battery (the GPU drew 10–14 W), and much slower when other
  processes hold GPU memory: Windows (WDDM) silently pages over-committed
  VRAM to system RAM instead of failing.
- Without a GPU or the model, `--generate auto` falls back to the v1
  surround.

## Risks

| Risk | Mitigation |
|---|---|
| Seams between views | Views overlap 40°; each repaints a thin band of known pixels, feathered. |
| Inconsistent depth between views | Scale and shift fitted to the overlap, trimmed least squares; 5×5 median. |
| Sky given depth (a "ceiling" over a landscape) | Sky is where the network's disparity is near zero; the mesh drops anything beyond 2× far and the dome shows it. |
| Invented people | Fixed negative prompt; "empty" in every prompt. |
| Water reflections read as deep | Open: a ground-plane prior below the horizon is the planned fix. |
| Build time (minutes) | Progress is reported per view; a cloud backend can replace the local one. |

## Phases

1. **Spike** (done): geometry, scene prompts, SDXL on 8 GB, seeding.
2. **Engine** (this change): modules above, pipeline, CLI/server flags,
   setup, tests with a stand-in backend.
3. **Quality**: water/ground prior, seam blending, view count and size per
   profile, walking room.
4. **Cloud backend**: a `ViewBackend` over HTTP (e.g. FLUX Fill), selected
   by flag.
