# World Engine v3: an explorable world, not a photo bubble

## Diagnosis

The user's verdict on v2: "only an image made 3D, no world behind it". Measured:

| Symptom | Cause |
|---|---|
| You cannot walk anywhere | The viewer clamps movement to a box of ±0.3 × the near-field depth (~0.3 units) around the camera. Everything was built for one viewpoint. |
| Free roam shreds the scene | Every surface (photo mesh, 360° surround) was built from the single original viewpoint. Step away and you see stretched sheets, holes and the sides of things nobody generated. |
| "No world behind it" | The generated surround is a depth-mapped panorama around one point: a bubble. Behind any object there is nothing; nothing exists outside the bubble. |

A world you can walk into has to be generated **from many viewpoints**, the way
image-to-world systems (WorldLabs Marble, LucidDreamer, Text2Room, WonderWorld)
do it: move the camera, render what exists, let the image model paint what the
move revealed, give it depth, add it to the world, repeat.

## Target

From any photo (nature, rooms, streets, vehicles), a world where you can:

- look all the way round from the start point (done in v2);
- **walk 10+ m in any open direction at eye height**, turning freely, and see
  coherent generated scenery with real geometry, including behind things;
- in rooms and crowds: empty spaces, no invented people.

## Architecture

```
photo ── depth ── scene/prompts ── 360° panorama at the origin (v2)
                                          │
                           metric scale from the ground (eye height 1.6 m)
                                          │
              ┌──────── world model: textured mesh patches ◄─────────┐
              │                                                       │
     GPU render (moderngl, offscreen) from a new station + direction  │
              │  RGB, depth, coverage                                 │
     holes = uncovered pixels (disocclusions, unseen space)           │
              │                                                       │
     SDXL inpaints the holes ── Depth Anything on the result          │
              │                                                       │
     depth aligned (scale + shift) to the rendered depth on known     │
     pixels ── new patch: the hole pixels (+ a band), meshed from     │
     this camera, cut at depth jumps ─────────────────────────────────┘
```

### P1. Metric scale and the walkable ground
- Fit the ground plane to the panorama's lower hemisphere (RANSAC on
  points below the horizon); scale the world so the eye is 1.6 m above it.
  Indoors and close-ups without a ground: nearest surface = 1.5 m.
- Walkable area: ground directions with no obstacle closer than the
  station distance (ray-marched in the panorama depth).

### P2. World model and GPU renderer
- `world/model.py`: a list of patches, each a textured mesh (positions,
  UVs, indices, an RGB texture) in world metres.
- `world/render.py`: moderngl standalone context; renders the patches from
  any pose with a perspective camera, returning RGB, metric depth and a
  coverage mask. Also used for previews (milliseconds, not minutes).

### P3. Exploration
- Stations on the walkable ground: a ring at ~3 m and one at ~7 m, biased
  toward the photo's direction (walking *into* the photo first), then
  sideways and behind; at each, views in the directions that show the most
  holes.
- For each view: render; if the uncovered fraction is significant, inpaint
  (prefill + noise, colour-matched, as in v2), estimate depth, align to the
  rendered depth over covered pixels (trimmed least squares in inverse
  depth), and add a patch for the hole pixels plus a band.
- New patches are meshed from their own camera with the v1 cutting rule
  (absolute disparity jump), so each disocclusion is a clean edge that a
  later view can fill.
- Budget: ~12 extra views (~5 min on the RTX 4060 Laptop) on top of the
  panorama.

### P4. Export and viewer
- GLB: every patch is a mesh (own texture); the photo keeps its v1 layers.
- Manifest: `navigation` becomes the walkable region (centre, half-extents
  in metres around the stations), `eye_height`, the ground plane.
- Viewer: walk mode at eye height on the ground plane (WASD, drag to look,
  pinch/scroll to walk), clamped to the walkable region; Free roam
  unchanged.

### P5. Verification (what the user will judge)
- A **walkthrough** sheet per world, rendered with the GPU renderer: the
  start view, then walking 3 m and 7 m forward, sideways, and turning round
  at each, at eye height.
- Photos: the lake (user's), a canyon, a city street, a room (indoor,
  must stay empty).
- The real viewer on WebGL2 at the end.

### P6. Product
- Progress per stage in the server; CLI/server flags; tests with stand-in
  backends and a fake renderer where the GPU is unavailable; docs; commit
  on `3D-engine` (nothing to main without the user's word).

## Built so far (and what changed on the way)

- `render.py`: offscreen OpenGL renderer (moderngl, on the NVIDIA card with
  SDXL; on the integrated GPU it fought the desktop for a GPU that was
  already busy drawing it): colour, metric depth, coverage
  (surface / sky / hole) and texel density per pixel. Every mesh but the
  dome and sky belongs to a *station*; only the nearest station's are drawn.
- `explore.py`: a **360° bubble per station** (round 2, below). Stations go
  into the photo first (3, 6.5 m ahead), then to either side, 10 m ahead
  and behind, as the budget allows (desktop 4, high 6, mobile 1).
- Depth for each bubble: Depth Anything on the repainted views, aligned to
  the rendered depth, then the ground prior and the plane prior
  (`planes.py`).
- Viewer: walking at 1.6 m eye height and 2 m/s, kept within 2.5 m of a
  station; the nearest station's world is drawn, dissolving into the next
  over a metre.
- `benchmark-harness/world_walk.py`: the walkthrough sheet, along each
  world's own stations.

## Round 2: one coherent surface wherever you stand

**Finding 1: a mosaic.** Walking into the corridor world, every view past
3 m was shredded. A diagnostic render colouring each patch showed why: about
ten patches painted from different stations, plus the photo's own mesh,
covered the same walls, each with its own depth estimate a few centimetres
off the others, and the depth test interleaved them in slivers. Better depth
per patch (a scale and shift, a correction field, plane snapping) and
gentler repainting each helped a little; nothing made a dozen independent
estimates agree.

**Attempt: patches per station.** Each station's patches drawn only while it
is the nearest, generated against the photo world alone. The mosaic between
stations went, but the corridor stayed shredded.

**Finding 2: the start bubble itself.** The broken parts were the photo's
own layers and the panorama mesh, seen from 3–6 m away: torn at every depth
edge, melted, stretched. They are *sharp*, so neither the hole test nor the
blur test found them, and no patch was ever painted over them.

**Fix: a bubble per station.** Every station gets what the start point
has, a complete 360° world of its own:

1. render a nearby bubble from the station all the way round (six cube
   faces into an equirectangular canvas): the one with the fewest repaints
   behind it, unless another is a metre nearer;
2. repaint, view by view round the panorama (the same 11 views as the start
   panorama), one pass per view with a strength per pixel (differential
   diffusion): holes are seeded and generated outright (0.98), surfaces
   magnified more than ~3x are wiped down to their colours and repainted
   nearly so (0.9), and everything else within 50 m is repainted
   image-to-image (0.7, plus 0.05 for every repaint already behind the
   source bubble). In natural scenes, with the optional depth guidance (an
   SDXL depth ControlNet), every view also follows the world's geometry as
   rendered from the station, and near surfaces are repainted harder (0.8
   to 0.9). Built places are not guided: the geometry seen from a new spot
   is a smooth bowl where walls should be flat, and guided walls came out
   bent. The
   layout survives; the tears, smears and melt do not.
   Far scenery and sky are kept, so they match the start view. Each view
   blends over its neighbours' work across a band, and over a rim of sky
   beyond near silhouettes, so neither views nor skylines show a seam;
3. depth for the repainted canvas from Depth Anything, each view aligned to
   the rendered depth (right in scale, torn in detail); sky the network
   glued to trees and posts given back to the sky by colour
   (`panorama.separate_sky`); then the ground prior (not from a height) and
   the plane prior;
4. one continuous mesh from the station, like the start point's surround,
   with the station's own panorama as a dome behind it.

The viewer draws only the current station's bubble (the start point's is
the photo and its panorama) and dissolves into the next over half a second
as you cross into its area. Standing anywhere, you see one surface made for
that spot.

| Tried | Result |
|---|---|
| Repaint at 0.45–0.6 | The stretched start bubble's radial smears survived as "grass"; the photo's own patch stayed a hard-edged rectangle |
| Repaint at full strength (patches) | A corridor wall came back as a new corridor: the patch kept the wall's depth but not its picture |
| Repaint at 0.8 (bubbles) | Clean, and the geometry follows the picture (depth comes from the repaint), but the model followed the prompt over the photo: a Gothic stone corridor became a hotel hallway |
| 0.7, with the place's style in the prompt | CLIP picks the style and materials of built places ("with gothic stone arches and vaulted ceilings"); the whole world stays Gothic |
| Smears blurred 8 px before the repaint | Not enough: the ground around a station is magnified 5–10x, and 100-pixel streaks survived as motion blur. Blurred to the colour layout (~40 px) instead |
| The start point's dome behind every bubble | Bright blobs of the wrong scenery in each cut at a depth edge; each station now has its own dome |
| The dome's land mask as exported (eroded a pixel) | From a station the ridge at infinity peeks out beside the 3D ridge; its outermost pixels counted as sky, and the repaint turned the sliver into a cable along every ridge. Exploration widens the mask, so the gap is generated |
| Dissolve by position (0.5 m band) | Stopping between two stations left both worlds screen-doored together; a timed dissolve as you cross replaced it |
| Depth guidance fed inverse *distance* | Level ground is a circle of equal distance round the station: a bright disc straight down, arcs at the foot of every view. The ControlNet painted them as ripple rings in the water. It is fed depth along each view's axis, as a camera's depth map is |
| Guided indoors (the Gothic corridor) | Walls and ribs at 6.5 m melted into curves along the bowl-shaped geometry; unguided views were straight corridors. Built places are no longer guided |
| Guided, with faint arcs already on the ground | A photo's waves run across it; laid on the ground they are circles round the start point, and the depth network gives each wave a bump. Guided at 0.9, both came back as ripple rings and banded rock, and the next spot inherited them. The guidance now draws near-level ground exactly level (within 15%), and smears are wiped to their colours alone, since the geometry now carries the structure |

Measured on the RTX 4060 Laptop, plugged in: a station takes about 3.5
minutes (8–10 views), about 5 with depth guidance; generation peaks at
6.3 GB of VRAM with the VAE decoding in tiles (7.8 GB without, which spilled
into system memory and ran five times slower), 7.1 GB with the ControlNet
and the renderer on the same card. On battery the card is held to about
28 W instead of 60 and everything takes twice as long.

## Risks

| Risk | Mitigation |
|---|---|
| Inconsistent content between stations | Each view is conditioned on everything already rendered into it; colour matching; prompts fixed per scene. |
| Depth drift between stations | Each bubble's depth is aligned to the rendered depth of the bubble it started from; only one bubble is drawn at a time. |
| Build time | Station/view budget per profile; cached panorama; generation is the only slow part. |
| GLB size | One equirectangular texture per station (4096 × 2048 on desktop, about 2.5 MB); mobile gets one station at 2048 × 1024. |
| No OpenGL (headless servers) | Fall back to the v2 bubble world; flagged in the manifest. |
