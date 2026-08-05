# Benchmark dataset inventory — Phase 1

Discovery, evaluation, and download of geolocation benchmark datasets via the
Kaggle CLI. Only datasets with **per-image GPS coordinates**, reasonable size
(~hundreds to a few thousand images), permissive licensing, and real-world
diverse photography were downloaded. Raw downloads live under
`benchmark-data/kaggle/<dataset>/` and are gitignored; this file records
provenance.

## Downloaded

### kaggle/4k-benchmark-images

| Field | Value |
|---|---|
| Kaggle slug | `jameswicker5/4k-benchmark-images` |
| URL | https://www.kaggle.com/datasets/jameswicker5/4k-benchmark-images |
| License | **ODbL-1.0** (Open Database License — open, attribution + share-alike) |
| Images | **4,002** JPGs (~321 MB) |
| Coverage | Global street-view, deliberately balanced: **667 images each across Africa, Asia, Europe, North America, Oceania, South America** |
| Usability rating | 1.0 |

**Ground-truth coordinate format — two redundant sources (they agree):**

1. **Filename-encoded:** `image_{id}_{lat}_{lon}.jpg`
   e.g. `image_10013_-35.40054119113824_140.07605424767013.jpg`
   → latitude `-35.40054…`, longitude `140.07605…`. Parse: split on `_`, take
   the last two fields (strip `.jpg`).
2. **CSV:** `balanced_4000_image_subset.csv` with columns
   `image_name, lat, lon, folder_path, continent, full_path`.
   `image_name` matches the JPG basename; `lat`/`lon` are decimal degrees.
   (`full_path`/`folder_path` are Kaggle-internal `/kaggle/input/...` paths and
   are **not** usable locally — resolve images by `image_name` instead.)

`build_manifest.py` uses the CSV as authoritative and falls back to filename
parsing for any image not in the CSV.

**Data-quality checks:** 4,002 JPGs ↔ 4,002 CSV rows (exact 1:1); **0** images
with missing coordinates; **0** out-of-range or `(0,0)` placeholder coordinates;
**0** exactly-duplicated `lat,lon` pairs. No quality concerns.

### kaggle/google-street-view

| Field | Value |
|---|---|
| Kaggle slug | `paulchambaz/google-street-view` |
| URL | https://www.kaggle.com/datasets/paulchambaz/google-street-view |
| License | **GNU LGPL-3.0** (open) |
| Images | **10,000** PNGs, 640x640 (~630 MB) |
| Coverage | Global outdoor street-view sampled from the Google Street View API |
| Usability rating | 0.875 |

**Ground-truth coordinate format:** `dataset/coords.csv` — **headerless**, two
columns `latitude,longitude` in decimal degrees. Rows map to images
**positionally by index**: CSV row *N* (0-based) ↔ `dataset/N.png`. There is no
filename-encoded coordinate and no EXIF, so the positional join is the only
route. `build_manifest.py` detects the missing header and joins by index (it also
handles a headered variant should the dataset be republished).

The author states coordinates "come directly from google street map so they are
100% accurate."

**Data-quality checks:** 10,000 PNGs ↔ 10,000 CSV rows (exact 1:1); **0** rows
out of range; **0** `(0,0)` placeholders; **0** duplicate image paths in the
manifest.

## Excluded: Im2GPS local 2k set (no ground truth)

The local `2k_random_test.zip` (2,000 images, ~308 MB) was checked exhaustively
for per-image coordinates and has **none**:

- **Companion files:** the zip contains **2,000 entries, all `.jpg`** — no
  `.txt`, `.csv`, `.mat`, `.json`, or `.xml` of any kind.
- **EXIF:** **0 of 2,001** extracted images carry *any* EXIF block (metadata was
  stripped), so there are no GPS tags.
- **Filenames:** the pattern is `{PlaceName}_{index}_{flickr_photo_id}_{hash}_{n}_{flickr_user}@Nxx.jpg`
  (e.g. `Abidjan_00001_999026917_1838ecb7ef_1158_10371983@N06.jpg`). **0
  filenames** contain decimal-coordinate patterns.

The leading token is a place name only, and its granularity is **inconsistent**
across 387 distinct tokens — mixing cities (`London`, `Paris`, `nyc`), US states
(`California`), countries (`Afghanistan`, `Germany`) and even continents
(`africa`). Geocoding these would inject ground-truth error ranging from a few km
to well over a thousand km, with no way to quantify it per image, which would
corrupt every distance-based benchmark metric. Per the benchmark rules, **no
place-name geocoding fallback was used** and the dataset is **excluded**.

*Recoverable:* the images themselves are fine. The upstream Im2GPS/Im2GPS3k
release ships coordinates in a separate file from the CMU authors; if that file
is obtained, `build_manifest.py` already has a `read_im2gps()` reader that joins
a coordinate CSV to these images by filename (`--im2gps-images` /
`--im2gps-coords`).

## Evaluated but rejected

| Dataset | Reason |
|---|---|
| `lbgan2000/imgps3k-yfcc4k-cleaned` | Has coordinate CSVs (`filtered_yfcc4k.csv`, `im2gps3k_places365.csv`) **but license is "unknown"** — flagged rather than downloaded per the licensing rule. Also 1.6 GB (~7k images, above target) and overlaps im2gps3k. Re-consider only if the unknown license is acceptable. |
| `lctngdng/im2gps3k` | Apache-2.0, but ships **only Flickr-named JPGs with no coordinate file** — fails the hard per-image-GPS requirement. Also duplicates the existing local Im2GPS set. |
| `ubitquitin/geolocation-geoguessr-images-50k` | Has GPS, but **7.1 GB / 50k images** — well above the "a few thousand" budget. |
| `habedi/large-dataset-of-geotagged-images` | **99 GB** — far too large. |
| `jbakerdstl/geolocated-imagery-dataset-scotland` | GPS present but **single region (Scotland only)** — fails the geographic-diversity requirement. |
| `ifeanyichukwunwobodo/tokyo-geotagged-flickr-images` | **Single city (Tokyo)** and only ~483 KB (too small / metadata-only). |
| `outerview/14k-fire-hydrants-worldwide-images-geo`, `outerview/global-potholes-dataset` | Geo-tagged but **single-object** (hydrants / potholes), not scene photography. |
| `google/google-landmarks-dataset` | Only URL references (~1 KB), **no images included**. |
| `mattop/geoguessr-*` (user/leaderboard/IDs) | Tabular Geoguessr **user stats**, not images with GPS. |
| ships/planes-in-satellite-imagery, geoplant, restaurants, palm-trees, etc. | Satellite/aerial single-object or non-image-geolocation — not usable. |
| `nomihsa965/traffic-signs-dataset-mapillary-and-dfg`, `zeuss2k3/mapillary-traffic-sign-dataset` | Mapillary-sourced but **single-object (traffic signs)** and 12–35 GB — not scene geolocation. |
| `stanfordu/street-view-house-numbers` (SVHN), `sahityasetu/street-view-house-numbers-images` | **House-number crops, no GPS** — digit recognition, not geolocation. |
| `amaralibey/gsv-cities` | Visual place recognition, but **23 GB** — far above budget. |
| `reiper/gtav-enhanced-dataset-for-visual-place-recognition` | **Synthetic (GTA V)** and 70 GB — fails real-world + size. |
| `abdoghanem/bdd-trajectores-p1/p2` | Driving trajectories, **6–104 GB** — far too large. |
| `mrisdal/zurich-urban-micro-aerial-vehicle` | **Aerial/MAV, single city (Zurich)** — fails diversity. |
| `sanjidh090/khulna-google-maps-dataset` | **Single city (Khulna)** and map tiles, not photography. |
| `abtinzandi/obstacle-detection-dataset` | Obstacle detection, **no per-image GPS ground truth**. |
| `joebeachcapital/pet-cats-australia`, `trainingdatapro/electric-scooters-tracking`, `zsoltlazar/equipment-losses-*` | GPS present but **tracking/telemetry tables**, not geolocatable scene photography. |
| tourist-spot / real-estate / crime / meteorite / election tabular sets | **Tabular data, no images** — irrelevant. |

## Search terms used

`im2gps`, `geolocation images gps`, `street view geolocation`, `geotagged
photos`, `gps coordinates images`, `reverse image geolocation`, `geoguessr
dataset`, `worldwide image geolocation`, `geotagged images dataset`, `image
localization gps coordinates`, `yfcc geo`, `flickr geotagged`, `photo location
prediction`, `global geolocation benchmark`, `clip geolocation dataset`, `image
coordinates latitude longitude`, `panorama geolocation`.

Round 2 added: `mapillary dataset`, `world street photos gps`, `travel photos
location`, `landscape gps coordinates`, `outdoor scene geolocation`, `im2gps3k`,
`gps image benchmark`, `street view images`, `google street view`, `worldwide
images countries`, `image geolocation`, `cross-view geo-localization`, `place
recognition dataset`, `photo gps metadata`, `scene geotagged`, `geolocation
dataset images coordinates`, `country classification images`, `world images
dataset gps`, `photo geolocation benchmark`, `street level imagery`
(**26 distinct search terms across both rounds**).

## Current manifest

| Source | Images |
|---|---|
| `kaggle/google-street-view` | 10,000 |
| `kaggle/4k-benchmark-images` | 4,002 |
| **Total** | **14,002** |

Rebuild with:

```bash
python benchmark-data/build_manifest.py --benchmark-root benchmark-data
```
