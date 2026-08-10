# Dataset and model decision record

View Finder now uses pretrained global models instead of copying hundreds of
gigabytes of third-party training imagery into this repository. This gives the
program broader coverage while keeping the original licenses and provenance
visible.

## Selected

### OSV-5M / PLONK OSV-5M

- Official dataset: <https://huggingface.co/datasets/osv5m/osv5m>
- Project and paper: <https://osv5m.github.io/> and
  <https://arxiv.org/abs/2404.18873>
- Scale: 5.1 million street-view images from 225 countries and territories.
- Dataset license: CC BY-SA 4.0.
- Integration: the official
  [`nicolas-dufour/PLONK_OSV_5M`](https://huggingface.co/nicolas-dufour/PLONK_OSV_5M)
  checkpoint, not a local copy of the dataset.
- Best fit: road scenes, storefronts, architecture, and other street-level
  imagery.

### YFCC-100M / PLONK YFCC

- Dataset paper: <https://arxiv.org/abs/1503.01817>
- PLONK paper and model:
  <https://arxiv.org/abs/2412.06781> and
  <https://huggingface.co/nicolas-dufour/PLONK_YFCC>
- Integration: the official pretrained checkpoint.
- Best fit: general consumer photography whose visual distribution differs
  from street-view imagery.
- License caution: original Flickr media has per-item licenses. View Finder
  downloads model weights, not the source photographs.

## Local evidence stores

- **GeoNames** `allCountries` and `alternateNamesV2` are downloaded from the
  official dump, normalized into SQLite, and indexed for exact and fuzzy
  full-text search. GeoNames is CC BY 4.0.
- **Wikimedia Commons** geotagged thumbnails are discovered through the
  official MediaWiki API. The local catalog preserves each source page,
  license, artist, credit, coordinates, hashes, and download provenance.
- **OpenStreetMap** regional Geofabrik extracts can augment names for parks,
  monuments, businesses, and roads. OSM data is licensed under ODbL.

The runtime image stays local. Only the one-time dataset builder makes network
requests.

## Training-source licence verification (2026-08-08)

Verified from primary sources only, for large-scale location-encoder training.
Findings are enforced in code by `finetune/datasets/` — an adapter whose licence
is unclear or whose access is gated raises rather than ingesting.

### Mapillary Street-Level Sequences (MSLS), ~1.6M images — **UNCLEAR, blocked**

| Field | Finding |
|---|---|
| Code licence | MIT (`mapillary/mapillary_sls`), copyright Meta Platforms — **covers the evaluation code, not the imagery** |
| Imagery licence | **Not established.** Mapillary's Terms place user imagery under CC BY-SA, but explicitly reserve releasing some content as CC BY-NC-SA |
| Editions | Mapillary's announcement states MSLS is "available both as a commercial edition and a research edition" — a separate research edition implies terms distinct from plain CC BY-SA |
| Primary sources unreadable | Dataset page is client-rendered and serves no terms; CVPR 2020 paper PDF returns HTTP 403; Mapillary's CC BY-SA help article returns HTTP 403 |
| Coordinates | Genuine **per-image** GPS (`raw.csv`: `lon, lat, ca, captured_at, pano`) |
| Access | `https://www.mapillary.com/dataset/places`, gated |

**Not ingested.** The specific terms could not be read from any primary source,
which is precisely the case to flag rather than proceed. Unblocking requires a
human to read and accept the edition terms — accepting terms is an owner action.
This is the same test that rejected GSV-Cities (CC BY-NC-ND).

### YFCC100M geotagged subset, ~48.4M geotagged — **licence CLEAR after filtering, access gated**

Per-item CC licences, not uniform (arXiv:1503.01817, Table 3):

| Licence | Photos | Videos | Usable? |
|---|---:|---:|---|
| CC-BY | 17,210,144 | 137,503 | yes |
| CC-BY-SA | 9,408,154 | 72,116 | yes |
| CC-BY-ND | 4,910,766 | 37,542 | no — NoDerivatives |
| CC-BY-NC | 12,674,885 | 102,288 | no — NonCommercial |
| CC-BY-NC-SA | 28,776,835 | 235,319 | no |
| CC-BY-NC-ND | 26,225,780 | 208,668 | no |

~67.7% is NonCommercial and ~31.1M items carry NoDerivatives. Because the
licence is recorded per item, the permissive subset (CC-BY + CC-BY-SA, 26.6M
items) is cleanly separable, and the adapter yields nothing else. Extracted
*features* are CC0; the restriction is on the source photographs. Coordinates
are genuine per-image (Flickr EXIF/user-placed); 48,366,323 photos are geotagged.

**Not ingested yet.** Metadata (`yfcc100m_dataset.bz2`, ~15 GB) and the
`multimedia-commons` S3 media both require an **AWS account**; the Yahoo
Webscope route requires a **Yahoo account**. Creating accounts is an owner
action. This is the strongest of the three once access is arranged.

### Google Landmarks v2, ~5M images — **FAILS on coordinates**

| Field | Finding |
|---|---|
| Annotations | CC BY 4.0 (Google) |
| Images | "publicly available on the web, and may have different licenses"; **train** subset restricted to "CC-BY licenses without the NonDerivs (ND) restriction"; index/test are CC-0 or Public Domain |
| **Coordinates** | **None.** `train.csv` is `id, url, landmark_id`; `train_attribution.csv` is `id, url, author, license, title`; `train_label_to_category.csv` maps `landmark_id → Wikimedia category`. No latitude/longitude field exists in any released metadata file. |

Licensing is actually the cleanest of the three. It fails the *technical*
requirement: coordinates are not per-image and are not shipped at all. Obtaining
them means joining `landmark_id → Wikimedia Commons category → Wikidata`, which
yields one approximate coordinate **per landmark**, shared by every image of it,
regardless of where the photographer stood. Not usable as per-image ground truth.

### OpenStreetView-5M (OSV-5M), 5.1M images — **CLEAR, primary training source**

Verified 2026-08-09 from primary sources.

| Field | Finding |
|---|---|
| Licence | **CC-BY-SA-4.0**, confirmed structurally via the Hugging Face Hub API — both `cardData.license` and the repo tag list return `cc-by-sa-4.0`, rather than inferred from a rendered page or "open-access" framing |
| Restrictions | **None blocking.** No NonCommercial, no NoDerivatives. Attribution + share-alike apply to redistribution |
| Gating | **Not gated** (`gated: False`, `private: False`) — no terms acceptance, no account, no approval. Unlike MSLS (terms unreadable) and YFCC100M (AWS/Yahoo account) |
| Intended use | **Training and evaluation**, checked explicitly against the eval-only concern. The paper benchmarks "image encoders, spatial representations, and **training strategies**", and enforces strict train/test separation so features can be evaluated "beyond mere memorization" |
| Coordinates | **Genuine per-image GPS** — camera positions, not subject centroids. This is what GLv2 and the Commons landmark set both lack |
| Coverage | 222 countries present in the indexed metadata (paper states 225 countries/territories) |
| Source | Mapillary (thumbnail URLs resolve to Meta CDN) |
| Citation | Astruc et al., *OpenStreetView-5M: The Many Roads to Global Visual Geolocation*, CVPR 2024 |

**Layout and the batch unit.** 98 train shards of ~50,000 images at ~2.35 GB each
(229.5 GB total), 5 test shards (8.8 GB), plus `train.csv` (2.78 GB) and
`test.csv` (111 MB). One shard is the natural batch unit.

**Shard membership is not contiguous in `train.csv`.** Measured: reading shard
00's index over HTTP range requests (4.4 MB, not 2.35 GB) gives its 50,000 ids;
the first 60 MB of the CSV (105,547 rows) contains only 1,111 of them — 2.2%.
So a shard's coordinates cannot be resolved from a CSV slice.
`finetune/datasets/osv5m_index.py` therefore streams the CSV **once** into a
269 MB parquet of `id/latitude/longitude/sequence/country` (4,894,684 rows,
1,391,425 sequences); every later batch joins locally and transfers no metadata.

That streaming is done in resumable HTTP Range chunks with backoff — a single
long stream failed at ~0.4 GB with `ConnectionResetError`, so a dropped
connection now costs one 8 MB chunk instead of the whole download.

`sequence` is used as the split unit: consecutive frames of one Mapillary
sequence are metres apart along the same road, so splitting by image would leak.

### Wikimedia Commons landmarks (local) — **CLEAR, used for the first batch**

Already held locally in `local-data/landmarks/catalog.sqlite3` via the official
MediaWiki API. 21,852 `ready` assets; per-image licence recorded in
`license_short`. A scan of all of them found **zero NonCommercial and zero
NoDerivatives** items (CC BY / CC BY-SA / CC0 / Public domain / GFDL). The
adapter filters explicitly anyway, since the catalog grows.

**Same coordinate caveat as GLv2, and it must be stated:** `camera_lat/lon`
equals `landmark_lat/lon` for all 21,852 rows, so these are **per-landmark**
positions, not camera positions. The error is bounded by how far a photographer
stands from a landmark rather than being unbounded, which is milder than GLv2's
Wikidata centroids — but it is an approximation, and any result trained on it
carries that caveat.

Its value is **location diversity**: 16,222 distinct locations at ≥100 m
spacing (74.2% retention from 21,849 images), against 184–730 for every previous
run.

## Evaluated but not automatically ingested

- **KartaView** provides open street imagery and geographic API queries
  (<https://kartaview.org/doc/>). It is a reasonable future gallery source, but
  its per-image terms and deduplication need to be recorded before ingestion.
- **Multimedia Commons/YFCC features** are available through the Registry of
  Open Data on AWS
  (<https://registry.opendata.aws/multimedia-commons/>). They are useful for
  research, but raw-image licensing varies and the whole corpus is too large for
  this repository's retrieval gallery.

## Evaluation rule

Training data scale is not evidence that a prediction is correct. All model
changes must be tested on held-out, geotagged images with no train/test spatial
leakage. `benchmark-harness/calibrate_trust.py` only enables automated emission
when a fixed agreement rule has enough accepted examples and its 95% Wilson
lower bound reaches the configured success target.
