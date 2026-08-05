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
