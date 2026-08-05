# Local evidence data pipeline

This pipeline builds the default, API-key-free evidence stores used by
View Finder:

- a global GeoNames SQLite/FTS gazetteer with alternate spellings;
- optional regional OpenStreetMap names inserted into the same gazetteer;
- an attributed Wikimedia Commons thumbnail catalog;
- an OpenCLIP/FAISS landmark index plus perceptual hashes.

The default configuration targets 30,000 US features and 20,000 major worldwide
features, downloads at most 250,000 Commons thumbnails, and stops publishing
images at 40 GiB. Individual Commons licenses, source pages, artists, and credit
fields are retained in `metadata.parquet`, along with each Commons structured
data entity ID and any linked Wikidata ID exposed by the file metadata.
The bundled regional OSM configuration enriches Nebraska and Cyprus with
business names, parks, monuments, highways, and route references; add more
Geofabrik extract URLs in a copied config as needed.

## Build

From the repository root:

```powershell
python -m pip install -r viewfinder/requirements-local-evidence.txt
python local-data-pipeline/build_local_data.py all
```

Each network stage reuses completed files or database rows. To run and inspect
stages separately:

```powershell
python local-data-pipeline/build_local_data.py geonames-download
python local-data-pipeline/build_local_data.py geonames-build
python local-data-pipeline/build_local_data.py landmarks-plan
python local-data-pipeline/build_local_data.py landmarks-discover
python local-data-pipeline/build_local_data.py landmarks-download
python local-data-pipeline/build_local_data.py landmarks-index
python local-data-pipeline/build_local_data.py osm-download
python local-data-pipeline/build_local_data.py osm-ingest
python local-data-pipeline/build_local_data.py status
```

Use `--config` with a copy of `config.default.json` to change the storage
budget, target counts, model, or OSM regions. Runtime data is intentionally
excluded from Git.

## Trust behavior

The gallery is retrieval evidence, not a universal classifier. The runtime
automatically emits only for EXIF, a near-exact local reference match, or
agreement between independently derived OCR/gazetteer and visual-landmark
coordinates. Otherwise it returns `REVIEW` or `ABSTAIN`. PLONK is kept only as a
geographic prior and cannot authorize an output by itself.
