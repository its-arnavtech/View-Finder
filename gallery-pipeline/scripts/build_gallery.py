"""Run validate -> transform -> load for an already-extracted dataset.

Breadth pulls stop after `extract`, because fine-tuning only needs raw images
plus coordinates. Making that imagery usable by the retrieval predictor requires
the remaining stages: validate (integrity + perceptual-hash dedup), transform
(quality filters + OpenCLIP embeddings) and load (FAISS index + metadata +
manifest). This runs them outside Airflow, in the same order the DAG does.

    python scripts/build_gallery.py --config config/tiling_config_new_regions.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

GALLERY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-validate", action="store_true",
                        help="reuse existing validation results")
    parser.add_argument("--skip-quality-check", action="store_true")
    args = parser.parse_args()

    os.chdir(GALLERY_ROOT)
    sys.path.insert(0, GALLERY_ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from src.config import load_config
    from src.extract.checkpoint_store import CheckpointStore
    from src.load.index_builder import run_load
    from src.quality_check.gallery_validator import run_quality_check
    from src.transform.runner import run_transform
    from src.validate.runner import run_validate

    config = load_config(args.config)
    store = CheckpointStore(config["paths"]["catalog_path"])
    tile_ids = store.tile_ids("completed")
    print(f"config      : {args.config}")
    print(f"data_dir    : {config['paths']['data_dir']}")
    print(f"completed tiles to process: {len(tile_ids)}")
    if not tile_ids:
        raise SystemExit("No completed tiles; run the extract stage first.")

    if not args.skip_validate:
        print("\n--- validate ---", flush=True)
        validation = run_validate(config, tile_ids)
        print(f"validate: {validation}")
        tile_ids = validation.get("tile_ids") or tile_ids

    print("\n--- transform (OpenCLIP embeddings) ---", flush=True)
    transformed = run_transform(config, tile_ids)
    print(f"transform: {transformed}")

    print("\n--- load (FAISS index) ---", flush=True)
    build = run_load(config)
    print(f"load: version={build.get('version')} dir={build.get('version_dir')}")

    if not args.skip_quality_check:
        print("\n--- quality check ---", flush=True)
        try:
            report = run_quality_check(config, build["version_dir"])
            print(f"quality_check: {report}")
        except Exception as exc:  # noqa: BLE001 - a failed gate should not hide the built gallery
            print(f"quality_check FAILED (gallery still built): {type(exc).__name__}: {exc}")

    print(f"\ngallery built at: {build.get('version_dir')}")


if __name__ == "__main__":
    main()
