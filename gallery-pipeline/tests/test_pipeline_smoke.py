from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.extract.runner import run_extract
from src.load.index_builder import run_load
from src.quality_check.gallery_validator import run_quality_check
from src.transform.runner import run_transform
from src.validate.runner import run_validate


class FakeMapillaryClient:
    def __init__(self):
        self.iteration_count = 0

    def iter_images(self, tile, max_images=None):
        self.iteration_count += 1
        for number in range(3):
            yield {
                "image_id": f"image-{number}",
                "lat": tile.min_lat + 0.01,
                "lon": tile.min_lon + 0.01,
                "captured_at": "2025-01-01T00:00:00Z",
                "download_url": f"fake://image-{number}",
            }

    def download_image(self, url, destination):
        number = int(url.rsplit("-", 1)[1])
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (128, 128), (30 + number * 50, 90, 170 - number * 30))
        draw = ImageDraw.Draw(image)
        if number == 0:
            draw.line((0, 0, 127, 127), fill="white", width=10)
        elif number == 1:
            draw.ellipse((20, 15, 110, 100), outline="yellow", width=12)
        else:
            for x in range(0, 128, 16):
                draw.rectangle((x, 0, x + 7, 127), fill="black")
        image.save(destination, format="JPEG", quality=95)


class FakeEmbeddingGenerator:
    full_model_name = "fake:test:1"
    indoor_probability_threshold = 0.65

    def generate(self, image_paths):
        vectors = []
        for index, _ in enumerate(image_paths):
            vectors.append([1.0, float(index + 1), 0.5, -0.25])
        return np.asarray(vectors, dtype=np.float32), np.full(len(vectors), 0.1, dtype=np.float32)


def make_config(tmp_path: Path):
    data_dir = tmp_path / "data"
    return {
        "paths": {
            "data_dir": str(data_dir),
            "catalog_path": str(data_dir / "catalog.db"),
            "raw_dir": str(data_dir / "raw"),
            "transformed_dir": str(data_dir / "transformed"),
            "gallery_dir": str(data_dir / "gallery"),
        },
        "tiling": {
            "mode": "regions",
            "max_tiles_per_run": 1,
            "cell_size_degrees": 1.0,
            "regions": [{"name": "smoke", "bounds": [-88, 41, -87, 42]}],
        },
        "extract": {"max_images_per_tile": 10},
        "validate": {"perceptual_hash_distance": 2},
        "transform": {
            "max_images_per_tile": 10,
            "blur_variance_threshold": -1.0,
            "exclude_likely_indoor": True,
            "embedding_model": "fake",
            "embedding_pretrained": "test",
            "embedding_batch_size_gpu": 8,
            "embedding_batch_size_cpu": 4,
        },
        "quality_check": {
            "minimum_total_images": 1,
            "expected_regions": [],
            "minimum_region_fraction": 0.01,
        },
    }


def test_end_to_end_pipeline_and_resume(tmp_path: Path):
    config = make_config(tmp_path)
    client = FakeMapillaryClient()
    extraction = run_extract(config, client=client)
    assert extraction["tiles_completed"] == 1
    assert extraction["new_images"] == 3

    validation = run_validate(config, extraction["tile_ids"])
    assert validation["passed"] >= 2
    transformation = run_transform(
        config, validation["tile_ids"], generator=FakeEmbeddingGenerator()
    )
    assert transformation["retained"] == validation["passed"]

    first_build = run_load(config)
    first_report = run_quality_check(config, first_build["version_dir"])
    assert first_build["version"] == 1
    assert first_report["passed"]
    assert (Path(first_build["version_dir"]) / "quality_report.txt").exists()

    second_build = run_load(config)
    assert second_build["version"] == 2
    assert Path(first_build["version_dir"]).exists()

    resumed = run_extract(config, client=client)
    assert resumed["tiles_completed"] == 0
    assert client.iteration_count == 1

