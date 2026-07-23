"""The per-tile image cap must hold across interrupted and resumed extractions."""

from pathlib import Path
from typing import Any, Iterator

import pytest

from src.extract.checkpoint_store import CheckpointStore
from src.extract.runner import run_extract
from src.extract.tiling import Tile


CAP = 25


class UnstablePagingClient:
    """Mapillary stand-in whose result ordering changes between runs.

    This is the behaviour that caused the original overrun: a resumed tile
    re-queries from page 1 and gets a different sample, so images the store has
    never seen appear before the ones it already holds.
    """

    source = "mapillary"

    def __init__(self, ids: list[str]):
        self.ids = ids
        self.yielded = 0
        self.downloads = 0

    def iter_images(self, tile: Tile, max_images: int | None = None) -> Iterator[dict[str, Any]]:
        for image_id in self.ids:
            if max_images is not None and self.yielded >= max_images:
                return
            self.yielded += 1
            yield {
                "image_id": image_id,
                "lat": 41.87,
                "lon": -87.65,
                "captured_at": None,
                "download_url": f"https://example.invalid/{image_id}.jpg",
            }

    def download_image(self, url: str, destination: str | Path) -> None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG-ish stub
        self.downloads += 1


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    return {
        "paths": {
            "catalog_path": str(data / "catalog.db"),
            "raw_dir": str(data / "raw"),
        },
        "tiling": {
            "mode": "regions",
            "max_tiles_per_run": 1,
            "regions": [
                {"name": "t", "bounds": [-87.65, 41.87, -87.64, 41.88], "cell_size_degrees": 0.01}
            ],
        },
        "extract": {"max_images_per_tile": CAP, "page_size": 10},
    }


def _tile_id(config: dict[str, Any]) -> str:
    store = CheckpointStore(config["paths"]["catalog_path"])
    return store.tile_ids()[0]


def test_resumed_tile_never_exceeds_cap(config: dict[str, Any]):
    """An interrupted tile plus a resume with fresh images stays at the cap."""
    # First run is interrupted after 9 images: simulate by capping the client's supply.
    partial = UnstablePagingClient([f"old-{i}" for i in range(9)])
    run_extract(config, client=partial)

    store = CheckpointStore(config["paths"]["catalog_path"])
    tile_id = _tile_id(config)
    assert store.image_count_for_tile(tile_id) == 9

    # Force the tile back to a resumable state, as an interrupt would leave it.
    store.mark_tile(tile_id, "in_progress")

    # The resume sees an entirely different page-1 sample: 25 unseen ids first.
    resumed = UnstablePagingClient([f"new-{i}" for i in range(25)] + [f"old-{i}" for i in range(9)])
    run_extract(config, client=resumed)

    total = store.image_count_for_tile(tile_id)
    assert total == CAP, f"cap breached: tile holds {total} images, cap is {CAP}"
    # Only the 16 images needed to reach the cap should have been downloaded.
    assert resumed.downloads == CAP - 9


def test_tile_already_at_cap_makes_no_api_calls(config: dict[str, Any]):
    """A tile that already satisfies the cap completes without touching the API."""
    first = UnstablePagingClient([f"img-{i}" for i in range(CAP)])
    run_extract(config, client=first)

    store = CheckpointStore(config["paths"]["catalog_path"])
    tile_id = _tile_id(config)
    assert store.image_count_for_tile(tile_id) == CAP

    store.mark_tile(tile_id, "in_progress")
    second = UnstablePagingClient([f"extra-{i}" for i in range(50)])
    result = run_extract(config, client=second)

    assert second.yielded == 0, "API was queried for a tile already at its cap"
    assert second.downloads == 0
    assert store.image_count_for_tile(tile_id) == CAP
    assert tile_id in result["tile_ids"]


def test_extraction_stops_early_once_cap_reached(config: dict[str, Any]):
    """Extraction stops consuming pages as soon as the cap is met."""
    client = UnstablePagingClient([f"img-{i}" for i in range(500)])
    run_extract(config, client=client)

    store = CheckpointStore(config["paths"]["catalog_path"])
    assert store.image_count_for_tile(_tile_id(config)) == CAP
    # It must not have walked the whole 500-image supply to collect 25.
    assert client.yielded == CAP
