from pathlib import Path

from src.extract.checkpoint_store import CheckpointStore
from src.extract.tiling import generate_tiles


def test_tiling_covers_bounds_without_overrun():
    tiles = list(generate_tiles([-1.0, -1.0, 1.2, 1.2], 1.0, "test region"))
    assert len(tiles) == 9
    assert len({tile.tile_id for tile in tiles}) == 9
    assert min(tile.min_lon for tile in tiles) == -1.0
    assert min(tile.min_lat for tile in tiles) == -1.0
    assert max(tile.max_lon for tile in tiles) == 1.2
    assert max(tile.max_lat for tile in tiles) == 1.2


def test_checkpoint_and_image_insert_are_idempotent(tmp_path: Path):
    store = CheckpointStore(tmp_path / "catalog.db")
    store.mark_tile("tile-1", "in_progress")
    assert not store.is_complete("tile-1")
    metadata = {
        "image_id": "123",
        "source": "mapillary",
        "lat": 41.0,
        "lon": -87.0,
        "tile_id": "tile-1",
        "raw_file_path": str(tmp_path / "123.jpg"),
    }
    assert store.add_image(metadata)
    assert not store.add_image(metadata)
    store.mark_tile("tile-1", "completed", images_found=1)
    assert store.is_complete("tile-1")
    assert store.completed_tile_ids() == {"tile-1"}

