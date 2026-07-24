"""Dense cells must subdivide and retry rather than failing the whole extraction."""

from pathlib import Path
from typing import Any, Iterator

import pytest
import requests

from src.extract.checkpoint_store import CheckpointStore
from src.extract.runner import run_extract, select_pending_tiles
from src.extract.tiling import Tile, subdivide_tile


def _density_error() -> requests.HTTPError:
    """An HTTPError shaped like a real Mapillary density rejection."""
    response = requests.Response()
    response.status_code = 500
    return requests.HTTPError(
        "500 Server Error: Internal Server Error for url: https://example.invalid"
        " | Please reduce the amount of data you're asking for, then retry your request",
        response=response,
    )


def _area_error() -> requests.HTTPError:
    """An HTTPError shaped like Mapillary's bbox-area-too-large rejection."""
    response = requests.Response()
    response.status_code = 500
    return requests.HTTPError(
        "500 Server Error: Internal Server Error for url: https://example.invalid"
        " | Bounding box area is too large. Maximum allowed area is 0.010 square"
        " degrees, but got 0.010 square degrees.",
        response=response,
    )


class DensityLimitedClient:
    """Rejects any bbox wider than `max_span`, like the live API does."""

    source = "mapillary"

    def __init__(self, max_span: float, per_tile: int = 5, error=_density_error):
        self.max_span = max_span
        self.per_tile = per_tile
        self.error = error
        self.rejected: list[str] = []
        self.served: list[str] = []

    def iter_images(self, tile: Tile, max_images: int | None = None) -> Iterator[dict[str, Any]]:
        span = tile.max_lon - tile.min_lon
        if span > self.max_span + 1e-9:
            self.rejected.append(tile.tile_id)
            raise self.error()
        self.served.append(tile.tile_id)
        for i in range(self.per_tile):
            yield {
                "image_id": f"{tile.tile_id}-{i}",
                "lat": tile.min_lat,
                "lon": tile.min_lon,
                "captured_at": None,
                "download_url": f"https://example.invalid/{tile.tile_id}-{i}.jpg",
            }

    def download_image(self, url: str, destination: str | Path) -> None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\xff\xd8\xff\xd9")


def _config(tmp_path: Path, **tiling: Any) -> dict[str, Any]:
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    base = {
        "mode": "regions",
        "max_tiles_per_run": 4,
        "regions": [{"name": "r", "bounds": [0.0, 0.0, 0.04, 0.04], "cell_size_degrees": 0.04}],
    }
    base.update(tiling)
    return {
        "paths": {"catalog_path": str(data / "catalog.db"), "raw_dir": str(data / "raw")},
        "tiling": base,
        "extract": {"max_images_per_tile": 50},
    }


def test_subdivide_tile_covers_same_bounds():
    parent = Tile("p", 0.0, 0.0, 0.04, 0.04, "r")
    children = subdivide_tile(parent)
    assert len(children) == 4
    assert min(c.min_lon for c in children) == pytest.approx(parent.min_lon)
    assert max(c.max_lon for c in children) == pytest.approx(parent.max_lon)
    assert min(c.min_lat for c in children) == pytest.approx(parent.min_lat)
    assert max(c.max_lat for c in children) == pytest.approx(parent.max_lat)
    # Children must not collide with the parent in the checkpoint table.
    assert parent.tile_id not in {c.tile_id for c in children}


def test_dense_tile_subdivides_instead_of_failing(tmp_path: Path):
    config = _config(tmp_path)
    # 0.04 is refused, 0.02 is accepted: exactly one subdivision level is required.
    client = DensityLimitedClient(max_span=0.02)
    result = run_extract(config, client=client)

    assert client.rejected, "the oversized cell should have been rejected once"
    assert len(result["subdivisions"]) == 1
    assert result["subdivisions"][0]["children"] == 4
    assert result["tiles_completed"] == 4, "all four children should complete"
    assert result["new_images"] == 20

    store = CheckpointStore(config["paths"]["catalog_path"])
    assert len(store.completed_tile_ids()) == 4


def test_area_too_large_error_also_subdivides(tmp_path: Path):
    """The 'bounding box area is too large' 500 must subdivide, not fail the run.

    This is the failure the live global run hit: a 0.1-degree cell equals
    Mapillary's exact 0.01 sq-degree bbox-area limit and is refused with a
    different message than the density rejection.
    """
    config = _config(tmp_path)
    client = DensityLimitedClient(max_span=0.02, error=_area_error)
    result = run_extract(config, client=client)

    assert client.rejected, "the oversized cell should have been rejected"
    assert len(result["subdivisions"]) == 1
    assert result["tiles_completed"] == 4
    store = CheckpointStore(config["paths"]["catalog_path"])
    assert len(store.completed_tile_ids()) == 4


def test_subdivision_recurses_until_accepted(tmp_path: Path):
    config = _config(tmp_path, max_subdivision_depth=3)
    # Needs two levels: 0.04 -> 0.02 -> 0.01.
    client = DensityLimitedClient(max_span=0.01)
    result = run_extract(config, client=client)

    assert len(result["subdivisions"]) == 5  # 1 parent + its 4 children all rejected
    assert result["tiles_completed"] == 16
    assert max(s["depth"] for s in result["subdivisions"]) == 1


def test_subdivision_gives_up_past_max_depth(tmp_path: Path):
    config = _config(tmp_path, max_subdivision_depth=0)
    client = DensityLimitedClient(max_span=0.02)
    with pytest.raises(requests.HTTPError):
        run_extract(config, client=client)


def test_interleave_regions_samples_every_region(tmp_path: Path):
    regions = [
        {"name": "alpha", "bounds": [0.0, 0.0, 0.1, 0.01], "cell_size_degrees": 0.01},
        {"name": "beta", "bounds": [10.0, 10.0, 10.1, 10.01], "cell_size_degrees": 0.01},
    ]
    config = _config(tmp_path, regions=regions, max_tiles_per_run=4, interleave_regions=True)
    store = CheckpointStore(config["paths"]["catalog_path"])
    picked = select_pending_tiles(config, store)

    assert len(picked) == 4
    assert {t.region for t in picked} == {"alpha", "beta"}, "both regions must be sampled"

    # Default (no interleaving) preserves the original config-order behaviour.
    config["tiling"]["interleave_regions"] = False
    picked_ordered = select_pending_tiles(config, store)
    assert {t.region for t in picked_ordered} == {"alpha"}
