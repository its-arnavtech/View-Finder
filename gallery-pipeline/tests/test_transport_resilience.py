"""A long pull must survive transient transport failures.

Two independent defects let a single connection reset abort a 500-tile run:

1. ``MapillaryClient._request`` retried transport errors and retryable HTTP
   statuses out of one shared attempt budget. A dense tile that spent four
   attempts on Mapillary's HTTP 500 density rejection had nothing left when the
   fifth attempt hit a connection reset, so the reset re-raised immediately.

2. ``run_extract`` re-raised after marking a tile failed, discarding every
   remaining tile even though completed ones were already checkpointed.

These tests pin both fixes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest
import requests

from src.extract.checkpoint_store import CheckpointStore
from src.extract.mapillary_client import MapillaryClient
from src.extract.runner import run_extract
from src.extract.tiling import Tile


# --- client-level: independent retry budgets -----------------------------------


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._payload = payload or {"data": []}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


class ScriptedSession:
    """Replays a scripted sequence of responses/exceptions for each GET."""

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls = 0

    def get(self, url: str, **kwargs: Any):
        self.calls += 1
        if not self.script:
            raise AssertionError("ScriptedSession ran out of scripted responses")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    headers: dict[str, str] = {}


def _client(script: list[Any], **kwargs: Any) -> MapillaryClient:
    session = ScriptedSession(script)
    session.headers = {}
    return MapillaryClient(
        access_token="test-token",
        session=session,  # type: ignore[arg-type]
        backoff_base=0.0,          # keep the tests fast
        min_request_interval=0.0,
        **kwargs,
    )


def test_connection_error_is_retried_and_recovers() -> None:
    """A reset mid-request retries rather than propagating."""
    client = _client(
        [
            requests.ConnectionError("Connection aborted."),
            requests.ConnectionError("Connection aborted."),
            FakeResponse(200, {"data": []}),
        ],
        max_attempts=5,
    )
    response = client._request("https://graph.mapillary.invalid/images")
    assert response.status_code == 200


def test_http_500s_do_not_consume_the_transport_budget() -> None:
    """The original crash: 500s exhaust the budget, then a reset kills the run.

    Four HTTP 500s followed by a connection reset must still recover, because the
    two failure modes draw from independent budgets.
    """
    client = _client(
        [
            FakeResponse(500),
            FakeResponse(500),
            FakeResponse(500),
            FakeResponse(500),
            requests.ConnectionError("Connection aborted."),
            FakeResponse(200, {"data": []}),
        ],
        max_attempts=5,
        max_transport_attempts=5,
    )
    response = client._request("https://graph.mapillary.invalid/images")
    assert response.status_code == 200


def test_transport_budget_is_finite() -> None:
    """Retries must not loop forever when the link is genuinely down."""
    client = _client([requests.ConnectionError("down")] * 10, max_transport_attempts=3)
    with pytest.raises(requests.ConnectionError):
        client._request("https://graph.mapillary.invalid/images")


def test_exhausted_status_budget_still_raises() -> None:
    client = _client([FakeResponse(500)] * 10, max_attempts=3)
    with pytest.raises(requests.HTTPError):
        client._request("https://graph.mapillary.invalid/images")


def test_token_is_redacted_from_transport_errors() -> None:
    client = _client(
        [requests.ConnectionError("failed for url: https://x.invalid/?access_token=SECRET123")],
        max_transport_attempts=1,
    )
    with pytest.raises(requests.ConnectionError) as excinfo:
        client._request("https://graph.mapillary.invalid/images")
    assert "SECRET123" not in str(excinfo.value)
    assert "<redacted>" in str(excinfo.value)


# --- runner-level: one bad tile must not abort the run -------------------------


class FlakyTileClient:
    """Raises a transport error on selected tiles, serves images for the rest."""

    source = "mapillary"

    def __init__(self, failing_tile_indices: set[int], images_per_tile: int = 2):
        self.failing = failing_tile_indices
        self.images_per_tile = images_per_tile
        self.seen_tiles: list[str] = []

    def iter_images(self, tile: Tile, max_images: int | None = None) -> Iterator[dict[str, Any]]:
        index = len(self.seen_tiles)
        self.seen_tiles.append(tile.tile_id)
        if index in self.failing:
            raise requests.ConnectionError(
                "('Connection aborted.', ConnectionResetError(10054, 'forcibly closed'))"
            )
        for i in range(self.images_per_tile):
            yield {
                "image_id": f"{tile.tile_id}-{i}",
                "lat": 41.87,
                "lon": -87.65,
                "captured_at": None,
                "download_url": f"https://example.invalid/{tile.tile_id}-{i}.jpg",
            }

    def download_image(self, url: str, destination: str | Path) -> None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\xff\xd8\xff\xd9")


@pytest.fixture
def multi_tile_config(tmp_path: Path) -> dict[str, Any]:
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    return {
        "paths": {
            "catalog_path": str(data / "catalog.db"),
            "raw_dir": str(data / "raw"),
        },
        "tiling": {
            "mode": "regions",
            "max_tiles_per_run": 10,
            "regions": [
                {"name": "t", "bounds": [-87.70, 41.87, -87.65, 41.88], "cell_size_degrees": 0.01}
            ],
        },
        "extract": {"max_images_per_tile": 5, "page_size": 10},
    }


def test_connection_error_mid_pull_does_not_abort_the_run(multi_tile_config) -> None:
    """The headline regression: a reset on one tile must not discard the others."""
    client = FlakyTileClient(failing_tile_indices={1})
    result = run_extract(multi_tile_config, client=client)

    # The run continued past the failure instead of raising.
    assert result["tiles_failed"] == 1
    assert result["tiles_completed"] >= 2, result
    assert not result["aborted_early"]

    # Work from the surviving tiles is durable: two images per completed tile.
    assert result["new_images"] == 2 * result["tiles_completed"]
    store = CheckpointStore(multi_tile_config["paths"]["catalog_path"])
    assert len(store.completed_tile_ids()) == result["tiles_completed"]


def test_systemic_failure_still_aborts(multi_tile_config) -> None:
    """Back-to-back failures are an outage, not a bad tile: stop rather than grind."""
    multi_tile_config["extract"]["max_consecutive_tile_failures"] = 3
    client = FlakyTileClient(failing_tile_indices=set(range(50)))
    result = run_extract(multi_tile_config, client=client)

    assert result["aborted_early"]
    assert result["tiles_completed"] == 0
    assert result["tiles_failed"] == 3


class DenseTileClient:
    """Always rejects with Mapillary's 'too much data' 500, like an undividable metro."""

    source = "mapillary"

    def __init__(self) -> None:
        self.calls = 0

    def iter_images(self, tile: Tile, max_images: int | None = None) -> Iterator[dict[str, Any]]:
        self.calls += 1
        response = FakeResponse(500)
        raise requests.HTTPError(
            "500 Server Error | Please reduce the amount of data you're asking for",
            response=response,
        )
        yield  # pragma: no cover - unreachable, keeps this a generator

    def download_image(self, url: str, destination: str | Path) -> None:  # pragma: no cover
        raise AssertionError("should never download")


def test_undividable_dense_tile_fails_fast_by_default(multi_tile_config) -> None:
    """Gallery builds must see a density rejection they cannot subdivide away."""
    multi_tile_config["tiling"]["max_subdivision_depth"] = 0
    with pytest.raises(requests.HTTPError):
        run_extract(multi_tile_config, client=DenseTileClient())


def test_breadth_pulls_can_skip_undividable_tiles(multi_tile_config) -> None:
    """Breadth pulls opt into skipping: many places matters more than any one."""
    multi_tile_config["tiling"]["max_subdivision_depth"] = 0
    multi_tile_config["extract"]["skip_unsubdividable_tiles"] = True
    multi_tile_config["extract"]["max_consecutive_tile_failures"] = 99

    result = run_extract(multi_tile_config, client=DenseTileClient())

    assert result["tiles_failed"] > 0
    assert not result["aborted_early"]


def test_failure_streak_resets_after_a_success(multi_tile_config) -> None:
    """Isolated failures separated by successes must not accumulate to an abort."""
    multi_tile_config["extract"]["max_consecutive_tile_failures"] = 3
    client = FlakyTileClient(failing_tile_indices={0, 2, 4})
    result = run_extract(multi_tile_config, client=client)

    assert not result["aborted_early"]
    assert result["tiles_failed"] == 3
    assert result["tiles_completed"] >= 2
