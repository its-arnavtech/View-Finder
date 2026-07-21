"""Resilient Mapillary Graph API client."""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import requests

from src.extract.tiling import Tile


LOGGER = logging.getLogger(__name__)
GRAPH_IMAGES_URL = "https://graph.mapillary.com/images"


class MapillaryClient:
    def __init__(
        self,
        access_token: str | None = None,
        *,
        page_size: int = 100,
        request_timeout: float = 30,
        min_request_interval: float = 0.25,
        max_attempts: int = 6,
        backoff_base: float = 1.0,
        session: requests.Session | None = None,
    ):
        self.access_token = access_token or os.getenv("MAPILLARY_ACCESS_TOKEN")
        if not self.access_token:
            raise ValueError("MAPILLARY_ACCESS_TOKEN is required for live extraction")
        self.page_size = min(max(page_size, 1), 2000)
        self.request_timeout = request_timeout
        self.min_request_interval = min_request_interval
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.session = session or requests.Session()
        self._last_request_at = 0.0

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

    def _request(self, url: str, **kwargs: Any) -> requests.Response:
        for attempt in range(1, self.max_attempts + 1):
            self._wait_for_rate_limit()
            try:
                response = self.session.get(url, timeout=self.request_timeout, **kwargs)
                self._last_request_at = time.monotonic()
            except requests.RequestException:
                if attempt == self.max_attempts:
                    raise
                delay = self.backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                LOGGER.warning("Mapillary request failed; retrying in %.2fs (attempt %d/%d)", delay, attempt, self.max_attempts)
                time.sleep(delay)
                continue

            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response
            if attempt == self.max_attempts:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else self.backoff_base * (2 ** (attempt - 1))
            delay += random.uniform(0, 0.25)
            LOGGER.warning("Mapillary returned HTTP %d; retrying in %.2fs (attempt %d/%d)", response.status_code, delay, attempt, self.max_attempts)
            time.sleep(delay)
        raise RuntimeError("Unreachable retry state")

    def iter_images(self, tile: Tile, max_images: int | None = None) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] | None = {
            "access_token": self.access_token,
            "fields": "id,geometry,thumb_original_url,captured_at",
            "bbox": ",".join(str(value) for value in tile.bbox),
            "limit": self.page_size,
        }
        url: str | None = GRAPH_IMAGES_URL
        yielded = 0
        while url and (max_images is None or yielded < max_images):
            payload = self._request(url, params=params).json()
            params = None  # paging.next already contains the cursor and token
            for item in payload.get("data", []):
                geometry = item.get("geometry") or {}
                coordinates = geometry.get("coordinates") or []
                if len(coordinates) < 2 or not item.get("thumb_original_url"):
                    LOGGER.warning("Skipping Mapillary image %s with incomplete metadata", item.get("id"))
                    continue
                yield {
                    "image_id": str(item["id"]),
                    "lon": float(coordinates[0]),
                    "lat": float(coordinates[1]),
                    "captured_at": item.get("captured_at"),
                    "download_url": item["thumb_original_url"],
                }
                yielded += 1
                if max_images is not None and yielded >= max_images:
                    break
            url = (payload.get("paging") or {}).get("next")

    def download_image(self, url: str, destination: str | Path) -> None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        response = self._request(url, stream=True)
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if chunk:
                    handle.write(chunk)
        temporary.replace(destination)

