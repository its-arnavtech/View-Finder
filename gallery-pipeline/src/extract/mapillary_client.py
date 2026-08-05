"""Resilient Mapillary Graph API client."""

from __future__ import annotations

import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterator

import requests

from src.extract.tiling import Tile


LOGGER = logging.getLogger(__name__)
GRAPH_IMAGES_URL = "https://graph.mapillary.com/images"

# Mapillary's own `paging.next` cursors carry ?access_token=... even when the request
# was authenticated by header, so any URL echoed into an exception or log line must be
# scrubbed. Matches both raw and percent-encoded pipe separators.
_TOKEN_QUERY = re.compile(r"(access_token=)[^&\s]+", re.IGNORECASE)

# Statuses worth another attempt: rate limiting and the 5xx family. Mapillary
# also returns 500 for "too much data in this bbox", which the runner turns into
# a subdivision rather than a failure once the retries are exhausted.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def redact(text: str) -> str:
    """Replace any access token embedded in a URL or message with a placeholder."""
    return _TOKEN_QUERY.sub(r"\1<redacted>", str(text))


class MapillaryClient:
    def __init__(
        self,
        access_token: str | None = None,
        *,
        page_size: int = 100,
        request_timeout: float = 30,
        min_request_interval: float = 0.25,
        max_attempts: int = 6,
        max_transport_attempts: int | None = None,
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
        # Transport failures get their own budget so a burst of HTTP 500s cannot
        # leave a transient connection reset with no attempts left.
        self.max_transport_attempts = (
            max_attempts if max_transport_attempts is None else max_transport_attempts
        )
        self.backoff_base = backoff_base
        self.session = session or requests.Session()
        # Authenticate with a header rather than an ?access_token= query parameter.
        # requests embeds the full URL in HTTPError messages, so a query-string token
        # leaks verbatim into tracebacks, Airflow task logs and any error tracker the
        # moment the API returns a non-2xx status.
        self.session.headers.update({"Authorization": f"OAuth {self.access_token}"})
        self._last_request_at = 0.0

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        """raise_for_status, but with any access token scrubbed from the message.

        requests builds its HTTPError message from the full request URL, which for
        paging cursors contains the caller's token.
        """
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            # Surface the API's own error text: Mapillary returns 500 both for real
            # outages and for "too much data" density rejections, and only the body
            # distinguishes them.
            detail = ""
            try:
                error = response.json().get("error", {})
                message = error.get("message")
                if message:
                    detail = f" | {message}"
            except ValueError:
                pass
            raise requests.HTTPError(
                redact(f"{exc}{detail}"), response=exc.response, request=exc.request
            ) from None

    def _request(self, url: str, **kwargs: Any) -> requests.Response:
        """GET with retries, budgeting transport and status failures separately.

        Transport failures (``ConnectionError``, ``Timeout``,
        ``ChunkedEncodingError`` -- all subclasses of ``RequestException``) and
        retryable HTTP statuses are both retried with exponential backoff, but
        they draw from *independent* budgets.

        Sharing one counter looked reasonable and was not: a dense tile that
        spends four attempts on Mapillary's 500 "reduce the amount of data"
        rejection had nothing left when the fifth attempt hit a transient
        connection reset, so the reset re-raised immediately and aborted a
        500-tile pull. The two failure modes are unrelated -- a server-side
        density rejection says nothing about link stability -- so exhausting one
        budget must not consume the other.

        Both counters increase monotonically and neither resets, so the loop is
        bounded by ``max_attempts + max_transport_attempts`` iterations.
        """
        status_attempt = 0
        transport_attempt = 0

        while True:
            self._wait_for_rate_limit()
            try:
                response = self.session.get(url, timeout=self.request_timeout, **kwargs)
                self._last_request_at = time.monotonic()
            except requests.RequestException as exc:
                transport_attempt += 1
                if transport_attempt >= self.max_transport_attempts:
                    raise type(exc)(redact(exc)) from None
                delay = self.backoff_base * (2 ** (transport_attempt - 1)) + random.uniform(0, 0.25)
                LOGGER.warning(
                    "Mapillary request failed (%s); retrying in %.2fs (transport attempt %d/%d)",
                    type(exc).__name__, delay, transport_attempt, self.max_transport_attempts,
                )
                time.sleep(delay)
                continue

            if response.status_code not in RETRYABLE_STATUS_CODES:
                self._raise_for_status(response)
                return response

            status_attempt += 1
            if status_attempt >= self.max_attempts:
                self._raise_for_status(response)
            retry_after = response.headers.get("Retry-After")
            delay = (
                float(retry_after)
                if retry_after and retry_after.isdigit()
                else self.backoff_base * (2 ** (status_attempt - 1))
            )
            delay += random.uniform(0, 0.25)
            LOGGER.warning(
                "Mapillary returned HTTP %d; retrying in %.2fs (status attempt %d/%d)",
                response.status_code, delay, status_attempt, self.max_attempts,
            )
            time.sleep(delay)

    def iter_images(self, tile: Tile, max_images: int | None = None) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] | None = {
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
        # thumb_original_url is a pre-signed CDN link that needs no credentials of ours;
        # suppress the session's Authorization header so the token is never sent to a
        # third-party host.
        response = self._request(url, stream=True, headers={"Authorization": None})
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if chunk:
                    handle.write(chunk)
        temporary.replace(destination)

