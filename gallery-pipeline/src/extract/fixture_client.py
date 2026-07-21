"""Deterministic credential-free client used only for Airflow smoke testing."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from src.extract.tiling import Tile


class FixtureMapillaryClient:
    """Implements the Mapillary client interface without network access.

    Activation requires the explicit ``GALLERY_EXTRACT_MODE=fixture`` environment
    variable; live extraction remains the default and always requires a real token.
    """

    source = "mapillary_fixture"

    def iter_images(self, tile: Tile, max_images: int | None = None):
        count = min(3, max_images) if max_images is not None else 3
        prefix = hashlib.sha1(tile.tile_id.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
        for number in range(count):
            yield {
                "image_id": f"fixture-{prefix}-{number}",
                "lat": tile.min_lat + (tile.max_lat - tile.min_lat) * (number + 1) / (count + 1),
                "lon": tile.min_lon + (tile.max_lon - tile.min_lon) * (number + 1) / (count + 1),
                "captured_at": "2025-01-01T00:00:00Z",
                "download_url": f"fixture://{number}",
            }

    def download_image(self, url: str, destination: str | Path) -> None:
        number = int(url.rsplit("/", 1)[1])
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (256, 160), (35 + 50 * number, 100, 190 - 30 * number))
        draw = ImageDraw.Draw(image)
        if number == 0:
            draw.line((0, 0, 255, 159), fill="white", width=12)
        elif number == 1:
            draw.ellipse((40, 20, 200, 140), outline="yellow", width=14)
        else:
            for x in range(0, 256, 24):
                draw.rectangle((x, 0, x + 9, 159), fill="black")
        image.save(destination, "JPEG", quality=95)

