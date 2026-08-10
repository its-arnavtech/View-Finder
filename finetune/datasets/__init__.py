"""Dataset ingestion adapters for large-scale location-encoder training.

The training loop does not know where images come from. Each adapter turns some
source into the one frame shape the loop consumes -- ``image_id, lat, lon,
tile_id, region, dataset, image_path`` -- and declares its licence status.

Licence status is enforced in code, not left to a README. An adapter whose
terms are unclear or whose access is gated raises on ``load()`` with the reason
and what would unblock it, so a training run can never quietly ingest data
nobody verified. See :mod:`finetune.datasets.base` for the statuses and
DATASETS.md for the findings behind each one.
"""

from .base import (
    Availability,
    DatasetAdapter,
    LicenceStatus,
    REQUIRED_COLUMNS,
    validate_frame,
)
from .commons_landmarks import CommonsLandmarksAdapter
from .mapillary_catalog import MapillaryCatalogAdapter
from .msls import MSLSAdapter
from .osv5m import OSV5MAdapter
from .yfcc100m import YFCC100MAdapter

ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "commons-landmarks": CommonsLandmarksAdapter,
    "mapillary-catalog": MapillaryCatalogAdapter,
    "msls": MSLSAdapter,
    "osv5m": OSV5MAdapter,
    "yfcc100m": YFCC100MAdapter,
}


def get_adapter(name: str, **kwargs) -> DatasetAdapter:
    """Instantiate an adapter by name."""
    if name not in ADAPTERS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {', '.join(sorted(ADAPTERS))}")
    return ADAPTERS[name](**kwargs)


__all__ = [
    "ADAPTERS",
    "Availability",
    "CommonsLandmarksAdapter",
    "DatasetAdapter",
    "LicenceStatus",
    "MSLSAdapter",
    "OSV5MAdapter",
    "MapillaryCatalogAdapter",
    "REQUIRED_COLUMNS",
    "YFCC100MAdapter",
    "get_adapter",
    "validate_frame",
]
