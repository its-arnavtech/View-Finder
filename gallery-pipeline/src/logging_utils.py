"""Consistent logging configuration shared by CLI code and Airflow tasks."""

from __future__ import annotations

import logging
import os


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("GALLERY_LOG_LEVEL", "INFO").upper(),
        format=LOG_FORMAT,
        force=False,
    )

