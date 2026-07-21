"""SQLite checkpoint and metadata catalog.

All mutations are committed in small transactions. A tile is only marked complete after
its final API page and downloads finish, so an interrupted run safely retries that tile.
Image inserts are idempotent, making such retries harmless.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointStore:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tiles (
                    tile_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('pending','in_progress','completed','failed')),
                    last_updated TEXT NOT NULL,
                    images_found INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    captured_at TEXT,
                    download_timestamp TEXT NOT NULL,
                    tile_id TEXT NOT NULL,
                    raw_file_path TEXT NOT NULL,
                    validation_status TEXT NOT NULL DEFAULT 'pending',
                    validation_reason TEXT,
                    perceptual_hash TEXT,
                    blur_score REAL,
                    likely_indoor INTEGER,
                    transform_status TEXT NOT NULL DEFAULT 'pending'
                );
                CREATE INDEX IF NOT EXISTS idx_images_tile ON images(tile_id);
                CREATE INDEX IF NOT EXISTS idx_images_validation ON images(validation_status);

                CREATE TABLE IF NOT EXISTS stage_results (
                    tile_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    run_timestamp TEXT NOT NULL,
                    passed_count INTEGER NOT NULL,
                    failed_count INTEGER NOT NULL,
                    reasons_json TEXT NOT NULL,
                    PRIMARY KEY (tile_id, stage)
                );

                CREATE TABLE IF NOT EXISTS transformed_assets (
                    tile_id TEXT PRIMARY KEY,
                    parquet_path TEXT NOT NULL,
                    image_count INTEGER NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def is_complete(self, tile_id: str) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT status FROM tiles WHERE tile_id = ?", (tile_id,)
            ).fetchone()
            return bool(row and row["status"] == "completed")

    def completed_tile_ids(self) -> set[str]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT tile_id FROM tiles WHERE status = 'completed'"
            ).fetchall()
            return {row["tile_id"] for row in rows}

    def tile_ids(self, status: str | None = None) -> list[str]:
        query = "SELECT tile_id FROM tiles"
        parameters: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY tile_id"
        with self.connection() as connection:
            return [row["tile_id"] for row in connection.execute(query, parameters).fetchall()]

    def mark_tile(self, tile_id: str, status: str, images_found: int = 0, error: str | None = None) -> None:
        if status not in {"pending", "in_progress", "completed", "failed"}:
            raise ValueError(f"Unsupported tile status: {status}")
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO tiles(tile_id, status, last_updated, images_found, error_message)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tile_id) DO UPDATE SET
                    status = excluded.status,
                    last_updated = excluded.last_updated,
                    images_found = excluded.images_found,
                    error_message = excluded.error_message
                """,
                (tile_id, status, utc_now(), images_found, error),
            )

    def image_exists(self, image_id: str) -> bool:
        with self.connection() as connection:
            return connection.execute(
                "SELECT 1 FROM images WHERE image_id = ?", (image_id,)
            ).fetchone() is not None

    def add_image(self, metadata: dict[str, Any]) -> bool:
        """Insert metadata once; return True when this call created the row."""
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO images(
                    image_id, source, lat, lon, captured_at, download_timestamp,
                    tile_id, raw_file_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(metadata["image_id"]),
                    metadata["source"],
                    metadata["lat"],
                    metadata["lon"],
                    metadata.get("captured_at"),
                    metadata.get("download_timestamp", utc_now()),
                    metadata["tile_id"],
                    metadata["raw_file_path"],
                ),
            )
            return cursor.rowcount == 1

    def images_for_tile(self, tile_id: str, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM images WHERE tile_id = ?"
        parameters: list[Any] = [tile_id]
        if status is not None:
            query += " AND validation_status = ?"
            parameters.append(status)
        query += " ORDER BY image_id"
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(query, parameters).fetchall()]

    def update_validation(
        self, image_id: str, status: str, reason: str | None, perceptual_hash: str | None
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """UPDATE images SET validation_status = ?, validation_reason = ?, perceptual_hash = ?
                   WHERE image_id = ?""",
                (status, reason, perceptual_hash, image_id),
            )

    def update_transform(
        self, image_id: str, status: str, blur_score: float | None, likely_indoor: bool | None
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """UPDATE images SET transform_status = ?, blur_score = ?, likely_indoor = ?
                   WHERE image_id = ?""",
                (status, blur_score, None if likely_indoor is None else int(likely_indoor), image_id),
            )

    def reset_transform_status_for_tile(self, tile_id: str) -> None:
        with self.connection() as connection:
            connection.execute(
                """UPDATE images SET transform_status = 'pending', blur_score = NULL,
                   likely_indoor = NULL WHERE tile_id = ? AND validation_status = 'passed'""",
                (tile_id,),
            )

    def record_stage_result(
        self, tile_id: str, stage: str, passed: int, failed: int, reasons: dict[str, int]
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO stage_results(tile_id, stage, run_timestamp, passed_count, failed_count, reasons_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tile_id, stage) DO UPDATE SET
                    run_timestamp = excluded.run_timestamp,
                    passed_count = excluded.passed_count,
                    failed_count = excluded.failed_count,
                    reasons_json = excluded.reasons_json
                """,
                (tile_id, stage, utc_now(), passed, failed, json.dumps(reasons, sort_keys=True)),
            )

    def upsert_transformed_asset(
        self, tile_id: str, parquet_path: str, count: int, dimension: int, model_name: str
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO transformed_assets VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tile_id) DO UPDATE SET
                    parquet_path = excluded.parquet_path,
                    image_count = excluded.image_count,
                    embedding_dimension = excluded.embedding_dimension,
                    model_name = excluded.model_name,
                    updated_at = excluded.updated_at
                """,
                (tile_id, parquet_path, count, dimension, model_name, utc_now()),
            )

    def transformed_assets(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM transformed_assets WHERE image_count > 0 ORDER BY tile_id"
                ).fetchall()
            ]
