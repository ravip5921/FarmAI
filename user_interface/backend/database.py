from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path, timeout=30)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress_current INTEGER NOT NULL DEFAULT 0,
                progress_total INTEGER NOT NULL DEFAULT 0,
                original_filename TEXT NOT NULL,
                template_id TEXT,
                ocr_engine TEXT NOT NULL,
                extra_filtered_columns_json TEXT NOT NULL,
                input_path TEXT NOT NULL,
                ground_truth_path TEXT,
                result_path TEXT,
                artifact_directory TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL,
                error_code TEXT,
                user_safe_error TEXT,
                technical_error TEXT
            )
            """)
