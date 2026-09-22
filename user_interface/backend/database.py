from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


Migration = tuple[int, str, Callable[[sqlite3.Connection], None]]


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path, timeout=30)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            connection.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError as exc:
            # Another process can hold the schema lock while it enables WAL for
            # the same newly-created database. BEGIN IMMEDIATE below will wait.
            if "locked" not in str(exc).casefold():
                raise
        connection.execute("PRAGMA foreign_keys=ON")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _migration_001_initial_jobs(connection: sqlite3.Connection) -> None:
    """Create the original persistent-job schema.

    CREATE IF NOT EXISTS deliberately makes this safe for databases created by
    FarmAI versions that predate the migration ledger.
    """

    connection.execute(
        """
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
        """
    )


def _migration_002_document_inbox(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            content_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            input_path TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_batches (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            template_id TEXT,
            ocr_engine TEXT NOT NULL,
            extra_filtered_columns_json TEXT NOT NULL,
            document_count INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )

    job_columns = _columns(connection, "jobs")
    if "document_id" not in job_columns:
        connection.execute(
            "ALTER TABLE jobs ADD COLUMN document_id TEXT "
            "REFERENCES documents(id) ON DELETE RESTRICT"
        )
    if "batch_id" not in job_columns:
        connection.execute(
            "ALTER TABLE jobs ADD COLUMN batch_id TEXT "
            "REFERENCES analysis_batches(id) ON DELETE SET NULL"
        )

    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_status_created "
        "ON documents(status, created_at)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_document_created "
        "ON jobs(document_id, created_at DESC)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_batch ON jobs(batch_id)"
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_one_active_per_document
        ON jobs(document_id)
        WHERE document_id IS NOT NULL AND status IN ('queued', 'running')
        """
    )


def _migration_003_job_leases(connection: sqlite3.Connection) -> None:
    job_columns = _columns(connection, "jobs")
    additions = (
        ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
        ("lease_owner", "TEXT"),
        ("lease_expires_at", "TEXT"),
        ("heartbeat_at", "TEXT"),
    )
    for column_name, definition in additions:
        if column_name not in job_columns:
            connection.execute(
                f"ALTER TABLE jobs ADD COLUMN {column_name} {definition}"
            )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_claimable "
        "ON jobs(status, created_at)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_expired_leases "
        "ON jobs(status, lease_expires_at)"
    )


MIGRATIONS: tuple[Migration, ...] = (
    (1, "initial_jobs", _migration_001_initial_jobs),
    (2, "document_inbox", _migration_002_document_inbox),
    (3, "job_leases", _migration_003_job_leases),
)
LATEST_SCHEMA_VERSION = MIGRATIONS[-1][0]


def get_schema_version(path: Path) -> int:
    if not path.is_file():
        return 0
    with connect(path) as connection:
        exists = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'schema_migrations'
            """
        ).fetchone()
        if exists is None:
            return 0
        row = connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
        ).fetchone()
    return int(row["version"]) if row is not None else 0


def migrate_database(path: Path) -> list[int]:
    """Apply every pending migration in order and return applied versions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    applied: list[int] = []
    with connect(path) as connection:
        # Serialize migration discovery and application across API/worker starts.
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        known_versions = {
            int(row["version"])
            for row in connection.execute(
                "SELECT version FROM schema_migrations"
            ).fetchall()
        }
        unknown_versions = known_versions - {version for version, _, _ in MIGRATIONS}
        if unknown_versions:
            versions = ", ".join(str(value) for value in sorted(unknown_versions))
            raise RuntimeError(f"Database has unknown migration version(s): {versions}")

        for version, name, operation in MIGRATIONS:
            if version in known_versions:
                continue
            connection.execute("SAVEPOINT schema_migration")
            try:
                operation(connection)
                connection.execute(
                    """
                    INSERT INTO schema_migrations (version, name, applied_at)
                    VALUES (?, ?, ?)
                    """,
                    (version, name, _utc_now()),
                )
                connection.execute("RELEASE SAVEPOINT schema_migration")
            except Exception:
                connection.execute("ROLLBACK TO SAVEPOINT schema_migration")
                connection.execute("RELEASE SAVEPOINT schema_migration")
                raise
            applied.append(version)
    return applied


def initialize_database(path: Path) -> None:
    migrate_database(path)
