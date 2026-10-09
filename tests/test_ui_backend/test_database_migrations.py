from __future__ import annotations

import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from user_interface.backend import database as database_module
from user_interface.backend.database import (
    LATEST_SCHEMA_VERSION,
    connect,
    get_schema_version,
    initialize_database,
    migrate_database,
)
from user_interface.backend.repository import JobRepository

LEGACY_JOBS_SCHEMA = """
CREATE TABLE jobs (
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


class TestDatabaseMigrations(unittest.TestCase):
    def test_connection_propagates_non_lock_journal_error(self) -> None:
        connection = Mock()
        connection.execute.side_effect = [None, sqlite3.OperationalError("disk full")]
        with patch.object(sqlite3, "connect", return_value=connection):
            with self.assertRaisesRegex(sqlite3.OperationalError, "disk full"):
                with connect(Path("unused.sqlite3")):
                    pass
        connection.rollback.assert_called_once()
        connection.close.assert_called_once()

    def test_schema_version_and_migration_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database_path = Path(tmpdir) / "jobs.sqlite3"
            self.assertEqual(get_schema_version(database_path), 0)
            sqlite3.connect(database_path).close()
            self.assertEqual(get_schema_version(database_path), 0)
            initialize_database(database_path)
            with connect(database_path) as connection:
                connection.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (99, 'future', 'now')"
                )
            with self.assertRaisesRegex(RuntimeError, "unknown migration"):
                migrate_database(database_path)
            latest_version = database_module.MIGRATIONS[-1][0]
            with connect(database_path) as connection:
                connection.execute("DELETE FROM schema_migrations WHERE version = 99")
                connection.execute(
                    "DELETE FROM schema_migrations WHERE version = ?",
                    (latest_version,),
                )
            migration = (
                latest_version,
                "failing",
                lambda connection: (_ for _ in ()).throw(RuntimeError("broken")),
            )
            with patch.object(
                database_module,
                "MIGRATIONS",
                (*database_module.MIGRATIONS[:-1], migration),
            ):
                with self.assertRaisesRegex(RuntimeError, "broken"):
                    migrate_database(database_path)
            self.assertEqual(get_schema_version(database_path), latest_version - 1)

    def test_upgrades_legacy_database_without_changing_job_or_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            database_path = root / "farmai.sqlite3"
            artifact = root / "jobs" / "legacy-id" / "input" / "record.pdf"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"%PDF-1.7\nlegacy")
            connection = sqlite3.connect(database_path)
            connection.execute(LEGACY_JOBS_SCHEMA)
            connection.execute(
                """
                INSERT INTO jobs (
                    id, status, stage, original_filename, ocr_engine,
                    extra_filtered_columns_json, input_path, artifact_directory,
                    created_at, updated_at
                ) VALUES (?, 'completed', 'completed', ?, 'tesseract', '[]', ?, ?, ?, ?)
                """,
                (
                    "legacy-id",
                    "legacy.pdf",
                    str(artifact),
                    str(artifact.parents[1]),
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            connection.commit()
            connection.close()

            applied = migrate_database(database_path)

            self.assertEqual(applied, [1, 2, 3, 4])
            self.assertEqual(get_schema_version(database_path), LATEST_SCHEMA_VERSION)
            self.assertEqual(migrate_database(database_path), [])
            with connect(database_path) as migrated:
                job = migrated.execute(
                    "SELECT * FROM jobs WHERE id = 'legacy-id'"
                ).fetchone()
                columns = {
                    row["name"]
                    for row in migrated.execute("PRAGMA table_info(jobs)").fetchall()
                }
            self.assertIsNotNone(job)
            self.assertEqual(job["original_filename"], "legacy.pdf")
            self.assertIsNone(job["document_id"])
            self.assertEqual(job["attempt_count"], 0)
            self.assertTrue(artifact.is_file())
            self.assertTrue(
                {"document_id", "batch_id", "lease_owner", "lease_expires_at"}
                <= columns
            )

    def test_concurrent_initialization_is_serialized_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database_path = Path(tmpdir) / "farmai.sqlite3"
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(
                    executor.map(lambda _: migrate_database(database_path), range(2))
                )

            self.assertIn([1, 2, 3, 4], results)
            self.assertIn([], results)
            self.assertEqual(get_schema_version(database_path), LATEST_SCHEMA_VERSION)

    def test_only_expired_leases_are_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            database_path = root / "farmai.sqlite3"
            initialize_database(database_path)
            repository = JobRepository(database_path)
            repository.create_job(
                job_id="job-id",
                original_filename="record.pdf",
                template_id=None,
                ocr_engine="tesseract",
                extra_filtered_columns=[],
                input_path=root / "record.pdf",
                ground_truth_path=None,
                artifact_directory=root / "jobs" / "job-id",
            )

            claimed = repository.claim_next_job(
                worker_id="worker-one", lease_seconds=3600
            )

            self.assertIsNotNone(claimed)
            self.assertEqual(claimed["attempt_count"], 1)
            self.assertEqual(repository.recover_interrupted_jobs(), 0)
            self.assertTrue(
                repository.heartbeat_job(
                    "job-id", worker_id="worker-one", lease_seconds=3600
                )
            )
            with connect(database_path) as connection:
                connection.execute("""
                    UPDATE jobs
                    SET lease_expires_at = '2000-01-01T00:00:00+00:00',
                        progress_current = 7, progress_total = 10
                    WHERE id = 'job-id'
                    """)

            self.assertEqual(repository.recover_interrupted_jobs(), 1)
            recovered = repository.get_job("job-id")
            self.assertEqual(recovered["status"], "queued")
            self.assertEqual(recovered["progress_current"], 7)
            self.assertIsNotNone(recovered["started_at"])
            reclaimed = repository.claim_next_job(
                worker_id="worker-two", lease_seconds=3600
            )
            self.assertEqual(reclaimed["attempt_count"], 2)
            self.assertEqual(reclaimed["lease_owner"], "worker-two")


if __name__ == "__main__":
    unittest.main()
