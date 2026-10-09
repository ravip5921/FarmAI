from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from user_interface.backend.database import connect, initialize_database
from user_interface.backend.repository import (
    JobCannotBeCancelledError,
    JobRepository,
    _sync_batch_status,
)


class TestRepositoryEdges(unittest.TestCase):
    def test_missing_records_and_cancelled_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            database = root / "jobs.sqlite3"
            initialize_database(database)
            repository = JobRepository(database)
            self.assertEqual(repository.create_documents([]), [])
            self.assertIsNone(repository.get_analysis_batch("missing"))
            self.assertIsNone(repository.reserve_job_deletion("missing"))
            self.assertFalse(repository.delete_job_record("missing"))
            self.assertIsNone(repository.cancel_job("missing"))
            self.assertFalse(repository.is_cancelled("missing"))
            with patch.object(repository, "get_document", return_value=None):
                with self.assertRaisesRegex(
                    RuntimeError, "Could not create uploaded document"
                ):
                    repository.create_documents(
                        [
                            {
                                "id": "doc",
                                "original_filename": "doc.pdf",
                                "content_type": "application/pdf",
                                "size_bytes": 1,
                                "sha256": "hash",
                                "input_path": root / "doc.pdf",
                            }
                        ]
                    )
            with patch.object(repository, "get_job", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "Could not create job"):
                    repository.create_job(
                        job_id="job",
                        original_filename="x.pdf",
                        template_id=None,
                        ocr_engine="tesseract",
                        extra_filtered_columns=[],
                        input_path=root / "x.pdf",
                        ground_truth_path=None,
                        artifact_directory=root / "jobs" / "job",
                    )
            self.assertFalse(repository.is_cancelled("job"))
            repository.update_progress("job", stage="queued", current=1, total=2)
            repository.set_ground_truth_path("job", root / "truth.csv")
            repository.touch("job")
            self.assertEqual(repository.cancel_job("job")["status"], "cancelled")
            self.assertTrue(repository.is_cancelled("job"))
            with self.assertRaises(JobCannotBeCancelledError):
                repository.cancel_job("job")

    def test_batch_status_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            database = root / "jobs.sqlite3"
            initialize_database(database)
            repository = JobRepository(database)
            documents = []
            for index in range(2):
                doc_id = f"doc-{index}"
                input_path = root / f"{doc_id}.pdf"
                input_path.write_bytes(b"%PDF-1.7\n")
                documents.append(
                    {
                        "id": doc_id,
                        "original_filename": f"{doc_id}.pdf",
                        "content_type": "application/pdf",
                        "size_bytes": 9,
                        "sha256": doc_id,
                        "input_path": input_path,
                    }
                )
            repository.create_documents(documents)
            batch = repository.create_analysis_batch(
                template_id=None,
                ocr_engine="tesseract",
                extra_filtered_columns=[],
                jobs_dir=root / "jobs",
            )
            batch_id = batch["id"]
            self.assertEqual(len(batch["jobs"]), 2)
            for statuses, expected in [
                (("running", "queued"), "running"),
                (("completed", "queued"), "running"),
                (("completed", "completed_with_warnings"), "completed_with_warnings"),
                (("cancelled", "cancelled"), "cancelled"),
                (("failed", "failed"), "failed"),
            ]:
                with self.subTest(statuses=statuses), connect(database) as connection:
                    for record, status in zip(batch["jobs"], statuses):
                        connection.execute(
                            "UPDATE jobs SET status = ? WHERE id = ?",
                            (status, record["id"]),
                        )
                    _sync_batch_status(connection, batch_id, now="2026-01-01")
                    actual = connection.execute(
                        "SELECT status FROM analysis_batches WHERE id = ?", (batch_id,)
                    ).fetchone()["status"]
                    self.assertEqual(actual, expected)
            with connect(database) as connection:
                _sync_batch_status(connection, "missing", now="2026-01-01")

    def test_batch_job_cancel_and_delete_restore_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            database = root / "jobs.sqlite3"
            initialize_database(database)
            repository = JobRepository(database)
            input_path = root / "doc.pdf"
            input_path.write_bytes(b"%PDF-1.7\n")
            repository.create_documents(
                [
                    {
                        "id": "doc",
                        "original_filename": "doc.pdf",
                        "content_type": "application/pdf",
                        "size_bytes": 9,
                        "sha256": "hash",
                        "input_path": input_path,
                    }
                ]
            )
            batch = repository.create_analysis_batch(
                template_id=None,
                ocr_engine="tesseract",
                extra_filtered_columns=[],
                jobs_dir=root / "jobs",
            )
            job_id = batch["jobs"][0]["id"]
            repository.cancel_job(job_id)
            self.assertEqual(repository.get_document("doc")["status"], "cancelled")
            self.assertEqual(
                repository.get_analysis_batch(batch["id"])["status"], "cancelled"
            )
            self.assertTrue(repository.delete_job_record(job_id))
            self.assertEqual(repository.get_document("doc")["status"], "pending")

    def test_claim_race_returns_no_job(self) -> None:
        connection = Mock()
        row = {"id": "job", "document_id": None, "batch_id": None}

        def execute(statement, *args):
            if "SELECT * FROM jobs" in statement:
                result = Mock()
                result.fetchone.return_value = row
                return result
            if "UPDATE jobs" in statement:
                result = Mock()
                result.rowcount = 0
                return result
            return Mock()

        connection.execute.side_effect = execute

        @contextmanager
        def fake_connect(_path):
            yield connection

        with patch("user_interface.backend.repository.connect", fake_connect):
            self.assertIsNone(JobRepository(Path("unused")).claim_next_job())

    def test_expired_batch_job_is_requeued_and_synced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            database = root / "jobs.sqlite3"
            initialize_database(database)
            repository = JobRepository(database)
            input_path = root / "doc.pdf"
            input_path.write_bytes(b"%PDF-1.7\n")
            repository.create_documents(
                [
                    {
                        "id": "doc",
                        "original_filename": "doc.pdf",
                        "content_type": "application/pdf",
                        "size_bytes": 9,
                        "sha256": "hash",
                        "input_path": input_path,
                    }
                ]
            )
            batch = repository.create_analysis_batch(
                template_id=None,
                ocr_engine="tesseract",
                extra_filtered_columns=[],
                jobs_dir=root / "jobs",
            )
            repository.claim_next_job(worker_id="worker", lease_seconds=1)
            with connect(database) as connection:
                connection.execute(
                    "UPDATE jobs SET lease_expires_at = '2000-01-01' WHERE batch_id = ?",
                    (batch["id"],),
                )
            self.assertEqual(repository.recover_interrupted_jobs(), 1)
            self.assertEqual(repository.get_document("doc")["status"], "queued")
            self.assertEqual(
                repository.get_analysis_batch(batch["id"])["status"], "queued"
            )
