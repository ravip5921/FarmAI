from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from user_interface.backend.database import connect, initialize_database
from user_interface.backend.repository import JobRepository


def _document(root: Path, document_id: str) -> dict:
    input_path = root / "documents" / document_id / "document.pdf"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"%PDF-1.7\n%%EOF")
    return {
        "id": document_id,
        "original_filename": f"{document_id}.pdf",
        "content_type": "application/pdf",
        "size_bytes": input_path.stat().st_size,
        "sha256": document_id * 8,
        "input_path": input_path,
    }


class TestInboxRepository(unittest.TestCase):
    def test_batch_creation_is_an_atomic_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            database_path = root / "farmai.sqlite3"
            initialize_database(database_path)
            repository = JobRepository(database_path)
            repository.create_documents(
                [_document(root, "doc-one"), _document(root, "doc-two")]
            )

            def start_batch(_: int) -> dict:
                return repository.create_analysis_batch(
                    template_id="boar_room",
                    ocr_engine="tesseract",
                    extra_filtered_columns=["notes"],
                    jobs_dir=root / "jobs",
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                batches = list(executor.map(start_batch, range(2)))

            accepted = [batch for batch in batches if batch["document_count"] == 2]
            empty = [batch for batch in batches if batch["document_count"] == 0]
            self.assertEqual(len(accepted), 1)
            self.assertEqual(len(empty), 1)
            self.assertEqual(repository.document_counts()["queued"], 2)
            jobs = repository.list_jobs()
            self.assertEqual(len(jobs), 2)
            self.assertEqual({job["document_id"] for job in jobs}, {"doc-one", "doc-two"})
            self.assertTrue(all(job["batch_id"] == accepted[0]["id"] for job in jobs))
            self.assertTrue(all(job["template_id"] == "boar_room" for job in jobs))
            self.assertTrue(
                all(job["extra_filtered_columns_json"] == '["notes"]' for job in jobs)
            )

            repository.create_documents([_document(root, "doc-later")])
            self.assertEqual(repository.document_counts()["pending"], 1)
            self.assertNotIn(
                "doc-later", {job["document_id"] for job in repository.list_jobs()}
            )

    def test_job_transitions_update_document_and_batch_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            database_path = root / "farmai.sqlite3"
            initialize_database(database_path)
            repository = JobRepository(database_path)
            repository.create_documents(
                [_document(root, "doc-one"), _document(root, "doc-two")]
            )
            batch = repository.create_analysis_batch(
                template_id=None,
                ocr_engine="tesseract",
                extra_filtered_columns=[],
                jobs_dir=root / "jobs",
            )

            first = repository.claim_next_job(worker_id="worker")
            self.assertEqual(repository.get_document(first["document_id"])["status"], "running")
            repository.complete_job(
                first["id"], result_path=root / "first.json", with_warnings=False
            )
            self.assertEqual(
                repository.get_document(first["document_id"])["status"], "completed"
            )

            second = repository.claim_next_job(worker_id="worker")
            repository.fail_job(
                second["id"],
                error_code="test",
                user_safe_error="Could not process this PDF.",
                technical_error="test failure",
            )

            counts = repository.document_counts()
            self.assertEqual(counts["completed"], 1)
            self.assertEqual(counts["failed"], 1)
            stored_batch = repository.get_analysis_batch(batch["id"])
            self.assertEqual(stored_batch["status"], "completed_with_errors")
            with connect(database_path) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) AS count FROM analysis_batches"
                    ).fetchone()["count"],
                    1,
                )


if __name__ == "__main__":
    unittest.main()
