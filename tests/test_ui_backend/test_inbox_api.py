from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import httpx

from user_interface.backend.app import app


def _pdf(label: str) -> bytes:
    return f"%PDF-1.7\n{label}\n%%EOF".encode()


@asynccontextmanager
async def _api_client(runtime_dir: str):
    with patch.dict(os.environ, {"FARMAI_UI_RUNTIME_DIR": runtime_dir}):
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                yield client


class TestInboxApi(unittest.IsolatedAsyncioTestCase):
    async def test_upload_batch_snapshot_and_later_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async with _api_client(tmpdir) as client:
                uploaded = await client.post(
                    "/api/documents",
                    files=[
                        ("documents", ("first.pdf", _pdf("first"), "application/pdf")),
                        ("documents", ("second.PDF", _pdf("second"), "application/pdf")),
                    ],
                )

                self.assertEqual(uploaded.status_code, 201)
                payload = uploaded.json()
                self.assertEqual(payload["counts"]["total"], 2)
                self.assertEqual(payload["counts"]["pending"], 2)
                self.assertEqual(len(payload["documents"]), 2)
                first_path = (
                    Path(tmpdir)
                    / "documents"
                    / payload["documents"][0]["document_id"]
                    / "document.pdf"
                )
                self.assertEqual(first_path.read_bytes(), _pdf("first"))

                started = await client.post(
                    "/api/analysis-batches",
                    json={
                        "template_id": "boar_room",
                        "ocr_engine": "tesseract",
                        "extra_filtered_columns": ["notes"],
                    },
                )

                self.assertEqual(started.status_code, 202)
                batch = started.json()
                self.assertEqual(batch["status"], "queued")
                self.assertEqual(batch["document_count"], 2)
                self.assertEqual(len(batch["job_ids"]), 2)
                self.assertEqual(batch["counts"]["pending"], 0)
                self.assertEqual(batch["counts"]["queued"], 2)
                duplicate = await client.post(
                    "/api/analysis-batches",
                    json={"ocr_engine": "tesseract"},
                )
                self.assertEqual(duplicate.status_code, 202)
                self.assertEqual(duplicate.json()["status"], "empty")
                self.assertEqual(duplicate.json()["job_ids"], [])

                later = await client.post(
                    "/api/documents",
                    files={
                        "documents": (
                            "later.pdf",
                            _pdf("later"),
                            "application/pdf",
                        )
                    },
                )
                self.assertEqual(later.status_code, 201)
                self.assertEqual(later.json()["counts"]["pending"], 1)
                self.assertEqual(later.json()["counts"]["queued"], 2)

                jobs = (await client.get("/api/jobs")).json()["jobs"]
                batch_jobs = [job for job in jobs if job["batch_id"] == batch["batch_id"]]
                self.assertEqual(len(batch_jobs), 2)
                self.assertTrue(all(job["document_id"] for job in batch_jobs))
                self.assertTrue(all(job["template_id"] == "boar_room" for job in batch_jobs))
                self.assertTrue(all(job["ocr_engine"] == "tesseract" for job in batch_jobs))
                stored_batch = await client.get(
                    f"/api/analysis-batches/{batch['batch_id']}"
                )
                self.assertEqual(stored_batch.status_code, 200)
                self.assertEqual(
                    stored_batch.json()["settings"]["extra_filtered_columns"],
                    ["notes"],
                )

    async def test_rejects_invalid_pdf_and_cleans_up_whole_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async with _api_client(tmpdir) as client:
                response = await client.post(
                    "/api/documents",
                    files=[
                        ("documents", ("valid.pdf", _pdf("valid"), "application/pdf")),
                        ("documents", ("fake.pdf", b"not-a-pdf", "application/pdf")),
                    ],
                )

                self.assertEqual(response.status_code, 400)
                documents = await client.get("/api/documents")
                self.assertEqual(documents.json()["counts"]["total"], 0)
                documents_root = Path(tmpdir) / "documents"
                self.assertEqual(list(documents_root.iterdir()), [])

    async def test_rejects_non_pdf_extension_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async with _api_client(tmpdir) as client:
                response = await client.post(
                    "/api/documents",
                    files={
                        "documents": ("scan.png", _pdf("disguised"), "image/png")
                    },
                )

                self.assertEqual(response.status_code, 400)
                self.assertEqual(list((Path(tmpdir) / "documents").iterdir()), [])

    async def test_document_counts_are_not_limited_by_list_pagination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async with _api_client(tmpdir) as client:
                for number in range(3):
                    response = await client.post(
                        "/api/documents",
                        files={
                            "documents": (
                                f"scan-{number}.pdf",
                                _pdf(str(number)),
                                "application/pdf",
                            )
                        },
                    )
                    self.assertEqual(response.status_code, 201)

                page = (await client.get("/api/documents?limit=1&offset=1")).json()
                self.assertEqual(len(page["documents"]), 1)
                self.assertEqual(page["counts"]["total"], 3)
                self.assertEqual(page["counts"]["pending"], 3)
                self.assertTrue(page["has_more"])


if __name__ == "__main__":
    unittest.main()
