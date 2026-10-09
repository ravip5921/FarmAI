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
    async def test_delete_selected_pdfs_and_reject_running_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async with _api_client(tmpdir) as client:
                uploaded = await client.post("/api/documents", files=[
                    ("documents", (f"{i}.pdf", _pdf(str(i)), "application/pdf")) for i in range(3)
                ])
                ids = [item["document_id"] for item in uploaded.json()["documents"]]
                deleted = await client.post("/api/documents/delete", json={"document_ids": [ids[2]]})
                self.assertEqual(deleted.status_code, 200)
                self.assertFalse((Path(tmpdir) / "documents" / ids[2]).exists())
                batch = (await client.post("/api/analysis-batches", json={"ocr_engine": "tesseract"})).json()
                claimed = app.state.repository.claim_next_job(worker_id="test")
                other = next(i for i in ids[:2] if i != claimed["document_id"])
                rejected = await client.post("/api/documents/delete", json={"document_ids": [other, claimed["document_id"]]})
                self.assertEqual(rejected.status_code, 409)
                self.assertIsNotNone(app.state.repository.get_document(other))
                self.assertTrue((Path(tmpdir) / "documents" / other).exists())
                missing = await client.post("/api/documents/delete", json={"document_ids": [other, "missing"]})
                self.assertEqual(missing.status_code, 404)
                self.assertIsNotNone(app.state.repository.get_document(other))
                jobs = app.state.repository.list_jobs()
                queued = next(j for j in jobs if j["document_id"] == other)
                artifact = Path(queued["artifact_directory"])
                artifact.mkdir(parents=True)
                (artifact / "result.json").write_text("{}")
                deleted = await client.post("/api/documents/delete", json={"document_ids": [other, other]})
                self.assertEqual(deleted.status_code, 200)
                self.assertEqual(deleted.json()["deleted"], 1)
                self.assertFalse(artifact.exists())
                self.assertIsNone(app.state.repository.get_job(queued["id"]))
                self.assertEqual(app.state.repository.get_analysis_batch(batch["batch_id"])["document_count"], 1)
                app.state.repository.cancel_job(claimed["id"])
                deleted = await client.post("/api/documents/delete", json={"document_ids": [claimed["document_id"]]})
                self.assertEqual(deleted.status_code, 200)
                self.assertEqual(app.state.repository.document_counts()["total"], 0)
                self.assertIsNone(app.state.repository.get_analysis_batch(batch["batch_id"]))

    async def test_upload_saves_per_file_settings_before_analysis(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            async with _api_client(tmpdir) as client:
                files = [
                    ("documents", ("same.pdf", _pdf("one"), "application/pdf")),
                    ("documents", ("same.pdf", _pdf("two"), "application/pdf")),
                ]
                custom = {"template_id": "boar_room", "ocr_engine": "tesseract",
                          "extra_filtered_columns": ["comments"]}
                for invalid in ("bad json", "[]", json.dumps([custom, {"template_id": "unknown"}])):
                    response = await client.post("/api/documents", files=files, data={"settings": invalid})
                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(app.state.repository.document_counts()["total"], 0)
                uploaded = await client.post("/api/documents", files=files,
                                             data={"settings": json.dumps([custom, None])})
                self.assertEqual(uploaded.status_code, 201)
                documents = uploaded.json()["documents"]
                self.assertEqual(documents[0]["settings"], custom)
                self.assertIsNone(documents[1]["settings"])
                self.assertTrue(all(item["status"] == "pending" for item in documents))
                self.assertEqual((await client.get("/api/jobs")).json()["jobs"], [])
                await client.post("/api/analysis-batches", json={"ocr_engine": "llm-vision"})
                jobs = {item["document_id"]: item for item in (await client.get("/api/jobs")).json()["jobs"]}
                self.assertEqual(jobs[documents[0]["document_id"]]["template_id"], "boar_room")
                self.assertEqual(jobs[documents[0]["document_id"]]["extra_filtered_columns"], ["comments"])
                self.assertEqual(jobs[documents[1]["document_id"]]["ocr_engine"], "llm-vision")

    async def test_per_document_settings_and_queue_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async with _api_client(tmpdir) as client:
                uploaded = await client.post(
                    "/api/documents",
                    files=[
                        ("documents", ("same.pdf", _pdf("one"), "application/pdf")),
                        ("documents", ("same.pdf", _pdf("two"), "application/pdf")),
                    ],
                )
                ids = [item["document_id"] for item in uploaded.json()["documents"]]
                custom = {
                    "template_id": "boar_room",
                    "ocr_engine": "tesseract",
                    "extra_filtered_columns": ["comments"],
                }
                edited = await client.patch(
                    "/api/documents",
                    json={
                        "document_ids": [ids[0]],
                        "settings": custom,
                        "reference_id": "Barn A",
                        "comments": "September records",
                    },
                )
                self.assertEqual(edited.status_code, 200)
                await client.post(
                    "/api/analysis-batches", json={"ocr_engine": "llm-vision"}
                )
                jobs = {
                    j["document_id"]: j
                    for j in (await client.get("/api/jobs")).json()["jobs"]
                }
                self.assertEqual(jobs[ids[0]]["template_id"], "boar_room")
                self.assertEqual(jobs[ids[0]]["extra_filtered_columns"], ["comments"])
                self.assertEqual(jobs[ids[0]]["reference_id"], "Barn A")
                self.assertEqual(jobs[ids[0]]["comments"], "September records")
                self.assertEqual(jobs[ids[1]]["ocr_engine"], "llm-vision")
                bulk = await client.patch(
                    "/api/documents", json={"document_ids": ids, "settings": custom}
                )
                self.assertEqual(bulk.status_code, 200)
                job_id = jobs[ids[1]]["job_id"]
                edited = await client.patch(
                    f"/api/jobs/{job_id}",
                    json={
                        "settings": {"ocr_engine": "llm-vision"},
                        "comments": "Updated",
                    },
                )
                self.assertEqual(edited.status_code, 200)
                self.assertEqual(edited.json()["extra_filtered_columns"], [])
                document = app.state.repository.get_document(ids[1])
                self.assertEqual(document["comments"], "Updated")
                claimed = app.state.repository.claim_next_job(worker_id="test-worker")
                blocked = await client.patch(
                    f"/api/jobs/{claimed['id']}", json={"settings": custom}
                )
                self.assertEqual(blocked.status_code, 409)
                # Bulk changes must roll back even if an earlier record was editable.
                other = next(i for i in ids if i != claimed["document_id"])
                blocked = await client.patch(
                    "/api/documents",
                    json={
                        "document_ids": [other, claimed["document_id"]],
                        "settings": custom,
                        "comments": "Must roll back",
                    },
                )
                self.assertEqual(blocked.status_code, 409)
                self.assertNotEqual(
                    app.state.repository.get_document(other)["comments"],
                    "Must roll back",
                )
                metadata = await client.patch(
                    f"/api/jobs/{claimed['id']}", json={"comments": "Still editable"}
                )
                self.assertEqual(metadata.status_code, 200)
                invalid = await client.patch(
                    "/api/documents",
                    json={
                        "document_ids": [other],
                        "settings": {"template_id": "missing"},
                    },
                )
                self.assertEqual(invalid.status_code, 422)
                missing = await client.patch(
                    "/api/documents",
                    json={"document_ids": ["missing"], "comments": "x"},
                )
                self.assertEqual(missing.status_code, 404)

    async def test_upload_batch_snapshot_and_later_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async with _api_client(tmpdir) as client:
                uploaded = await client.post(
                    "/api/documents",
                    files=[
                        ("documents", ("first.pdf", _pdf("first"), "application/pdf")),
                        (
                            "documents",
                            ("second.PDF", _pdf("second"), "application/pdf"),
                        ),
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
                batch_jobs = [
                    job for job in jobs if job["batch_id"] == batch["batch_id"]
                ]
                self.assertEqual(len(batch_jobs), 2)
                self.assertTrue(all(job["document_id"] for job in batch_jobs))
                self.assertTrue(
                    all(job["template_id"] == "boar_room" for job in batch_jobs)
                )
                self.assertTrue(
                    all(job["ocr_engine"] == "tesseract" for job in batch_jobs)
                )
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
                    files={"documents": ("scan.png", _pdf("disguised"), "image/png")},
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
