from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx

from user_interface.backend.app import app
from user_interface.backend.repository import JobCannotBeCancelledError
from user_interface.backend.services.artifact_store import write_result


@asynccontextmanager
async def api_client(runtime_dir: str):
    with patch.dict(os.environ, {"FARMAI_UI_RUNTIME_DIR": runtime_dir}):
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                yield client


class TestJobApiReview(unittest.IsolatedAsyncioTestCase):
    async def test_upload_limits_and_ground_truth_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async with api_client(tmpdir) as client:
                with patch.object(
                    app.state, "config", replace(app.state.config, max_upload_bytes=2)
                ):
                    response = await client.post(
                        "/api/jobs",
                        files={
                            "record": ("scan.pdf", b"many bytes", "application/pdf")
                        },
                    )
                    self.assertEqual(response.status_code, 413)
                created = await client.post(
                    "/api/jobs",
                    files={
                        "record": ("scan.pdf", b"pdf", "application/pdf"),
                        "ground_truth": ("truth.csv", b"Note\nGood\n", "text/csv"),
                    },
                )
                self.assertEqual(created.status_code, 202)
                job = app.state.repository.get_job(created.json()["job_id"])
                self.assertEqual(
                    Path(job["ground_truth_path"]).read_bytes(), b"Note\nGood\n"
                )

    async def test_cancel_delete_and_failed_result_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async with api_client(tmpdir) as client:
                created = await client.post(
                    "/api/jobs",
                    files={"record": ("scan.pdf", b"pdf", "application/pdf")},
                )
                job_id = created.json()["job_id"]
                repository = app.state.repository
                with patch.object(
                    repository,
                    "edit_records",
                    side_effect=LookupError("missing during edit"),
                ):
                    response = await client.patch(
                        f"/api/jobs/{job_id}", json={"comments": "updated"}
                    )
                    self.assertEqual(response.status_code, 404)
                with patch.object(
                    repository,
                    "cancel_job",
                    side_effect=JobCannotBeCancelledError("cannot"),
                ):
                    self.assertEqual(
                        (await client.post(f"/api/jobs/{job_id}/cancel")).status_code,
                        409,
                    )
                with patch.object(repository, "cancel_job", return_value=None):
                    self.assertEqual(
                        (await client.post(f"/api/jobs/{job_id}/cancel")).status_code,
                        404,
                    )
                with patch.object(
                    repository, "reserve_job_deletion", return_value=None
                ):
                    self.assertEqual(
                        (await client.delete(f"/api/jobs/{job_id}")).status_code, 404
                    )
                original = repository.get_job(job_id)
                invalid = {
                    **original,
                    "artifact_directory": str(Path(tmpdir) / "outside"),
                }
                with patch.object(
                    repository, "reserve_job_deletion", return_value=invalid
                ):
                    self.assertEqual(
                        (await client.delete(f"/api/jobs/{job_id}")).status_code, 500
                    )
                with patch.object(repository, "delete_job_record", return_value=False):
                    self.assertEqual(
                        (await client.delete(f"/api/jobs/{job_id}")).status_code, 500
                    )
                with patch.object(
                    repository,
                    "get_job",
                    return_value={
                        **original,
                        "status": "failed",
                        "user_safe_error": "failed safely",
                    },
                ):
                    self.assertEqual(
                        (await client.get(f"/api/jobs/{job_id}/result")).json()[
                            "detail"
                        ],
                        "failed safely",
                    )

    async def _completed_job(self, client: httpx.AsyncClient) -> tuple[str, Path]:
        created = await client.post(
            "/api/jobs",
            files={"record": ("scan.pdf", b"%PDF-1.7\n%%EOF", "application/pdf")},
        )
        self.assertEqual(created.status_code, 202)
        job_id = created.json()["job_id"]
        repository = app.state.repository
        job = repository.claim_next_job()
        self.assertEqual(job["id"], job_id)
        artifact_dir = Path(job["artifact_directory"])
        result_path = artifact_dir / "result.json"
        write_result(
            result_path,
            {
                "metrics": None,
                "pages": [
                    {
                        "page_number": 1,
                        "data_row_count": 1,
                        "columns": [
                            {
                                "key": "note",
                                "name": "Note",
                                "value_type": "english_text",
                            }
                        ],
                        "cells": [
                            {
                                "row": 1,
                                "column_key": "note",
                                "ocr_text": "Good",
                                "state": "unscored",
                            }
                        ],
                    }
                ],
            },
        )
        repository.complete_job(job_id, result_path=result_path, with_warnings=False)
        return job_id, artifact_dir

    async def test_review_download_and_ground_truth_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async with api_client(tmpdir) as client:
                job_id, artifact_dir = await self._completed_job(client)
                page_dir = artifact_dir / "pages" / "1"
                page_dir.mkdir(parents=True)
                (page_dir / "overlay.png").write_bytes(b"image")
                self.assertEqual(
                    (await client.get(f"/api/jobs/{job_id}/pages/1/overlay")).content,
                    b"image",
                )
                self.assertEqual(
                    (await client.get(f"/api/jobs/{job_id}/result")).status_code, 200
                )
                self.assertIn(
                    "Note", (await client.get(f"/api/jobs/{job_id}/download.csv")).text
                )
                self.assertEqual(
                    (await client.get(f"/api/jobs/{job_id}/download.json")).status_code,
                    200,
                )
                edited = await client.patch(
                    f"/api/jobs/{job_id}/cells",
                    json={
                        "edits": [
                            {
                                "page_number": 1,
                                "row": 1,
                                "column_key": "note",
                                "reviewed_text": "Better",
                            }
                        ]
                    },
                )
                self.assertEqual(edited.status_code, 200)
                self.assertEqual(
                    edited.json()["pages"][0]["cells"][0]["reviewed_text"], "Better"
                )
                self.assertIn(
                    "Better",
                    (await client.get(f"/api/jobs/{job_id}/download.csv")).text,
                )
                self.assertEqual(
                    (
                        await client.patch(
                            f"/api/jobs/{job_id}/cells",
                            json={
                                "edits": [
                                    {
                                        "page_number": 1,
                                        "row": 3,
                                        "column_key": "note",
                                        "reviewed_text": "X",
                                    }
                                ]
                            },
                        )
                    ).status_code,
                    404,
                )
                for name, data, expected in [
                    ("truth.txt", b"Note\nGood\n", 400),
                    ("truth.csv", b"\xff", 400),
                    ("truth.csv", b"Unknown\nGood\n", 400),
                    ("truth.csv", b"Note\nGood\n", 200),
                ]:
                    response = await client.post(
                        f"/api/jobs/{job_id}/ground-truth",
                        files={"ground_truth": (name, data, "text/csv")},
                    )
                    self.assertEqual(response.status_code, expected)
                self.assertTrue(
                    (artifact_dir / "ground_truth" / "ground_truth.csv").is_file()
                )
                self.assertEqual(
                    (
                        await client.delete(f"/api/jobs/{job_id}/ground-truth")
                    ).status_code,
                    200,
                )
                self.assertFalse(
                    (artifact_dir / "ground_truth" / "ground_truth.csv").exists()
                )

    async def test_result_and_upload_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            async with api_client(tmpdir) as client:
                for settings, expected in [
                    ("not json", 422),
                    ('{"template_id":"missing"}', 422),
                    ('{"ocr_engine":"missing"}', 422),
                ]:
                    response = await client.post(
                        "/api/jobs",
                        data={"settings": settings},
                        files={"record": ("scan.pdf", b"pdf", "application/pdf")},
                    )
                    self.assertEqual(response.status_code, expected)
                for filename, content, expected in [
                    ("scan.txt", b"x", 400),
                    ("scan.pdf", b"", 400),
                ]:
                    response = await client.post(
                        "/api/jobs",
                        files={"record": (filename, content, "application/pdf")},
                    )
                    self.assertEqual(response.status_code, expected)
                response = await client.post(
                    "/api/jobs",
                    files={
                        "record": ("scan.pdf", b"pdf", "application/pdf"),
                        "ground_truth": ("bad.txt", b"text", "text/plain"),
                    },
                )
                self.assertEqual(response.status_code, 400)
                created = await client.post(
                    "/api/jobs",
                    files={"record": ("scan.pdf", b"pdf", "application/pdf")},
                )
                job_id = created.json()["job_id"]
                self.assertEqual(
                    (await client.get("/api/jobs/not-a-uuid")).status_code, 404
                )
                self.assertEqual(
                    (await client.get(f"/api/jobs/{job_id}/result")).status_code, 409
                )
                self.assertEqual(
                    (await client.get(f"/api/jobs/{job_id}/download.csv")).status_code,
                    409,
                )
                self.assertEqual(
                    (await client.get(f"/api/jobs/{job_id}/download.json")).status_code,
                    409,
                )
                self.assertEqual(
                    (
                        await client.patch(
                            f"/api/jobs/{job_id}/cells", json={"edits": []}
                        )
                    ).status_code,
                    409,
                )
                self.assertEqual(
                    (
                        await client.delete(f"/api/jobs/{job_id}/ground-truth")
                    ).status_code,
                    409,
                )
                self.assertEqual(
                    (
                        await client.post(
                            f"/api/jobs/{job_id}/ground-truth",
                            files={
                                "ground_truth": (
                                    "truth.csv",
                                    b"Note\nGood\n",
                                    "text/csv",
                                )
                            },
                        )
                    ).status_code,
                    409,
                )
                self.assertEqual(
                    (
                        await client.get(f"/api/jobs/{job_id}/pages/0/source")
                    ).status_code,
                    404,
                )
                self.assertEqual(
                    (
                        await client.get(f"/api/jobs/{job_id}/pages/1/overlay")
                    ).status_code,
                    404,
                )
