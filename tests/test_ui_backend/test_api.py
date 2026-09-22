from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from user_interface.backend.app import app


class TestUiApi(unittest.TestCase):
    def test_settings_and_queued_job_survive_status_lookup(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {"FARMAI_UI_RUNTIME_DIR": tmpdir},
            ),
        ):
            with TestClient(app) as client:
                settings = client.get("/api/settings")
                self.assertEqual(settings.status_code, 200)
                self.assertEqual(
                    settings.json()["defaults"]["ocr_engine"], "llm-vision"
                )
                self.assertIsNone(settings.json()["defaults"]["template_id"])
                created = client.post(
                    "/api/jobs",
                    data={
                        "settings": json.dumps(
                            {
                                "template_id": "boar_room",
                                "ocr_engine": "llm-vision",
                                "extra_filtered_columns": [],
                            }
                        )
                    },
                    files={
                        "record": ("record.png", b"fake-image", "image/png"),
                    },
                )
                self.assertEqual(created.status_code, 202)
                job_id = created.json()["job_id"]

                status = client.get(f"/api/jobs/{job_id}")

                self.assertEqual(status.status_code, 200)
                self.assertEqual(status.json()["status"], "queued")
                listed = client.get("/api/jobs")
                self.assertEqual(listed.status_code, 200)
                self.assertEqual(listed.json()["jobs"][0]["job_id"], job_id)
                input_file = Path(tmpdir) / "jobs" / job_id / "input" / "record.png"
                self.assertEqual(input_file.read_bytes(), b"fake-image")
                deleted = client.delete(f"/api/jobs/{job_id}")
                self.assertEqual(deleted.status_code, 204)
                self.assertEqual(client.get(f"/api/jobs/{job_id}").status_code, 404)
                self.assertEqual(client.get("/api/jobs").json()["jobs"], [])
                self.assertFalse(input_file.parent.parent.exists())

    def test_rejects_unsupported_upload(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {"FARMAI_UI_RUNTIME_DIR": tmpdir},
            ),
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/api/jobs",
                    files={
                        "record": ("record.txt", b"not an image", "text/plain"),
                    },
                )

        self.assertEqual(response.status_code, 400)

    def test_running_job_cannot_be_deleted(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {"FARMAI_UI_RUNTIME_DIR": tmpdir},
            ),
        ):
            with TestClient(app) as client:
                created = client.post(
                    "/api/jobs",
                    files={
                        "record": ("record.png", b"fake-image", "image/png"),
                    },
                )
                job_id = created.json()["job_id"]
                app.state.repository.claim_next_job()

                response = client.delete(f"/api/jobs/{job_id}")

                self.assertEqual(response.status_code, 409)
                self.assertEqual(client.get(f"/api/jobs/{job_id}").status_code, 200)

    def test_queued_job_can_be_cancelled(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {"FARMAI_UI_RUNTIME_DIR": tmpdir},
            ),
        ):
            with TestClient(app) as client:
                created = client.post(
                    "/api/jobs",
                    files={
                        "record": ("record.png", b"fake-image", "image/png"),
                    },
                )
                job_id = created.json()["job_id"]

                response = client.post(f"/api/jobs/{job_id}/cancel")

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "cancelled")
                self.assertIsNone(app.state.repository.claim_next_job())

    def test_running_job_can_be_cancelled(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {"FARMAI_UI_RUNTIME_DIR": tmpdir},
            ),
        ):
            with TestClient(app) as client:
                created = client.post(
                    "/api/jobs",
                    files={
                        "record": ("record.png", b"fake-image", "image/png"),
                    },
                )
                job_id = created.json()["job_id"]
                app.state.repository.claim_next_job()

                response = client.post(f"/api/jobs/{job_id}/cancel")

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "cancelled")
                self.assertEqual(
                    client.get(f"/api/jobs/{job_id}").json()["stage"],
                    "cancelled",
                )


if __name__ == "__main__":
    unittest.main()
