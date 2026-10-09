from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from user_interface.backend import app as app_module
from user_interface.backend.config import UiConfig
from user_interface.backend.database import LATEST_SCHEMA_VERSION, initialize_database


class TestAppRoutes(unittest.TestCase):
    def test_health_reports_migrated_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            database = directory / "jobs.sqlite3"
            initialize_database(database)
            config = UiConfig(directory, database, directory / "jobs", 100, ())
            with patch.object(app_module, "get_config", return_value=config):
                result = app_module.health()
        self.assertEqual(result["schema_version"], LATEST_SCHEMA_VERSION)

    def test_frontend_routes_serve_files_and_reject_api_and_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            with patch.object(app_module, "FRONTEND_DIST", directory):
                with self.assertRaises(HTTPException) as missing:
                    app_module.frontend_root()
                self.assertEqual(missing.exception.status_code, 404)
                (directory / "index.html").write_text("FarmAI", encoding="utf-8")
                (directory / "assets").mkdir()
                (directory / "assets" / "app.js").write_text("code", encoding="utf-8")
                self.assertEqual(
                    Path(app_module.frontend_root().path).resolve(),
                    (directory / "index.html").resolve(),
                )
                self.assertEqual(
                    Path(app_module.frontend_fallback("dashboard").path).resolve(),
                    (directory / "index.html").resolve(),
                )
                self.assertEqual(
                    Path(app_module.frontend_fallback("assets/app.js").path).resolve(),
                    (directory / "assets" / "app.js").resolve(),
                )
                for path in ("api/missing", "../outside"):
                    with (
                        self.subTest(path=path),
                        self.assertRaises(HTTPException) as rejected,
                    ):
                        app_module.frontend_fallback(path)
                    self.assertEqual(rejected.exception.status_code, 404)
