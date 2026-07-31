from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.application.result_models import (
    DocumentProcessingResult,
    PageProcessingResult,
)
from user_interface.backend.services.artifact_store import read_result
from user_interface.backend.services.job_runner import run_claimed_job


class TestJobRunner(unittest.TestCase):
    @patch("user_interface.backend.services.job_runner.process_document")
    def test_result_preserves_uploaded_filename(self, process_document) -> None:
        process_document.return_value = DocumentProcessingResult(
            filename="record.jpg",
            template_id="boar_room",
            template_name="Boar Room",
            ocr_engine="tesseract",
            pages=[
                PageProcessingResult(
                    page_number=1,
                    source_image=np.full((4, 4, 3), 255, dtype=np.uint8),
                    overlay_image=np.full((4, 4, 3), 255, dtype=np.uint8),
                    image_width=4,
                    image_height=4,
                )
            ],
        )
        repository = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            job = {
                "id": "job-id",
                "original_filename": "uploaded-farm-log.jpg",
                "artifact_directory": str(artifact_dir),
                "input_path": str(artifact_dir / "input" / "record.jpg"),
                "template_id": "boar_room",
                "ocr_engine": "tesseract",
                "extra_filtered_columns_json": "[]",
                "ground_truth_path": None,
            }

            run_claimed_job(job, repository)

            result = read_result(artifact_dir / "result.json")

        self.assertEqual(result["filename"], "uploaded-farm-log.jpg")
        repository.complete_job.assert_called_once()
        repository.fail_job.assert_not_called()


if __name__ == "__main__":
    unittest.main()
