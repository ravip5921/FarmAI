from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.application.ground_truth import GroundTruthError
from src.application.result_models import (
    DocumentProcessingResult,
    PageProcessingResult,
    ProcessingProgress,
)
from user_interface.backend.services.job_runner import (
    _resumable_engine,
    _user_safe_error,
    run_claimed_job,
)


def job(directory: Path) -> dict:
    return {
        "id": "job",
        "artifact_directory": str(directory),
        "input_path": "scan.pdf",
        "original_filename": "scan.pdf",
        "template_id": None,
        "ocr_engine": "tesseract",
        "extra_filtered_columns_json": "[]",
        "ground_truth_path": None,
    }


def processed() -> DocumentProcessingResult:
    image = np.zeros((3, 3), dtype=np.uint8)
    return DocumentProcessingResult(
        filename="scan.pdf",
        template_id=None,
        template_name=None,
        ocr_engine="tesseract",
        pages=[
            PageProcessingResult(
                page_number=1,
                source_image=image,
                overlay_image=image,
                image_width=3,
                image_height=3,
            )
        ],
    )


class TestJobRunnerFailures(unittest.TestCase):
    def test_public_error_mapping(self) -> None:
        cases = [
            (ValueError("could not find the table"), "table_not_found"),
            (RuntimeError("LLM OCR is not configured"), "llm_not_configured"),
            (RuntimeError("HTTP 403 Forbidden"), "llm_authorization"),
            (GroundTruthError("bad csv"), "ground_truth_invalid"),
            (RuntimeError("unexpected"), "processing_failed"),
        ]
        for error, code in cases:
            with self.subTest(code=code):
                self.assertEqual(_user_safe_error(error)[0], code)

    def test_missing_engine_is_created_for_llm_job(self) -> None:
        with patch(
            "user_interface.backend.services.job_runner.create_ocr_engine",
            return_value=Mock(),
        ) as create:
            wrapped = _resumable_engine(
                {"ocr_engine": "llm-vision"}, Path("unused"), None
            )
        self.assertIsNotNone(wrapped)
        create.assert_called_once_with("llm-vision")

    def test_progress_cancellation_before_and_after_update(self) -> None:
        for cancelled in [(True,), (False, True)]:
            with (
                self.subTest(cancelled=cancelled),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                repo = Mock()
                repo.is_cancelled.side_effect = cancelled
                with patch(
                    "user_interface.backend.services.job_runner.process_document"
                ) as process:
                    process.side_effect = lambda *args, **kwargs: kwargs[
                        "progress_callback"
                    ](ProcessingProgress(stage="preparing", completed=1, total=2))
                    run_claimed_job(job(Path(tmpdir)), repo)
                repo.complete_job.assert_not_called()
                repo.fail_job.assert_not_called()
                self.assertEqual(
                    repo.update_progress.call_count, 0 if cancelled[0] else 1
                )

    def test_preview_write_errors_are_reported(self) -> None:
        for writes in [(False,), (True, False)]:
            with self.subTest(writes=writes), tempfile.TemporaryDirectory() as tmpdir:
                repo = Mock()
                with (
                    patch(
                        "user_interface.backend.services.job_runner.process_document",
                        return_value=processed(),
                    ),
                    patch(
                        "user_interface.backend.services.job_runner.cv2.imwrite",
                        side_effect=writes,
                    ),
                ):
                    run_claimed_job(job(Path(tmpdir)), repo)
                repo.fail_job.assert_called_once()
                self.assertIn(
                    "Could not save", repo.fail_job.call_args.kwargs["technical_error"]
                )

    def test_invalid_ground_truth_keeps_completed_result_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            truth = directory / "truth.csv"
            truth.write_text("Bad\n1\n", encoding="utf-8")
            task = job(directory)
            task["ground_truth_path"] = str(truth)
            repo = Mock()
            with patch(
                "user_interface.backend.services.job_runner.process_document",
                return_value=processed(),
            ):
                run_claimed_job(task, repo)
            self.assertTrue(repo.complete_job.call_args.kwargs["with_warnings"])
            self.assertIn("ground_truth_error", (directory / "result.json").read_text())

    def test_unexpected_processing_error_fails_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Mock()
            with patch(
                "user_interface.backend.services.job_runner.process_document",
                side_effect=RuntimeError("boom"),
            ):
                run_claimed_job(job(Path(tmpdir)), repo)
            self.assertEqual(
                repo.fail_job.call_args.kwargs["error_code"], "processing_failed"
            )
