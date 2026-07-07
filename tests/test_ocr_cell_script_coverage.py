from __future__ import annotations

import io
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ocr_cell
from src.ocr import OcrText


class _NoPrepareEngine:
    def recognize(self, image: np.ndarray) -> OcrText:
        return OcrText(text="done", confidence=None)


class _PrepareEngine:
    def _prepare_image(self, image: np.ndarray) -> list[list[int]]:
        return [[0, 255]]

    def recognize(self, image: np.ndarray) -> OcrText:
        return OcrText(text="done", confidence=None)


class TestOcrCellScriptCoverage(unittest.TestCase):
    def test_load_image_returns_cv_image_and_raises_for_missing_file(self) -> None:
        image = np.zeros((1, 1), dtype=np.uint8)

        with patch("ocr_cell.cv2.imread", return_value=image):
            self.assertIs(ocr_cell.load_image(Path("cell.png")), image)

        with patch("ocr_cell.cv2.imread", return_value=None):
            with self.assertRaises(FileNotFoundError):
                ocr_cell.load_image(Path("missing.png"))

    def test_grayscale_conversion_handles_color_alpha_clip_and_invalid_shape(
        self,
    ) -> None:
        bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)
        bgra = np.array([[[255, 0, 0, 255]]], dtype=np.uint8)
        clipped = ocr_cell._to_grayscale(np.array([[300, -5]], dtype=np.int16))

        self.assertEqual(ocr_cell._to_grayscale(bgr).shape, (1, 1))
        self.assertEqual(ocr_cell._to_grayscale(bgra).shape, (1, 1))
        self.assertEqual(clipped.dtype, np.uint8)
        self.assertEqual(clipped.tolist(), [[255, 0]])
        with self.assertRaisesRegex(ValueError, "2D grayscale"):
            ocr_cell._to_grayscale(np.zeros((1, 1, 2), dtype=np.uint8))

    def test_prepare_image_for_debug_handles_plain_and_list_outputs(self) -> None:
        image = np.zeros((1, 2), dtype=np.uint8)

        self.assertIs(
            ocr_cell.prepare_image_for_debug(_NoPrepareEngine(), image), image
        )
        prepared = ocr_cell.prepare_image_for_debug(_PrepareEngine(), image)
        self.assertEqual(prepared.tolist(), [[0, 255]])

    def test_save_image_wraps_cv_write_failure(self) -> None:
        with patch("ocr_cell.cv2.imwrite", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Could not write image"):
                ocr_cell.save_image(Path("out") / "cell.png", np.zeros((1, 1)))

    def test_main_prints_input_and_prepared_ascii_with_na_confidence(self) -> None:
        image = np.array([[0, 255], [128, 64]], dtype=np.uint8)

        with (
            patch("ocr_cell.load_image", return_value=image),
            patch("ocr_cell.create_ocr_engine", return_value=_PrepareEngine()),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                ocr_cell.main(
                    [
                        "cell.png",
                        "--print-image",
                        "--print-prepared-image",
                        "--ascii-width",
                        "8",
                    ]
                )

        text = output.getvalue()
        self.assertIn("--- INPUT IMAGE ---", text)
        self.assertIn("--- PREPARED IMAGE ---", text)
        self.assertIn("Confidence: n/a", text)

    def test_module_entrypoint_runs_main(self) -> None:
        with (
            patch.object(sys, "argv", ["ocr_cell.py", "cell.png"]),
            patch("cv2.imread", return_value=np.zeros((1, 1), dtype=np.uint8)),
            patch("src.ocr.create_ocr_engine", return_value=_NoPrepareEngine()),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                runpy.run_module("ocr_cell", run_name="__main__")

        self.assertIn("OCR engine:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
