from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import ocr_cell
from src.ocr import OcrText


class _FakeEngine:
    def __init__(self) -> None:
        self.seen_image = None

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        return np.full((2, 3), 255, dtype=np.uint8)

    def recognize(self, image: np.ndarray) -> OcrText:
        self.seen_image = image
        return OcrText(text="101.4", confidence=88.5)


class TestOcrCellScript(unittest.TestCase):
    def test_image_to_ascii_returns_preview(self) -> None:
        image = np.array([[255, 0], [128, 64]], dtype=np.uint8)

        preview = ocr_cell.image_to_ascii(image, width=8)

        self.assertTrue(preview.strip())

    def test_main_loads_image_runs_engine_and_saves_prepared_image(self) -> None:
        fake_engine = _FakeEngine()
        image = np.zeros((3, 4), dtype=np.uint8)

        with TemporaryDirectory() as tmpdir:
            prepared_path = Path(tmpdir) / "prepared.png"
            with (
                patch("ocr_cell.load_image", return_value=image),
                patch("ocr_cell.create_ocr_engine", return_value=fake_engine) as create,
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    ocr_cell.main(
                        [
                            "cell.png",
                            "--ocr-engine",
                            "tesseract",
                            "--save-prepared-image",
                            str(prepared_path),
                        ]
                    )

            text = output.getvalue()

        create.assert_called_once()
        self.assertIs(fake_engine.seen_image, image)
        self.assertIn("OCR engine: tesseract", text)
        self.assertIn("Confidence: 88.50", text)
        self.assertIn("101.4", text)
        self.assertIn("Saved prepared image to:", text)


if __name__ == "__main__":
    unittest.main()
