from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.ocr.tesseract_engine import OcrText, TesseractConfig, TesseractOcrEngine


class TestTesseractOcrEngine(unittest.TestCase):
    def test_prepare_image_converts_color_and_clips_to_uint8(self) -> None:
        engine = TesseractOcrEngine()
        image = np.array(
            [
                [[300.0, 0.0, 0.0], [0.0, 300.0, 0.0]],
                [[0.0, 0.0, 300.0], [-10.0, 10.0, 20.0]],
            ],
            dtype=np.float32,
        )

        prepared = engine._prepare_image(image)

        self.assertEqual(prepared.dtype, np.uint8)
        self.assertEqual(prepared.shape, (2, 2))
        self.assertTrue(np.all(prepared >= 0))
        self.assertTrue(np.all(prepared <= 255))

    def test_prepare_image_keeps_uint8_grayscale_values(self) -> None:
        engine = TesseractOcrEngine(TesseractConfig(lang="spa", psm=7, oem=1))
        image = np.array([[0, 128], [200, 255]], dtype=np.uint8)

        prepared = engine._prepare_image(image)

        np.testing.assert_array_equal(prepared, image)

    @patch("src.ocr.tesseract_engine.pytesseract.image_to_data")
    @patch("src.ocr.tesseract_engine.pytesseract.image_to_string")
    def test_recognize_returns_trimmed_text_and_mean_confidence(
        self,
        image_to_string,
        image_to_data,
    ) -> None:
        image_to_string.return_value = "  total  \n"
        image_to_data.return_value = {
            "conf": ["90", "-1", "not-a-number", None, "80.5"]
        }
        engine = TesseractOcrEngine(TesseractConfig(lang="eng", psm=6, oem=3))

        result = engine.recognize(np.array([[0, 255]], dtype=np.uint8))

        self.assertEqual(result, OcrText(text="total", confidence=85.25))
        image_to_string.assert_called_once()
        image_to_data.assert_called_once()
        self.assertEqual(image_to_string.call_args.kwargs["lang"], "eng")
        self.assertEqual(image_to_string.call_args.kwargs["config"], "--oem 3 --psm 6")

    @patch("src.ocr.tesseract_engine.pytesseract.image_to_data")
    def test_mean_confidence_returns_none_without_valid_values(
        self, image_to_data
    ) -> None:
        image_to_data.return_value = {"conf": ["-1", "bad", None]}
        engine = TesseractOcrEngine()

        confidence = engine._mean_confidence(
            np.array([[0, 255]], dtype=np.uint8),
            "--oem 3 --psm 6",
        )

        self.assertIsNone(confidence)


if __name__ == "__main__":
    unittest.main()
