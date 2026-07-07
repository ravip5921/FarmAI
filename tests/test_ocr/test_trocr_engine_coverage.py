from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.ocr.trocr_engine import TrOcrConfig, TrOcrHandwrittenEngine
from tests.test_ocr.test_trocr_engine import _fake_modules


class TestTrOcrEngineCoverage(unittest.TestCase):
    def _engine(self) -> TrOcrHandwrittenEngine:
        with patch.dict("sys.modules", _fake_modules()):
            return TrOcrHandwrittenEngine(TrOcrConfig(device="cpu"))

    def test_prepare_image_converts_bgr_and_bgra_from_preprocessor_output(self) -> None:
        engine = self._engine()

        with patch(
            "src.ocr.trocr_engine.prepare_cell_image_for_ocr",
            return_value=np.zeros((1, 1, 3), dtype=np.uint8),
        ):
            bgr = engine._prepare_image(np.zeros((1, 1), dtype=np.uint8))
        with patch(
            "src.ocr.trocr_engine.prepare_cell_image_for_ocr",
            return_value=np.zeros((1, 1, 4), dtype=np.uint8),
        ):
            bgra = engine._prepare_image(np.zeros((1, 1), dtype=np.uint8))

        self.assertEqual(bgr[1].shape, (1, 1, 3))
        self.assertEqual(bgra[1].shape, (1, 1, 3))

    def test_prepare_image_rejects_bad_preprocessor_shape_and_clips_rgb(self) -> None:
        engine = self._engine()

        with patch(
            "src.ocr.trocr_engine.prepare_cell_image_for_ocr",
            return_value=np.array([[[300.0, -1.0, 5.0]]], dtype=np.float32),
        ):
            prepared = engine._prepare_image(np.zeros((1, 1), dtype=np.uint8))
        with patch(
            "src.ocr.trocr_engine.prepare_cell_image_for_ocr",
            return_value=np.zeros((1, 1, 2), dtype=np.uint8),
        ):
            with self.assertRaisesRegex(ValueError, "2D grayscale"):
                engine._prepare_image(np.zeros((1, 1), dtype=np.uint8))

        self.assertEqual(prepared[1].dtype, np.uint8)
        self.assertEqual(prepared[1].tolist(), [[[5, 0, 255]]])


if __name__ == "__main__":
    unittest.main()
