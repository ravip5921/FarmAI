from __future__ import annotations

import unittest
from unittest.mock import patch

from src.ocr.registry import (
    DEFAULT_OCR_ENGINE,
    create_ocr_engine,
    get_ocr_engine_names,
    get_ocr_engine_specs,
)
from src.ocr.tesseract_engine import TesseractOcrEngine


class TestOcrRegistry(unittest.TestCase):
    def test_registry_exposes_supported_engines(self) -> None:
        names = get_ocr_engine_names()
        specs = get_ocr_engine_specs()

        self.assertEqual(DEFAULT_OCR_ENGINE, "tesseract")
        self.assertIn("tesseract", names)
        self.assertIn("trocr-handwritten", names)
        self.assertEqual(names, [spec.name for spec in specs])

    def test_create_tesseract_engine(self) -> None:
        engine = create_ocr_engine("tesseract", psm=7, oem=1)

        self.assertIsInstance(engine, TesseractOcrEngine)
        self.assertEqual(engine.config.psm, 7)
        self.assertEqual(engine.config.oem, 1)

    def test_create_tesseract_engine_defaults_to_raw_line_psm(self) -> None:
        engine = create_ocr_engine("tesseract")

        self.assertIsInstance(engine, TesseractOcrEngine)
        self.assertEqual(engine.config.psm, 13)

    def test_create_trocr_engine_passes_model_options(self) -> None:
        with patch("src.ocr.registry.TrOcrHandwrittenEngine") as engine_cls:
            create_ocr_engine(
                "trocr-handwritten",
                model_name="custom/model",
                device="cpu",
                max_new_tokens=32,
            )

        config = engine_cls.call_args.args[0]
        self.assertEqual(config.model_name, "custom/model")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.max_new_tokens, 32)

    def test_unknown_engine_raises_useful_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown OCR engine"):
            create_ocr_engine("nope")


if __name__ == "__main__":
    unittest.main()
