from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.ocr.base import OcrText
from src.ocr.column_rules import (
    ColumnOcrRule,
    _best_result,
    _iter_recognize_candidates,
    recognize_with_column_rule,
)
from src.ocr.tesseract_engine import TesseractConfig, TesseractOcrEngine


class TestColumnRulesCoverage(unittest.TestCase):
    def test_validate_reports_numeric_error_when_range_requires_number(self) -> None:
        rule = ColumnOcrRule(
            index=0,
            key="temp",
            value_type="temperature",
            range_min=80,
        )

        valid, text, error = rule.validate("abc")

        self.assertFalse(valid)
        self.assertEqual(text, "abc")
        self.assertEqual(error, "expected numeric temperature")

    def test_tesseract_configs_without_whitelist_returns_base_config(self) -> None:
        rule = ColumnOcrRule(index=0, key="note", value_type="text")
        config = TesseractConfig(psm=7)

        self.assertEqual(rule.tesseract_configs(config), [config])

    def test_tesseract_candidate_iterator_stops_after_tesseract_retries(self) -> None:
        engine = TesseractOcrEngine(TesseractConfig(psm=6))
        rule = ColumnOcrRule(
            index=0,
            key="temp",
            value_type="temperature",
            pattern=r"^\d+$",
            tesseract_whitelist="0123456789",
            tesseract_retry_psms=(7,),
        )

        with patch.object(
            TesseractOcrEngine,
            "recognize",
            side_effect=[OcrText(text="bad"), OcrText(text="101")],
        ) as recognize:
            result = recognize_with_column_rule(
                engine,
                np.zeros((1, 1), dtype=np.uint8),
                rule,
            )

        self.assertEqual(result.text, "101")
        self.assertEqual(recognize.call_count, 2)

    def test_tesseract_candidate_iterator_can_be_fully_exhausted(self) -> None:
        engine = TesseractOcrEngine(TesseractConfig(psm=6))
        rule = ColumnOcrRule(
            index=0,
            key="temp",
            value_type="temperature",
            tesseract_whitelist="0123456789",
            tesseract_retry_psms=(7,),
        )

        with patch.object(
            TesseractOcrEngine,
            "recognize",
            side_effect=[OcrText(text="bad"), OcrText(text="also bad")],
        ):
            results = list(
                _iter_recognize_candidates(
                    engine,
                    np.zeros((1, 1), dtype=np.uint8),
                    rule,
                )
            )

        self.assertEqual([result.text for result in results], ["bad", "also bad"])

    def test_best_result_handles_empty_candidate_list(self) -> None:
        result = _best_result([])

        self.assertEqual(result, OcrText(text="", confidence=None))


if __name__ == "__main__":
    unittest.main()
