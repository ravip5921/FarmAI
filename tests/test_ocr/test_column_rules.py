from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.ocr import OcrText
from src.ocr.column_rules import (
    TEMPERATURE_PATTERN,
    ColumnOcrRule,
    build_column_ocr_rules,
    recognize_with_column_rule,
)
from src.ocr.tesseract_engine import TesseractConfig, TesseractOcrEngine
from src.templates import TemplateColumn


class SequenceEngine:
    def __init__(self, values: list[str]) -> None:
        self.values = values
        self.calls = 0

    def recognize(self, image: np.ndarray) -> OcrText:
        value = self.values[self.calls]
        self.calls += 1
        return OcrText(text=value, confidence=80.0 + self.calls)


class TestColumnOcrRules(unittest.TestCase):
    def test_temperature_rule_accepts_one_to_three_digit_values(self) -> None:
        rule = ColumnOcrRule(
            index=1,
            key="current_temperature",
            value_type="temperature",
            pattern=TEMPERATURE_PATTERN,
        )

        for value in ("1", "1.2", "1.23", "88", "88.1", "88.12", "100", "100.1"):
            with self.subTest(value=value):
                valid, text, error = rule.validate(value)
                self.assertTrue(valid)
                self.assertEqual(text, value)
                self.assertIsNone(error)

    def test_temperature_rule_applies_range_after_pattern_match(self) -> None:
        rule = ColumnOcrRule(
            index=1,
            key="current_temperature",
            value_type="temperature",
            pattern=TEMPERATURE_PATTERN,
            range_min=80.0,
            range_max=120.0,
        )

        valid, text, error = rule.validate("1")

        self.assertFalse(valid)
        self.assertEqual(text, "1")
        self.assertEqual(error, "value below minimum 80")

    def test_temperature_rule_rejects_letters_and_out_of_range_values(self) -> None:
        rule = ColumnOcrRule(
            index=1,
            key="current_temperature",
            value_type="temperature",
            pattern=TEMPERATURE_PATTERN,
            range_min=80.0,
            range_max=120.0,
        )

        for value in ("8A.1", "All good", "88l", "121.0"):
            with self.subTest(value=value):
                valid, _text, error = rule.validate(value)
                self.assertFalse(valid)
                self.assertTrue(error)

    def test_recognize_with_column_rule_blanks_invalid_text(self) -> None:
        rule = ColumnOcrRule(
            index=1,
            key="current_temperature",
            value_type="temperature",
            pattern=TEMPERATURE_PATTERN,
        )
        engine = SequenceEngine(["8A.1"])

        result = recognize_with_column_rule(
            engine,
            np.zeros((1, 1), dtype=np.uint8),
            rule,
        )

        self.assertEqual(result.text, "")
        self.assertEqual(result.raw_text, "8A.1")
        self.assertIn("temperature", result.validation_error or "")

    def test_tesseract_rule_uses_numeric_whitelist_and_retries_until_valid(
        self,
    ) -> None:
        rule = ColumnOcrRule(
            index=1,
            key="current_temperature",
            value_type="temperature",
            pattern=TEMPERATURE_PATTERN,
            tesseract_whitelist="0123456789.",
            tesseract_retry_psms=(7, 8),
        )
        engine = TesseractOcrEngine(TesseractConfig(psm=13, extra_config="-c old=1"))
        seen_configs: list[TesseractConfig] = []

        def fake_recognize(ocr_engine, image):
            seen_configs.append(ocr_engine.config)
            if len(seen_configs) == 1:
                return OcrText(text="abc", confidence=20.0)
            return OcrText(text="88.1", confidence=90.0)

        with patch.object(TesseractOcrEngine, "recognize", fake_recognize):
            result = recognize_with_column_rule(
                engine,
                np.zeros((1, 1), dtype=np.uint8),
                rule,
            )

        self.assertEqual(result.text, "88.1")
        self.assertEqual([config.psm for config in seen_configs], [13, 7])
        for config in seen_configs:
            self.assertIn("tessedit_char_whitelist=0123456789.", config.extra_config)
            self.assertIn("classify_bln_numeric_mode=1", config.extra_config)

    def test_build_column_ocr_rules_uses_template_column_context(self) -> None:
        rules = build_column_ocr_rules(
            [
                TemplateColumn(index=0, key="date", name="Date"),
                TemplateColumn(
                    index=1,
                    key="current_temperature",
                    name="Current Temperature",
                    value_type="temperature",
                    range_min=80.0,
                    range_max=120.0,
                ),
            ]
        )

        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0].index, 0)
        self.assertEqual(rules[0].name, "Date")
        self.assertIsNone(rules[0].tesseract_whitelist)
        self.assertEqual(rules[1].index, 1)
        self.assertEqual(rules[1].name, "Current Temperature")
        self.assertEqual(rules[1].range_min, 80.0)
        self.assertEqual(rules[1].tesseract_whitelist, "0123456789.")


if __name__ == "__main__":
    unittest.main()
