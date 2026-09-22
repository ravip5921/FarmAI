from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.ocr.base import OcrText
from user_interface.backend.services.checkpoint_engine import (
    CheckpointingOcrEngine,
    _checkpoint_key,
    _read_checkpoint,
    _rule_payload,
)


class _Engine:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image: np.ndarray) -> OcrText:
        self.calls += 1
        return OcrText(text="plain", confidence=91.0)

    def recognize_with_rule(self, image: np.ndarray, *, rule) -> OcrText:
        self.calls += 1
        return OcrText(text=rule.key, raw_text="raw")


class _PlainEngine:
    def recognize(self, image: np.ndarray) -> OcrText:
        return OcrText(text="fallback")


@dataclass(frozen=True)
class _Rule:
    key: str


class TestCheckpointingOcrEngine(unittest.TestCase):
    def test_reuses_plain_and_rule_results_across_wrappers(self) -> None:
        image = np.arange(9, dtype=np.uint8).reshape((3, 3))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            first_engine = _Engine()
            first = CheckpointingOcrEngine(first_engine, path)
            self.assertEqual(first.recognize(image).text, "plain")
            self.assertEqual(
                first.recognize_with_rule(image, rule=_Rule("temperature")).text,
                "temperature",
            )
            self.assertEqual(first_engine.calls, 2)

            second_engine = _Engine()
            second = CheckpointingOcrEngine(second_engine, path)
            self.assertEqual(second.recognize(image).confidence, 91.0)
            cached_rule = second.recognize_with_rule(
                image, rule=_Rule("temperature")
            )
            self.assertEqual(cached_rule.raw_text, "raw")
            self.assertEqual(second_engine.calls, 0)
            self.assertEqual(len(list(path.glob("*.json"))), 2)

    def test_rule_wrapper_falls_back_to_plain_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapped = CheckpointingOcrEngine(_PlainEngine(), Path(tmpdir))
            result = wrapped.recognize_with_rule(
                np.zeros((2, 2), dtype=np.uint8), rule=object()
            )
        self.assertEqual(result.text, "fallback")
        self.assertIn("index", _rule_payload(object()))
        self.assertIsNone(_rule_payload(None))

    def test_transient_failure_is_not_checkpointed_and_bad_files_are_ignored(
        self,
    ) -> None:
        image = np.zeros((1, 1), dtype=np.uint8)

        class FailingThenSuccessful:
            def __init__(self) -> None:
                self.calls = 0

            def recognize(self, value: np.ndarray) -> OcrText:
                self.calls += 1
                if self.calls == 1:
                    return OcrText(text="", validation_error="request failed: timeout")
                return OcrText(text="ok")

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            engine = FailingThenSuccessful()
            wrapped = CheckpointingOcrEngine(engine, directory)
            self.assertTrue(wrapped.recognize(image).validation_error)
            self.assertEqual(list(directory.glob("*.json")), [])
            self.assertEqual(wrapped.recognize(image).text, "ok")
            self.assertEqual(engine.calls, 2)

            key = _checkpoint_key(image, {"method": "recognize"})
            checkpoint = directory / f"{key}.json"
            checkpoint.write_text("not-json", encoding="utf-8")
            self.assertIsNone(_read_checkpoint(checkpoint, expected_key=key))
            checkpoint.write_text(
                json.dumps({"version": 2, "key": key, "text": "old"}),
                encoding="utf-8",
            )
            self.assertIsNone(_read_checkpoint(checkpoint, expected_key=key))
            checkpoint.write_text(
                json.dumps({"version": 1, "key": key, "confidence": "bad"}),
                encoding="utf-8",
            )
            self.assertIsNone(_read_checkpoint(checkpoint, expected_key=key))


if __name__ == "__main__":
    unittest.main()
