from __future__ import annotations

import unittest

import numpy as np

from src.ocr.base import OcrText
from user_interface.backend.services.retrying_engine import (
    PermanentOcrError,
    RetryingOcrEngine,
    _classify_error,
)


class _SequenceEngine:
    def __init__(self, values) -> None:
        self.values = iter(values)

    def recognize(self, image: np.ndarray) -> OcrText:
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value

    def recognize_with_rule(self, image: np.ndarray, *, rule) -> OcrText:
        return self.recognize(image)


class _PlainEngine:
    def recognize(self, image: np.ndarray) -> OcrText:
        return OcrText(text="plain")


class TestRetryingOcrEngine(unittest.TestCase):
    def test_retries_transient_results_with_backoff(self) -> None:
        sleeps = []
        wrapped = RetryingOcrEngine(
            _SequenceEngine(
                [
                    OcrText(text="", validation_error="request failed: timed out"),
                    OcrText(text="88"),
                ]
            ),
            max_attempts=3,
            backoff_seconds=0.25,
            sleep=sleeps.append,
        )
        result = wrapped.recognize(np.zeros((1, 1), dtype=np.uint8))
        self.assertEqual(result.text, "88")
        self.assertEqual(sleeps, [0.25])

    def test_returns_last_transient_result_after_bound(self) -> None:
        failure = OcrText(text="", validation_error="connection timeout")
        wrapped = RetryingOcrEngine(
            _SequenceEngine([failure, failure]),
            max_attempts=2,
            backoff_seconds=-1,
            sleep=lambda value: None,
        )
        self.assertIs(wrapped.recognize(np.zeros((1, 1))), failure)

    def test_retries_transient_exception_then_uses_rule_method(self) -> None:
        wrapped = RetryingOcrEngine(
            _SequenceEngine([TimeoutError("timed out"), OcrText(text="ok")]),
            max_attempts=2,
            backoff_seconds=0,
            sleep=lambda value: None,
        )
        self.assertEqual(
            wrapped.recognize_with_rule(np.zeros((1, 1)), rule=None).text, "ok"
        )

    def test_permanent_result_and_exception_fail_fast(self) -> None:
        image = np.zeros((1, 1))
        wrapped_result = RetryingOcrEngine(
            _SequenceEngine(
                [OcrText(text="", validation_error="LLM OCR is not configured")]
            )
        )
        with self.assertRaises(PermanentOcrError):
            wrapped_result.recognize(image)

        wrapped_exception = RetryingOcrEngine(
            _SequenceEngine([RuntimeError("HTTP Error 403: Forbidden")])
        )
        with self.assertRaises(PermanentOcrError):
            wrapped_exception.recognize(image)

    def test_non_transient_exception_is_not_retried(self) -> None:
        wrapped = RetryingOcrEngine(_SequenceEngine([ValueError("bad image")]))
        with self.assertRaises(ValueError):
            wrapped.recognize(np.zeros((1, 1)))
        self.assertEqual(_classify_error(""), "other")
        self.assertEqual(_classify_error("invalid response"), "other")
        self.assertEqual(_classify_error("rate limit"), "transient")

    def test_rule_falls_back_for_plain_engine(self) -> None:
        wrapped = RetryingOcrEngine(_PlainEngine(), max_attempts=0)
        self.assertEqual(
            wrapped.recognize_with_rule(np.zeros((1, 1)), rule=None).text, "plain"
        )


if __name__ == "__main__":
    unittest.main()
