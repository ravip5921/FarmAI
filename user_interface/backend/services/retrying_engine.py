from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from src.ocr.base import CellOcrEngine, OcrText


class PermanentOcrError(RuntimeError):
    """An OCR configuration/authentication problem that retries cannot fix."""


class RetryingOcrEngine:
    """Retry transient per-cell failures while failing fast on permanent errors."""

    def __init__(
        self,
        engine: CellOcrEngine,
        *,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.engine = engine
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self.sleep = sleep

    def recognize(self, image: np.ndarray) -> OcrText:
        return self._run(lambda: self.engine.recognize(image))

    def recognize_with_rule(self, image: np.ndarray, *, rule: Any | None) -> OcrText:
        recognize_with_rule = getattr(self.engine, "recognize_with_rule", None)
        if callable(recognize_with_rule):
            return self._run(lambda: recognize_with_rule(image, rule=rule))
        return self.recognize(image)

    def _run(self, operation: Callable[[], OcrText]) -> OcrText:
        last_result: OcrText | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = operation()
            except Exception as exc:
                classification = _classify_error(str(exc))
                if classification == "permanent":
                    raise PermanentOcrError(str(exc)) from exc
                if classification != "transient" or attempt == self.max_attempts:
                    raise
            else:
                last_result = result
                classification = _classify_error(result.validation_error or "")
                if classification == "permanent":
                    raise PermanentOcrError(result.validation_error or "OCR failed")
                if classification != "transient" or attempt == self.max_attempts:
                    return result
            self.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        if last_result is not None:  # pragma: no cover - loop always exits above
            return last_result
        raise RuntimeError(  # pragma: no cover - defensive type-checking fallback
            "OCR retry loop ended without a result"
        )


def _classify_error(message: str) -> str:
    normalized = message.casefold()
    if not normalized:
        return "other"
    permanent_markers = (
        "not configured",
        "authorization",
        "unauthorized",
        "forbidden",
        "http error 401",
        "http error 403",
    )
    if any(marker in normalized for marker in permanent_markers):
        return "permanent"
    transient_markers = (
        "request failed",
        "timed out",
        "timeout",
        "temporary",
        "connection",
        "rate limit",
        "http error 429",
        "http error 500",
        "http error 502",
        "http error 503",
        "http error 504",
    )
    return (
        "transient"
        if any(marker in normalized for marker in transient_markers)
        else "other"
    )
