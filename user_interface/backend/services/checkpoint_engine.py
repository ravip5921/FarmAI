from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.ocr.base import CellOcrEngine, OcrText


class CheckpointingOcrEngine:
    """Persist completed cell OCR calls so an interrupted job can resume safely."""

    def __init__(self, engine: CellOcrEngine, checkpoint_dir: Path):
        self.engine = engine
        self.checkpoint_dir = checkpoint_dir

    def recognize(self, image: np.ndarray) -> OcrText:
        return self._recognize(
            image,
            context={"method": "recognize"},
            operation=lambda: self.engine.recognize(image),
        )

    def recognize_with_rule(self, image: np.ndarray, *, rule: Any | None) -> OcrText:
        context = {"method": "recognize_with_rule", "rule": _rule_payload(rule)}

        def operation() -> OcrText:
            recognize_with_rule = getattr(self.engine, "recognize_with_rule", None)
            if callable(recognize_with_rule):
                return recognize_with_rule(image, rule=rule)
            return self.engine.recognize(image)

        return self._recognize(image, context=context, operation=operation)

    def _recognize(
        self,
        image: np.ndarray,
        *,
        context: dict[str, Any],
        operation: Callable[[], OcrText],
    ) -> OcrText:
        key = _checkpoint_key(image, context)
        path = self.checkpoint_dir / f"{key}.json"
        cached = _read_checkpoint(path, expected_key=key)
        if cached is not None:
            return cached

        result = operation()
        if _should_checkpoint(result):
            _write_checkpoint(path, key=key, result=result)
        return result


def _rule_payload(rule: Any | None) -> Any:
    if rule is None:
        return None
    if is_dataclass(rule) and not isinstance(rule, type):
        return asdict(rule)
    return {
        name: getattr(rule, name, None)
        for name in (
            "index",
            "key",
            "name",
            "value_type",
            "format",
            "pattern",
            "range_min",
            "range_max",
            "common_values",
        )
    }


def _checkpoint_key(image: np.ndarray, context: dict[str, Any]) -> str:
    array = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    digest.update(
        json.dumps(context, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def _read_checkpoint(path: Path, *, expected_key: str) -> OcrText | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if payload.get("version") != 1 or payload.get("key") != expected_key:
        return None
    try:
        confidence = payload.get("confidence")
        return OcrText(
            text=str(payload["text"]),
            confidence=float(confidence) if confidence is not None else None,
            raw_text=(
                str(payload["raw_text"]) if payload.get("raw_text") is not None else None
            ),
            validation_error=(
                str(payload["validation_error"])
                if payload.get("validation_error") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _should_checkpoint(result: OcrText) -> bool:
    if result.text.strip() or not result.validation_error:
        return True
    error = result.validation_error.casefold()
    transient_or_configuration_markers = (
        "request failed",
        "timed out",
        "timeout",
        "temporary",
        "connection",
        "rate limit",
        "not configured",
        "authorization",
        "forbidden",
    )
    return not any(marker in error for marker in transient_or_configuration_markers)


def _write_checkpoint(path: Path, *, key: str, result: OcrText) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "key": key,
        "text": result.text,
        "confidence": result.confidence,
        "raw_text": result.raw_text,
        "validation_error": result.validation_error,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
