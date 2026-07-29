from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .base import CellOcrEngine, OcrText
from .tesseract_engine import TesseractConfig, TesseractOcrEngine

TEMPERATURE_PATTERN = r"^\d{1,3}(?:\.\d{1,2})?$"


@dataclass(frozen=True)
class ColumnOcrRule:
    index: int
    key: str
    value_type: str
    name: str = ""
    format: str | None = None
    common_values: tuple[str, ...] = ()
    pattern: str | None = None
    range_min: float | None = None
    range_max: float | None = None
    tesseract_whitelist: str | None = None
    tesseract_retry_psms: tuple[int, ...] = ()

    def normalize_text(self, text: str) -> str:
        if self.value_type == "temperature":
            return "".join(text.strip().split())
        return text.strip()

    def validate(self, text: str) -> tuple[bool, str, str | None]:
        candidate = self.normalize_text(text)
        if self.pattern is not None and re.fullmatch(self.pattern, candidate) is None:
            return False, candidate, f"expected {self.value_type} pattern"
        if candidate and (self.range_min is not None or self.range_max is not None):
            try:
                value = float(candidate)
            except ValueError:
                return False, candidate, f"expected numeric {self.value_type}"
            if self.range_min is not None and value < self.range_min:
                return False, candidate, f"value below minimum {self.range_min:g}"
            if self.range_max is not None and value > self.range_max:
                return False, candidate, f"value above maximum {self.range_max:g}"
        return True, candidate, None

    def tesseract_configs(self, base_config: TesseractConfig) -> list[TesseractConfig]:
        if self.tesseract_whitelist is None:
            return [base_config]

        psms = _unique_ints((base_config.psm, *self.tesseract_retry_psms))
        extra_config = _append_tesseract_numeric_config(
            base_config.extra_config,
            self.tesseract_whitelist,
        )
        return [
            replace(base_config, psm=psm, extra_config=extra_config) for psm in psms
        ]


def build_column_ocr_rules(columns: Iterable[Any]) -> list[ColumnOcrRule]:
    rules: list[ColumnOcrRule] = []
    for column in columns:
        value_type = str(getattr(column, "value_type", "text"))
        is_temperature = value_type == "temperature"
        rules.append(
            ColumnOcrRule(
                index=int(getattr(column, "index")),
                key=str(getattr(column, "key", "")),
                name=str(getattr(column, "name", "")),
                value_type=value_type,
                format=getattr(column, "format", None),
                common_values=tuple(
                    str(value) for value in getattr(column, "common_values", ())
                ),
                pattern=TEMPERATURE_PATTERN if is_temperature else None,
                range_min=getattr(column, "range_min", None),
                range_max=getattr(column, "range_max", None),
                tesseract_whitelist="0123456789." if is_temperature else None,
                tesseract_retry_psms=(7, 8, 13) if is_temperature else (),
            )
        )
    return rules


def recognize_with_column_rule(
    engine: CellOcrEngine,
    image: np.ndarray,
    rule: ColumnOcrRule | None,
    *,
    llm_verifier: Any | None = None,
) -> OcrText:
    if rule is None:
        result = engine.recognize(image)
        return _maybe_verify_with_llm(
            result,
            image=image,
            rule=None,
            llm_verifier=llm_verifier,
        )
    recognize_with_rule = getattr(engine, "recognize_with_rule", None)
    if callable(recognize_with_rule):
        result = recognize_with_rule(image, rule=rule)
        if result.validation_error and not result.text.strip():
            return result
        valid, text, error = rule.validate(result.text)
        if valid:
            result = OcrText(
                text=text,
                confidence=result.confidence,
                raw_text=result.raw_text,
                validation_error=result.validation_error,
            )
            return _maybe_verify_with_llm(
                result,
                image=image,
                rule=rule,
                llm_verifier=llm_verifier,
            )
        result = OcrText(
            text="",
            confidence=result.confidence,
            raw_text=result.text if result.raw_text is None else result.raw_text,
            validation_error=error,
        )
        return _maybe_verify_with_llm(
            result,
            image=image,
            rule=rule,
            llm_verifier=llm_verifier,
        )

    results: list[OcrText] = []
    for result in _iter_recognize_candidates(engine, image, rule):
        results.append(result)
        valid, text, _error = rule.validate(result.text)
        if valid:
            result = OcrText(
                text=text,
                confidence=result.confidence,
                raw_text=result.text if text != result.text else result.raw_text,
                validation_error=None,
            )
            return _maybe_verify_with_llm(
                result,
                image=image,
                rule=rule,
                llm_verifier=llm_verifier,
            )

    best = _best_result(results)
    _valid, _text, error = rule.validate(best.text)
    result = OcrText(
        text="",
        confidence=best.confidence,
        raw_text=best.text if best.raw_text is None else best.raw_text,
        validation_error=error or f"invalid {rule.value_type}",
    )
    return _maybe_verify_with_llm(
        result,
        image=image,
        rule=rule,
        llm_verifier=llm_verifier,
    )


def _maybe_verify_with_llm(
    result: OcrText,
    *,
    image: np.ndarray,
    rule: ColumnOcrRule | None,
    llm_verifier: Any | None,
) -> OcrText:
    if llm_verifier is None or not llm_verifier.should_verify(result, rule):
        return result
    return llm_verifier.verify(image=image, rule=rule, ocr_text=result)


def _iter_recognize_candidates(
    engine: CellOcrEngine,
    image: np.ndarray,
    rule: ColumnOcrRule,
):
    if isinstance(engine, TesseractOcrEngine):
        for config in rule.tesseract_configs(engine.config):
            yield TesseractOcrEngine(config).recognize(image)
        return
    yield engine.recognize(image)


def _best_result(results: list[OcrText]) -> OcrText:
    if not results:
        return OcrText(text="", confidence=None)
    return max(
        results,
        key=lambda result: (
            result.confidence is not None,
            result.confidence if result.confidence is not None else -1.0,
        ),
    )


def _append_tesseract_numeric_config(extra_config: str, whitelist: str) -> str:
    numeric_config = (
        f"-c tessedit_char_whitelist={whitelist} -c classify_bln_numeric_mode=1"
    )
    return " ".join(part for part in (extra_config.strip(), numeric_config) if part)


def _unique_ints(values: Iterable[int]) -> tuple[int, ...]:
    unique: list[int] = []
    for value in values:
        value = int(value)
        if value not in unique:
            unique.append(value)
    return tuple(unique)
