from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract


@dataclass(frozen=True)
class OcrText:
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class TesseractConfig:
    lang: str = "eng"
    psm: int = 6
    oem: int = 3
    extra_config: str = ""

    def to_config_string(self) -> str:
        parts = [f"--oem {self.oem}", f"--psm {self.psm}"]
        if self.extra_config:
            parts.append(self.extra_config)
        return " ".join(parts)


class TesseractOcrEngine:
    """Small wrapper around pytesseract for cell OCR."""

    def __init__(self, config: TesseractConfig | None = None):
        self.config = config or TesseractConfig()

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def recognize(self, image: np.ndarray) -> OcrText:
        prepared = self._prepare_image(image)
        config = self.config.to_config_string()
        text = pytesseract.image_to_string(
            prepared,
            lang=self.config.lang,
            config=config,
        ).strip()
        confidence = self._mean_confidence(prepared, config)
        return OcrText(text=text, confidence=confidence)

    def _mean_confidence(self, image: np.ndarray, config: str) -> float | None:
        data = pytesseract.image_to_data(
            image,
            lang=self.config.lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        values: list[float] = []
        for value in data.get("conf", []):
            try:
                confidence = float(value)
            except (TypeError, ValueError):
                continue
            if confidence >= 0:
                values.append(confidence)
        if not values:
            return None
        return float(sum(values) / len(values))
