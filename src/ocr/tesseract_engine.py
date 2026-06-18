from __future__ import annotations

import shutil
from dataclasses import dataclass

import cv2
import numpy as np

from .base import OcrText

try:  # pragma: no cover - import availability depends on local environment
    import pytesseract
except Exception:  # pragma: no cover - handled at OCR runtime

    class _UnavailablePytesseract:
        class Output:
            DICT = "dict"

        def image_to_string(self, *args, **kwargs):
            raise RuntimeError(
                "pytesseract is not installed. Install project dependencies and "
                "make sure the Tesseract OCR executable is available on PATH."
            )

        def image_to_data(self, *args, **kwargs):
            raise RuntimeError(
                "pytesseract is not installed. Install project dependencies and "
                "make sure the Tesseract OCR executable is available on PATH."
            )

    pytesseract = _UnavailablePytesseract()


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
        executable = shutil.which("tesseract")
        if executable is None:
            raise RuntimeError(
                "Tesseract OCR executable was not found on PATH. Install the "
                "`tesseract-ocr` system package or use another OCR backend such "
                "as `--ocr-engine trocr-handwritten`."
            )
        prepared = self._prepare_image(image)
        config = self.config.to_config_string()
        try:
            text = pytesseract.image_to_string(
                prepared,
                lang=self.config.lang,
                config=config,
            ).strip()
            confidence = self._mean_confidence(prepared, config)
        except PermissionError as exc:
            raise RuntimeError(
                f"Tesseract OCR executable is not runnable: {executable}. Check "
                "that it is the real tesseract binary and has execute "
                "permission, or use `--ocr-engine trocr-handwritten`."
            ) from exc
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
