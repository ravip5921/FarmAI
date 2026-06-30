from __future__ import annotations

from dataclasses import dataclass

from .base import CellOcrEngine
from .image_preprocessing import CellImagePreprocessConfig
from .tesseract_engine import TesseractConfig, TesseractOcrEngine
from .trocr_engine import TrOcrConfig, TrOcrHandwrittenEngine

DEFAULT_OCR_ENGINE = "tesseract"


@dataclass(frozen=True)
class OcrEngineSpec:
    name: str
    label: str
    description: str


_OCR_ENGINE_SPECS = [
    OcrEngineSpec(
        name="tesseract",
        label="Tesseract",
        description="Fast classical OCR baseline for printed or very clean text.",
    ),
    OcrEngineSpec(
        name="trocr-handwritten",
        label="TrOCR handwritten",
        description="Transformer OCR model for cropped handwritten text images.",
    ),
]


def get_ocr_engine_specs() -> list[OcrEngineSpec]:
    return list(_OCR_ENGINE_SPECS)


def get_ocr_engine_names() -> list[str]:
    return [spec.name for spec in _OCR_ENGINE_SPECS]


def create_ocr_engine(
    name: str = DEFAULT_OCR_ENGINE,
    **kwargs,
) -> CellOcrEngine:
    normalized = name.strip().lower()
    preprocess = kwargs.get("preprocess")
    if preprocess is None:
        preprocess = CellImagePreprocessConfig()
    if normalized == "tesseract":
        return TesseractOcrEngine(
            TesseractConfig(
                lang=kwargs.get("lang", "eng"),
                psm=int(kwargs.get("psm", 13)),
                oem=int(kwargs.get("oem", 3)),
                extra_config=kwargs.get("extra_config", ""),
                preprocess=preprocess,
            )
        )
    if normalized in {"trocr", "trocr-handwritten"}:
        return TrOcrHandwrittenEngine(
            TrOcrConfig(
                model_name=kwargs.get(
                    "model_name",
                    "microsoft/trocr-base-handwritten",
                ),
                device=kwargs.get("device"),
                max_new_tokens=int(kwargs.get("max_new_tokens", 64)),
                preprocess=preprocess,
            )
        )

    choices = ", ".join(get_ocr_engine_names())
    raise ValueError(f"Unknown OCR engine '{name}'. Available engines: {choices}")
