from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import OcrText


@dataclass(frozen=True)
class TrOcrConfig:
    model_name: str = "microsoft/trocr-base-handwritten"
    device: str | None = None
    max_new_tokens: int = 64


class TrOcrHandwrittenEngine:
    """Handwritten OCR backend for cropped cell images using Microsoft TrOCR."""

    def __init__(self, config: TrOcrConfig | None = None):
        self.config = config or TrOcrConfig()
        try:
            import torch
            from PIL import Image
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except Exception as exc:  # pragma: no cover - depends on optional packages
            raise RuntimeError(
                "TrOCR OCR requires optional handwritten OCR dependencies. "
                "Install FarmAI's HTR extra plus a PyTorch build for your "
                "machine. For CPU-only Linux installs, use PyTorch's CPU wheel "
                "index to avoid downloading CUDA/NVIDIA packages."
            ) from exc

        self._torch = torch
        self._image_cls = Image
        self.processor = TrOCRProcessor.from_pretrained(self.config.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.config.model_name)
        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    def _prepare_image(self, image: np.ndarray):
        array = np.asarray(image)
        if array.ndim == 2:
            rgb = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        elif array.ndim == 3 and array.shape[2] == 3:
            rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        elif array.ndim == 3 and array.shape[2] == 4:
            rgb = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError("TrOCR expects a 2D grayscale or 3/4-channel image")

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return self._image_cls.fromarray(rgb)

    def recognize(self, image: np.ndarray) -> OcrText:
        prepared = self._prepare_image(image)
        pixel_values = self.processor(
            images=prepared,
            return_tensors="pt",
        ).pixel_values.to(self.device)
        with self._torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=self.config.max_new_tokens,
            )
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()
        return OcrText(text=text, confidence=None)
