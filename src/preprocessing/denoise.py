from __future__ import annotations

import cv2
import numpy as np

from src.core.image import DocumentImage
from src.core.stage import PipelineStage


class MorphologicalDenoiseStage(PipelineStage):
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size

    def _is_binary(self, image: np.ndarray) -> bool:
        if image.ndim != 2:
            return False
        unique = np.unique(image)
        return len(unique) <= 2

    def _remove_small_foreground_components(self, image: np.ndarray) -> np.ndarray:
        dark = image < 128
        light = ~dark
        foreground_is_dark = int(np.count_nonzero(dark)) <= int(np.count_nonzero(light))
        foreground = dark if foreground_is_dark else light
        foreground_mask = foreground.astype(np.uint8) * 255

        min_area = max(1, self.kernel_size)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground_mask,
            connectivity=8,
        )

        cleaned = image.copy()
        replacement = 255 if foreground_is_dark else 0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                cleaned[labels == label] = replacement

        return cleaned

    def process(self, doc: DocumentImage) -> DocumentImage:
        image = doc.image

        if self.kernel_size <= 1:
            cleaned = image.copy()
            method = "none"
        elif self._is_binary(image):
            cleaned = self._remove_small_foreground_components(image)
            method = "connected_components"
        else:
            kernel_size = self.kernel_size if self.kernel_size % 2 == 1 else self.kernel_size + 1
            cleaned = cv2.medianBlur(image, kernel_size)
            method = "median_blur"

        metadata = dict(doc.metadata)
        metadata["denoised"] = True
        metadata["denoise_method"] = method

        return DocumentImage(cleaned, metadata)
