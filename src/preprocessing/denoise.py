from __future__ import annotations

import cv2
import numpy as np

from src.core.image import DocumentImage
from src.core.stage import PipelineStage


class MorphologicalDenoiseStage(PipelineStage):
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size

    def process(self, doc: DocumentImage) -> DocumentImage:
        image = doc.image

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        cleaned = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        metadata = dict(doc.metadata)
        metadata["denoised"] = True

        return DocumentImage(cleaned, metadata)
