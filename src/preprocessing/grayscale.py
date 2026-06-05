from __future__ import annotations

import cv2

from src.core.image import DocumentImage
from src.core.stage import PipelineStage


class GrayscaleStage(PipelineStage):
    def process(self, doc: DocumentImage) -> DocumentImage:
        image = doc.image

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        metadata = dict(doc.metadata)
        metadata["grayscale"] = True

        return DocumentImage(gray, metadata)
