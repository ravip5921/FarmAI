from __future__ import annotations

import cv2

from src.core.image import DocumentImage
from src.templates import FormTemplate


def prepare_template_image(
    page: DocumentImage, template: FormTemplate | None
) -> DocumentImage:
    """Normalize opted-in large photographs before both detection and OCR."""
    limit = template.preprocessing_max_width if template else None
    height, width = page.image.shape[:2]
    if limit is None or width <= limit:
        return page
    resized = cv2.resize(
        page.image,
        (limit, max(1, round(height * limit / width))),
        interpolation=cv2.INTER_AREA,
    )
    return DocumentImage(
        resized, {**page.metadata, "template_image_scale": limit / width}
    )
