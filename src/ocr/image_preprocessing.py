from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CellImagePreprocessConfig:
    """Normalize a cropped table cell into a line-like OCR image."""

    crop_to_ink: bool = True
    ink_threshold: int = 200
    content_padding: int = 8
    border: int = 12
    target_height: int | None = 64
    max_scale: float = 4.0


def to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 3 and array.shape[2] == 4:
        gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("OCR expects a 2D grayscale or 3/4-channel image")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def ensure_light_background(image: np.ndarray) -> np.ndarray:
    """Return an image with dark foreground ink on a light background."""
    gray = to_grayscale_uint8(image)
    if float(np.mean(gray)) < 127.0:
        return 255 - gray
    return gray


def _content_bbox(
    image: np.ndarray,
    *,
    ink_threshold: int,
    min_pixels: int = 2,
) -> tuple[int, int, int, int] | None:
    ink = image < int(np.clip(ink_threshold, 0, 255))
    if int(np.count_nonzero(ink)) < min_pixels:
        return None

    ys, xs = np.where(ink)
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    return left, top, right, bottom


def _crop_with_padding(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    left, top, right, bottom = bbox
    pad = max(0, int(padding))
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(width, right + pad)
    bottom = min(height, bottom + pad)
    return image[top:bottom, left:right]


def _resize_to_target_height(
    image: np.ndarray,
    *,
    target_height: int | None,
    max_scale: float,
) -> np.ndarray:
    if target_height is None or target_height <= 0 or image.shape[0] <= 0:
        return image

    scale = min(float(max_scale), float(target_height) / float(image.shape[0]))
    if abs(scale - 1.0) < 0.05:
        return image

    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(image, (width, height), interpolation=interpolation)


def prepare_cell_image_for_ocr(
    image: np.ndarray,
    config: CellImagePreprocessConfig | None = None,
) -> np.ndarray:
    """Prepare a single cropped cell for OCR without page-layout context."""
    config = config or CellImagePreprocessConfig()
    prepared = ensure_light_background(image)

    if config.crop_to_ink:
        bbox = _content_bbox(prepared, ink_threshold=config.ink_threshold)
        if bbox is not None:
            prepared = _crop_with_padding(
                prepared,
                bbox,
                padding=config.content_padding,
            )

    prepared = _resize_to_target_height(
        prepared,
        target_height=config.target_height,
        max_scale=max(1.0, float(config.max_scale)),
    )

    border = max(0, int(config.border))
    if border:
        prepared = cv2.copyMakeBorder(
            prepared,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=255,
        )
    return prepared
