from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .image import DocumentImage


def _extract_image(image: Any) -> Any:  # pragma: no cover
    if isinstance(image, DocumentImage):
        return image.image
    return image


def show(title: str, image: Any) -> None:  # pragma: no cover
    import matplotlib.pyplot as plt

    array = _extract_image(image)
    plt.figure(figsize=(10, 8))
    if isinstance(array, np.ndarray) and array.ndim == 2:
        plt.imshow(array, cmap="gray")
    else:
        plt.imshow(array)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def _prepare_for_write(array: Any) -> np.ndarray:  # pragma: no cover
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)

    if array.dtype == bool:
        return array.astype(np.uint8) * 255

    if array.dtype == np.uint8:
        return array

    return np.clip(array, 0, 255).astype(np.uint8)


def save_debug(path: str | Path, title: str, image: Any) -> None:  # pragma: no cover
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    array = _prepare_for_write(_extract_image(image))
    success = cv2.imwrite(str(output_path), array)
    if not success:
        raise OSError(f"Could not write debug image: {output_path}")
