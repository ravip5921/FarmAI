from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .image import DocumentImage


def _extract_image(image: Any) -> Any:
    if isinstance(image, DocumentImage):
        return image.image
    return image


def show(title: str, image: Any) -> None:
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


def save_debug(path: str | Path, title: str, image: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    array = _extract_image(image)
    plt.figure(figsize=(10, 8))
    if isinstance(array, np.ndarray) and array.ndim == 2:
        plt.imshow(array, cmap="gray")
    else:
        plt.imshow(array)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()