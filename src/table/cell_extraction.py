from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid_reconstruction import GridCell, GridStructure


@dataclass(frozen=True)
class ExtractedCell:
    row: int
    col: int
    bbox: tuple[int, int, int, int]
    image: np.ndarray


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    padding: int = 0,
) -> tuple[int, int, int, int] | None:
    height, width = image_shape[:2]
    x, y, w, h = bbox
    pad = max(0, int(padding))

    left = max(0, int(x) + pad)
    top = max(0, int(y) + pad)
    right = min(width, int(x + w) - pad)
    bottom = min(height, int(y + h) - pad)

    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def crop_cell(
    image: np.ndarray,
    cell: GridCell,
    *,
    padding: int = 2,
) -> ExtractedCell | None:
    """Crop one grid cell from an image, clipping safely to image bounds."""
    clipped = _clip_bbox(cell.bbox, image.shape, padding=padding)
    if clipped is None:
        return None

    x, y, w, h = clipped
    return ExtractedCell(
        row=cell.row,
        col=cell.col,
        bbox=clipped,
        image=image[y : y + h, x : x + w].copy(),
    )


def extract_cell_images(
    image: np.ndarray,
    grid: GridStructure,
    *,
    padding: int = 2,
) -> list[ExtractedCell]:
    """Crop every valid cell in row-major order."""
    cells: list[ExtractedCell] = []
    for cell in sorted(grid.cells, key=lambda item: (item.row, item.col)):
        extracted = crop_cell(image, cell, padding=padding)
        if extracted is not None:
            cells.append(extracted)
    return cells
