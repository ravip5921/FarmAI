from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.table.cell_extraction import ExtractedCell, extract_cell_images
from src.table.grid_reconstruction import GridStructure

from .tesseract_engine import OcrText, TesseractOcrEngine


class CellOcrEngine(Protocol):
    def recognize(self, image: np.ndarray) -> OcrText: ...


@dataclass(frozen=True)
class OcrCell:
    row: int
    col: int
    bbox: tuple[int, int, int, int]
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class OcrTable:
    cells: list[OcrCell]
    row_count: int
    col_count: int

    def text_matrix(self, fill_value: str = "") -> list[list[str]]:
        matrix = [
            [fill_value for _ in range(self.col_count)] for _ in range(self.row_count)
        ]
        for cell in self.cells:
            if 0 <= cell.row < self.row_count and 0 <= cell.col < self.col_count:
                matrix[cell.row][cell.col] = cell.text
        return matrix


def recognize_extracted_cells(
    cells: list[ExtractedCell],
    *,
    engine: CellOcrEngine | None = None,
    row_count: int | None = None,
    col_count: int | None = None,
) -> OcrTable:
    ocr_engine = engine or TesseractOcrEngine()
    recognized: list[OcrCell] = []
    for cell in cells:
        result = ocr_engine.recognize(cell.image)
        recognized.append(
            OcrCell(
                row=cell.row,
                col=cell.col,
                bbox=cell.bbox,
                text=result.text,
                confidence=result.confidence,
            )
        )

    inferred_rows = max((cell.row for cell in recognized), default=-1) + 1
    inferred_cols = max((cell.col for cell in recognized), default=-1) + 1
    return OcrTable(
        cells=recognized,
        row_count=row_count if row_count is not None else inferred_rows,
        col_count=col_count if col_count is not None else inferred_cols,
    )


def recognize_table_cells(
    image: np.ndarray,
    grid: GridStructure,
    *,
    engine: CellOcrEngine | None = None,
    padding: int = 2,
) -> OcrTable:
    extracted_cells = extract_cell_images(image, grid, padding=padding)
    return recognize_extracted_cells(
        extracted_cells,
        engine=engine,
        row_count=max(0, len(grid.row_coords) - 1),
        col_count=max(0, len(grid.col_coords) - 1),
    )
