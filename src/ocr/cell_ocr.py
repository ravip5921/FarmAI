from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.table.cell_extraction import ExtractedCell, extract_cell_images
from src.table.grid_reconstruction import GridStructure

from .base import CellOcrEngine


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
    if engine is None:
        from .registry import create_ocr_engine

        ocr_engine = create_ocr_engine("tesseract")
    else:
        ocr_engine = engine
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


def _get_ocr_engine(engine: CellOcrEngine | None) -> CellOcrEngine:
    if engine is None:
        from .registry import create_ocr_engine

        return create_ocr_engine("tesseract")
    return engine


def _cell_image_filename(cell: ExtractedCell) -> str:
    x, y, width, height = cell.bbox
    return (
        f"row_{cell.row:03d}_col_{cell.col:03d}"
        f"_x{x:04d}_y{y:04d}_w{width:04d}_h{height:04d}.png"
    )


def save_extracted_cell_images(
    cells: list[ExtractedCell],
    directory: str | Path,
) -> list[Path]:
    """Save cropped cell images with row/column and bbox coordinates."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for cell in cells:
        path = output_dir / _cell_image_filename(cell)
        if not cv2.imwrite(str(path), cell.image):
            raise RuntimeError(f"Could not write cell image: {path}")
        written.append(path)
    return written


def _normalize_column_name(value: str) -> str:
    return " ".join(value.casefold().split())


def _compact_columns(
    cells: list[OcrCell],
    *,
    row_count: int,
    original_col_count: int,
    filtered_cols: set[int],
) -> OcrTable:
    kept_cols = [
        col for col in range(max(0, original_col_count)) if col not in filtered_cols
    ]
    col_map = {old_col: new_col for new_col, old_col in enumerate(kept_cols)}
    compacted = [
        OcrCell(
            row=cell.row,
            col=col_map[cell.col],
            bbox=cell.bbox,
            text=cell.text,
            confidence=cell.confidence,
        )
        for cell in cells
        if cell.col in col_map
    ]
    compacted.sort(key=lambda cell: (cell.row, cell.col))
    return OcrTable(
        cells=compacted,
        row_count=row_count,
        col_count=len(kept_cols),
    )


def recognize_table_cells(
    image: np.ndarray,
    grid: GridStructure,
    *,
    engine: CellOcrEngine | None = None,
    padding: int = 2,
    context_padding: int = 0,
    filter_out_columns: Collection[str] | None = None,
    filter_out_column_indices: Collection[int] | None = None,
    column_names: Collection[str] | None = None,
    cell_image_dir: str | Path | None = None,
) -> OcrTable:
    extracted_cells = extract_cell_images(
        image,
        grid,
        padding=padding,
        context_padding=context_padding,
    )
    if cell_image_dir is not None:
        save_extracted_cell_images(extracted_cells, cell_image_dir)

    row_count = max(0, len(grid.row_coords) - 1)
    col_count = max(0, len(grid.col_coords) - 1)
    filtered_cols = {
        int(col)
        for col in (filter_out_column_indices or ())
        if 0 <= int(col) < col_count
    }
    normalized_filter = {
        _normalize_column_name(name)
        for name in (filter_out_columns or ())
        if _normalize_column_name(name)
    }
    known_column_names = list(column_names or [])
    if known_column_names and len(known_column_names) == col_count:
        ocr_engine = _get_ocr_engine(engine)
        header_cells = {
            cell.col: cell
            for cell in extracted_cells
            if cell.row == 0 and cell.col not in filtered_cols
        }
        recognized = [
            OcrCell(
                row=0,
                col=col,
                bbox=header_cells[col].bbox if col in header_cells else (0, 0, 1, 1),
                text=known_column_names[col],
                confidence=None,
            )
            for col in range(col_count)
            if col not in filtered_cols
        ]
        for cell in (
            cell
            for cell in extracted_cells
            if cell.row != 0 and cell.col not in filtered_cols
        ):
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

        return _compact_columns(
            recognized,
            row_count=row_count,
            original_col_count=col_count,
            filtered_cols=filtered_cols,
        )

    if not normalized_filter:
        table = recognize_extracted_cells(
            [cell for cell in extracted_cells if cell.col not in filtered_cols],
            engine=engine,
            row_count=row_count,
            col_count=col_count if not filtered_cols else None,
        )
        if filtered_cols:
            return _compact_columns(
                table.cells,
                row_count=row_count,
                original_col_count=col_count,
                filtered_cols=filtered_cols,
            )
        return table

    ocr_engine = _get_ocr_engine(engine)
    recognized: list[OcrCell] = []

    for cell in (
        cell
        for cell in extracted_cells
        if cell.row == 0 and cell.col not in filtered_cols
    ):
        result = ocr_engine.recognize(cell.image)
        if _normalize_column_name(result.text) in normalized_filter:
            filtered_cols.add(cell.col)
        recognized.append(
            OcrCell(
                row=cell.row,
                col=cell.col,
                bbox=cell.bbox,
                text=result.text,
                confidence=result.confidence,
            )
        )

    for cell in (
        cell
        for cell in extracted_cells
        if cell.row != 0 and cell.col not in filtered_cols
    ):
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

    return _compact_columns(
        recognized,
        row_count=row_count,
        original_col_count=col_count,
        filtered_cols=filtered_cols,
    )
