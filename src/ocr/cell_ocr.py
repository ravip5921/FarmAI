from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from src.table.cell_extraction import ExtractedCell, extract_cell_images
from src.table.grid_reconstruction import GridStructure

from .base import CellOcrEngine
from .column_rules import ColumnOcrRule, recognize_with_column_rule

CellProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class OcrCell:
    row: int
    col: int
    bbox: tuple[int, int, int, int]
    text: str
    confidence: float | None = None
    raw_text: str | None = None
    validation_error: str | None = None


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
    progress_callback: CellProgressCallback | None = None,
) -> OcrTable:
    if engine is None:
        from .registry import create_ocr_engine

        ocr_engine = create_ocr_engine("tesseract")
    else:
        ocr_engine = engine
    recognized: list[OcrCell] = []
    total = len(cells)
    for completed, cell in enumerate(cells, start=1):
        result = ocr_engine.recognize(cell.image)
        recognized.append(
            OcrCell(
                row=cell.row,
                col=cell.col,
                bbox=cell.bbox,
                text=result.text,
                confidence=result.confidence,
                raw_text=result.raw_text,
                validation_error=result.validation_error,
            )
        )
        if progress_callback is not None:
            progress_callback(completed, total)

    inferred_rows = max((cell.row for cell in recognized), default=-1) + 1
    inferred_cols = max((cell.col for cell in recognized), default=-1) + 1
    return OcrTable(
        cells=recognized,
        row_count=row_count if row_count is not None else inferred_rows,
        col_count=col_count if col_count is not None else inferred_cols,
    )


def _recognize_extracted_cells_with_rules(
    cells: list[ExtractedCell],
    *,
    engine: CellOcrEngine | None = None,
    rules_by_col: dict[int, ColumnOcrRule],
    llm_verifier: Any | None = None,
    row_count: int | None = None,
    col_count: int | None = None,
    progress_callback: CellProgressCallback | None = None,
) -> OcrTable:
    ocr_engine = _get_ocr_engine(engine)
    recognized: list[OcrCell] = []
    total = len(cells)
    for completed, cell in enumerate(cells, start=1):
        result = recognize_with_column_rule(
            ocr_engine,
            cell.image,
            rules_by_col.get(cell.col),
            llm_verifier=llm_verifier,
        )
        recognized.append(
            OcrCell(
                row=cell.row,
                col=cell.col,
                bbox=cell.bbox,
                text=result.text,
                confidence=result.confidence,
                raw_text=result.raw_text,
                validation_error=result.validation_error,
            )
        )
        if progress_callback is not None:
            progress_callback(completed, total)

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


def _safe_filename_token(value: str, *, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return token or fallback


def _cell_image_filename(
    cell: ExtractedCell,
    *,
    column_keys: Collection[str] | None = None,
) -> str:
    x, y, width, height = cell.bbox
    keys = list(column_keys or [])
    fallback = f"col_{cell.col:03d}"
    column_key = (
        _safe_filename_token(keys[cell.col], fallback=fallback)
        if 0 <= cell.col < len(keys)
        else fallback
    )
    return (
        f"{column_key}_row_{cell.row:03d}"
        f"_coords_x{x:04d}_y{y:04d}_w{width:04d}_h{height:04d}.png"
    )


def save_extracted_cell_images(
    cells: list[ExtractedCell],
    directory: str | Path,
    *,
    column_keys: Collection[str] | None = None,
) -> list[Path]:
    """Save cropped cell images with row/column and bbox coordinates."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for cell in cells:
        path = output_dir / _cell_image_filename(cell, column_keys=column_keys)
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
            raw_text=cell.raw_text,
            validation_error=cell.validation_error,
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
    column_keys: Collection[str] | None = None,
    column_ocr_rules: Collection[ColumnOcrRule] | None = None,
    llm_verifier: Any | None = None,
    cell_image_dir: str | Path | None = None,
    progress_callback: CellProgressCallback | None = None,
) -> OcrTable:
    extracted_cells = extract_cell_images(
        image,
        grid,
        padding=padding,
        context_padding=context_padding,
    )

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
    rules_by_col = {
        int(rule.index): rule
        for rule in (column_ocr_rules or ())
        if 0 <= int(rule.index) < col_count
    }
    known_column_names = list(column_names or [])
    if known_column_names and len(known_column_names) == col_count:
        cells_to_save_or_ocr = [
            cell for cell in extracted_cells if cell.col not in filtered_cols
        ]
        if cell_image_dir is not None:
            save_extracted_cell_images(
                cells_to_save_or_ocr,
                cell_image_dir,
                column_keys=column_keys,
            )

        ocr_engine = _get_ocr_engine(engine)
        header_cells_by_col = {
            cell.col: cell for cell in cells_to_save_or_ocr if cell.row == 0
        }
        recognized = [
            OcrCell(
                row=0,
                col=col,
                bbox=(
                    header_cells_by_col[col].bbox
                    if col in header_cells_by_col
                    else (0, 0, 1, 1)
                ),
                text=known_column_names[col],
                confidence=None,
            )
            for col in range(col_count)
            if col not in filtered_cols
        ]
        data_cells = [cell for cell in cells_to_save_or_ocr if cell.row != 0]
        total = len(data_cells)
        for completed, cell in enumerate(data_cells, start=1):
            result = recognize_with_column_rule(
                ocr_engine,
                cell.image,
                rules_by_col.get(cell.col),
                llm_verifier=llm_verifier,
            )
            recognized.append(
                OcrCell(
                    row=cell.row,
                    col=cell.col,
                    bbox=cell.bbox,
                    text=result.text,
                    confidence=result.confidence,
                    raw_text=result.raw_text,
                    validation_error=result.validation_error,
                )
            )
            if progress_callback is not None:
                progress_callback(completed, total)

        return _compact_columns(
            recognized,
            row_count=row_count,
            original_col_count=col_count,
            filtered_cols=filtered_cols,
        )

    if not normalized_filter:
        cells_to_save_or_ocr = [
            cell for cell in extracted_cells if cell.col not in filtered_cols
        ]
        if cell_image_dir is not None:
            save_extracted_cell_images(
                cells_to_save_or_ocr,
                cell_image_dir,
                column_keys=column_keys,
            )

        if rules_by_col or llm_verifier is not None:
            table = _recognize_extracted_cells_with_rules(
                cells_to_save_or_ocr,
                engine=engine,
                rules_by_col=rules_by_col,
                llm_verifier=llm_verifier,
                row_count=row_count,
                col_count=col_count if not filtered_cols else None,
                progress_callback=progress_callback,
            )
        else:
            table = recognize_extracted_cells(
                cells_to_save_or_ocr,
                engine=engine,
                row_count=row_count,
                col_count=col_count if not filtered_cols else None,
                progress_callback=progress_callback,
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
    recognized = []

    header_cells_to_ocr = [
        cell
        for cell in extracted_cells
        if cell.row == 0 and cell.col not in filtered_cols
    ]
    for cell in header_cells_to_ocr:
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
                raw_text=result.raw_text,
                validation_error=result.validation_error,
            )
        )

    if cell_image_dir is not None:
        save_extracted_cell_images(
            [cell for cell in extracted_cells if cell.col not in filtered_cols],
            cell_image_dir,
            column_keys=column_keys,
        )

    data_cells = [
        cell
        for cell in extracted_cells
        if cell.row != 0 and cell.col not in filtered_cols
    ]
    total = len(header_cells_to_ocr) + len(data_cells)
    completed = len(header_cells_to_ocr)
    if progress_callback is not None and completed:
        progress_callback(completed, total)

    for cell in data_cells:
        result = recognize_with_column_rule(
            ocr_engine,
            cell.image,
            rules_by_col.get(cell.col),
            llm_verifier=llm_verifier,
        )
        recognized.append(
            OcrCell(
                row=cell.row,
                col=cell.col,
                bbox=cell.bbox,
                text=result.text,
                confidence=result.confidence,
                raw_text=result.raw_text,
                validation_error=result.validation_error,
            )
        )
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, total)

    return _compact_columns(
        recognized,
        row_count=row_count,
        original_col_count=col_count,
        filtered_cols=filtered_cols,
    )
