from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.table.grid_reconstruction import GridStructure

from .base import CellOcrEngine
from .cell_ocr import CellProgressCallback, OcrTable, recognize_table_cells
from .column_rules import ColumnOcrRule


@dataclass(frozen=True)
class TableOcrExportResult:
    table: OcrTable
    csv_path: Path | None = None
    json_path: Path | None = None


def run_table_ocr(
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
    return recognize_table_cells(
        image,
        grid,
        engine=engine,
        padding=padding,
        context_padding=context_padding,
        filter_out_columns=filter_out_columns,
        filter_out_column_indices=filter_out_column_indices,
        column_names=column_names,
        column_keys=column_keys,
        column_ocr_rules=column_ocr_rules,
        llm_verifier=llm_verifier,
        cell_image_dir=cell_image_dir,
        progress_callback=progress_callback,
    )


def export_table_ocr(
    image: np.ndarray,
    grid: GridStructure,
    *,
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
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
) -> TableOcrExportResult:

    from src.export.csv_export import write_table_csv
    from src.export.json_export import write_table_json

    table = run_table_ocr(
        image,
        grid,
        engine=engine,
        padding=padding,
        context_padding=context_padding,
        filter_out_columns=filter_out_columns,
        filter_out_column_indices=filter_out_column_indices,
        column_names=column_names,
        column_keys=column_keys,
        column_ocr_rules=column_ocr_rules,
        llm_verifier=llm_verifier,
        cell_image_dir=cell_image_dir,
        progress_callback=progress_callback,
    )
    written_csv = write_table_csv(table, csv_path) if csv_path is not None else None
    written_json = write_table_json(table, json_path) if json_path is not None else None
    return TableOcrExportResult(
        table=table,
        csv_path=written_csv,
        json_path=written_json,
    )
