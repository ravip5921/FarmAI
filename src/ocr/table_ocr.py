from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.table.grid_reconstruction import GridStructure

from .base import CellOcrEngine
from .cell_ocr import OcrTable, recognize_table_cells


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
    cell_image_dir: str | Path | None = None,
) -> OcrTable:
    return recognize_table_cells(
        image,
        grid,
        engine=engine,
        padding=padding,
        context_padding=context_padding,
        filter_out_columns=filter_out_columns,
        cell_image_dir=cell_image_dir,
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
    cell_image_dir: str | Path | None = None,
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
        cell_image_dir=cell_image_dir,
    )
    written_csv = write_table_csv(table, csv_path) if csv_path is not None else None
    written_json = write_table_json(table, json_path) if json_path is not None else None
    return TableOcrExportResult(
        table=table,
        csv_path=written_csv,
        json_path=written_json,
    )
