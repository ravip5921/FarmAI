from __future__ import annotations

import csv
import io
from pathlib import Path

from src.ocr.cell_ocr import OcrTable


def table_to_rows(table: OcrTable, fill_value: str = "") -> list[list[str]]:
    return table.text_matrix(fill_value=fill_value)


def table_to_csv_string(table: OcrTable, *, fill_value: str = "") -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(table_to_rows(table, fill_value=fill_value))
    return output.getvalue()


def write_table_csv(
    table: OcrTable,
    output_path: str | Path,
    *,
    fill_value: str = "",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(table_to_rows(table, fill_value=fill_value))
    return path
