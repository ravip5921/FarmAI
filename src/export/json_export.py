from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ocr.cell_ocr import OcrTable


def _cell_to_json_dict(cell) -> dict[str, Any]:
    data = {
        "row": cell.row,
        "col": cell.col,
        "bbox": list(cell.bbox),
        "text": cell.text,
        "confidence": cell.confidence,
    }
    if cell.raw_text is not None:
        data["raw_text"] = cell.raw_text
    if cell.validation_error is not None:
        data["validation_error"] = cell.validation_error
    return data


def table_to_json_dict(table: OcrTable) -> dict[str, Any]:
    return {
        "row_count": table.row_count,
        "col_count": table.col_count,
        "cells": [_cell_to_json_dict(cell) for cell in table.cells],
    }


def table_to_json_string(table: OcrTable, *, indent: int | None = 2) -> str:
    return json.dumps(table_to_json_dict(table), indent=indent)


def write_table_json(table: OcrTable, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table_to_json_string(table), encoding="utf-8")
    return path
