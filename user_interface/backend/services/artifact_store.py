from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any


def read_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def result_to_csv(result: dict[str, Any]) -> str:
    output = io.StringIO()
    pages = result.get("pages") or []
    include_page = len(pages) > 1
    first_columns = list(pages[0].get("columns") or []) if pages else []
    headers = [str(column["name"]) for column in first_columns]
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow((["Page"] if include_page else []) + headers)
    for page in pages:
        values = {
            (int(cell["row"]), str(cell["column_key"])): str(
                cell.get("reviewed_text", cell.get("ocr_text", ""))
            )
            for cell in page.get("cells", [])
        }
        for row in range(1, int(page.get("data_row_count", 0)) + 1):
            writer.writerow(
                ([page["page_number"]] if include_page else [])
                + [
                    values.get((row, str(column["key"])), "")
                    for column in first_columns
                ]
            )
    return output.getvalue()


def write_csv_artifact(path: Path, result: dict[str, Any]) -> None:
    path.write_text(result_to_csv(result), encoding="utf-8")
