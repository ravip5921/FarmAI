from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.export.json_export import (
    table_to_json_dict,
    table_to_json_string,
    write_table_json,
)
from src.ocr.cell_ocr import OcrCell, OcrTable


class TestJsonExport(unittest.TestCase):
    def test_table_to_json_dict_includes_cells_and_metadata(self) -> None:
        table = OcrTable(
            cells=[
                OcrCell(
                    row=1,
                    col=2,
                    bbox=(3, 4, 5, 6),
                    text="hello",
                    confidence=88.5,
                )
            ],
            row_count=3,
            col_count=4,
        )

        self.assertEqual(
            table_to_json_dict(table),
            {
                "row_count": 3,
                "col_count": 4,
                "cells": [
                    {
                        "row": 1,
                        "col": 2,
                        "bbox": [3, 4, 5, 6],
                        "text": "hello",
                        "confidence": 88.5,
                    }
                ],
            },
        )

    def test_table_to_json_string_serializes_json(self) -> None:
        table = OcrTable(cells=[], row_count=0, col_count=0)

        self.assertEqual(
            json.loads(table_to_json_string(table)), table_to_json_dict(table)
        )

    def test_table_to_json_dict_includes_validation_metadata_when_present(
        self,
    ) -> None:
        table = OcrTable(
            cells=[
                OcrCell(
                    row=0,
                    col=1,
                    bbox=(0, 0, 5, 5),
                    text="",
                    confidence=12.0,
                    raw_text="8A.1",
                    validation_error="expected temperature pattern",
                )
            ],
            row_count=1,
            col_count=2,
        )

        self.assertEqual(
            table_to_json_dict(table)["cells"][0],
            {
                "row": 0,
                "col": 1,
                "bbox": [0, 0, 5, 5],
                "text": "",
                "confidence": 12.0,
                "raw_text": "8A.1",
                "validation_error": "expected temperature pattern",
            },
        )

    def test_write_table_json_creates_parent_directory(self) -> None:
        table = OcrTable(cells=[], row_count=0, col_count=0)

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "table.json"
            written = write_table_json(table, output_path)

            self.assertEqual(written, output_path)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                table_to_json_dict(table),
            )


if __name__ == "__main__":
    unittest.main()
