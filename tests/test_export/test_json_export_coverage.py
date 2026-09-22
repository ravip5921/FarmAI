from __future__ import annotations

import unittest

from src.export.json_export import table_to_json_dict
from src.ocr.cell_ocr import OcrCell, OcrTable


class TestJsonExportCoverage(unittest.TestCase):
    def test_cell_metadata_fields_are_independently_optional(self) -> None:
        table = OcrTable(
            cells=[
                OcrCell(
                    row=0,
                    col=0,
                    bbox=(0, 0, 1, 1),
                    text="",
                    raw_text="raw",
                ),
                OcrCell(
                    row=0,
                    col=1,
                    bbox=(1, 0, 1, 1),
                    text="",
                    validation_error="invalid",
                ),
            ],
            row_count=1,
            col_count=2,
        )

        cells = table_to_json_dict(table)["cells"]

        self.assertEqual(cells[0]["raw_text"], "raw")
        self.assertNotIn("validation_error", cells[0])
        self.assertEqual(cells[1]["validation_error"], "invalid")
        self.assertNotIn("raw_text", cells[1])


if __name__ == "__main__":
    unittest.main()
