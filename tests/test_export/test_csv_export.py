from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.export.csv_export import table_to_csv_string, table_to_rows, write_table_csv
from src.ocr.cell_ocr import OcrCell, OcrTable


class TestCsvExport(unittest.TestCase):
    def test_table_to_rows_fills_missing_cells(self) -> None:
        table = OcrTable(
            cells=[OcrCell(row=0, col=1, bbox=(0, 0, 1, 1), text="value")],
            row_count=2,
            col_count=3,
        )

        self.assertEqual(
            table_to_rows(table, fill_value=""),
            [["", "value", ""], ["", "", ""]],
        )

    def test_table_to_csv_string_quotes_values(self) -> None:
        table = OcrTable(
            cells=[OcrCell(row=0, col=0, bbox=(0, 0, 1, 1), text="a,b")],
            row_count=2,
            col_count=2,
        )

        self.assertEqual(table_to_csv_string(table, fill_value="-"), '"a,b",-\n-,-\n')

    def test_write_table_csv_creates_parent_directory(self) -> None:
        table = OcrTable(
            cells=[OcrCell(row=0, col=0, bbox=(0, 0, 1, 1), text="ok")],
            row_count=1,
            col_count=1,
        )

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "table.csv"
            written = write_table_csv(table, output_path)

            self.assertEqual(written, output_path)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "ok\n")


if __name__ == "__main__":
    unittest.main()
