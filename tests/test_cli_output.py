from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from main import print_ocr_csv
from src.ocr.cell_ocr import OcrCell, OcrTable


class TestCliOutput(unittest.TestCase):
    def test_print_ocr_csv_prints_labeled_csv_block(self) -> None:
        table = OcrTable(
            cells=[
                OcrCell(row=0, col=0, bbox=(0, 0, 1, 1), text="Pen"),
                OcrCell(row=0, col=1, bbox=(1, 0, 1, 1), text="Date"),
                OcrCell(row=1, col=0, bbox=(0, 1, 1, 1), text="A1"),
                OcrCell(row=1, col=1, bbox=(1, 1, 1, 1), text="06/03/26"),
            ],
            row_count=2,
            col_count=2,
        )
        output = io.StringIO()

        with redirect_stdout(output):
            print_ocr_csv(table, image_name="sample_01.jpg")

        self.assertEqual(
            output.getvalue(),
            "\n--- OCR CSV: sample_01 ---\n"
            "Pen,Date\n"
            "A1,06/03/26\n"
            "--- END OCR CSV: sample_01 ---\n\n",
        )


if __name__ == "__main__":
    unittest.main()
