from __future__ import annotations

import unittest
from unittest.mock import patch

from src.ocr import OcrCell, OcrTable
from streamlit_app import (
    dataframe_to_csv,
    parse_filter_out_columns,
    resolve_filter_out_columns,
    table_to_dataframe,
)


class TestStreamlitColumnFilter(unittest.TestCase):
    def test_table_to_dataframe_uses_first_ocr_row_as_headers(self) -> None:
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

        df = table_to_dataframe(table)

        self.assertEqual(list(df.columns), ["Pen", "Date"])
        self.assertEqual(df.values.tolist(), [["A1", "06/03/26"]])
        self.assertEqual(dataframe_to_csv(df), "Pen,Date\nA1,06/03/26\n")

    def test_table_to_dataframe_falls_back_for_blank_or_duplicate_headers(
        self,
    ) -> None:
        table = OcrTable(
            cells=[
                OcrCell(row=0, col=0, bbox=(0, 0, 1, 1), text="Name"),
                OcrCell(row=0, col=1, bbox=(1, 0, 1, 1), text=" "),
                OcrCell(row=0, col=2, bbox=(2, 0, 1, 1), text="Name"),
            ],
            row_count=1,
            col_count=3,
        )

        df = table_to_dataframe(table)

        self.assertEqual(list(df.columns), ["Name", "Column 2", "Name (2)"])
        self.assertEqual(df.values.tolist(), [])

    def test_parse_filter_out_columns_reads_comma_separated_values(self) -> None:
        self.assertEqual(
            parse_filter_out_columns(" Pen, Date , , Notes "),
            {"Pen", "Date", "Notes"},
        )

    def test_resolve_filter_out_columns_uses_defaults_when_ui_is_empty(self) -> None:
        with patch("streamlit_app.FILTER_OUT_COLUMNS", {"Default"}):
            self.assertEqual(resolve_filter_out_columns(set()), {"Default"})

    def test_resolve_filter_out_columns_uses_ui_values_when_present(self) -> None:
        with patch("streamlit_app.FILTER_OUT_COLUMNS", {"Default"}):
            self.assertEqual(resolve_filter_out_columns({"UI"}), {"UI"})


if __name__ == "__main__":
    unittest.main()
