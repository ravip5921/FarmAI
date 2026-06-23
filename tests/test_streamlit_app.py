from __future__ import annotations

import unittest
from unittest.mock import patch

from streamlit_app import parse_filter_out_columns, resolve_filter_out_columns


class TestStreamlitColumnFilter(unittest.TestCase):
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
