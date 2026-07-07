from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.ocr.base import OcrText
from src.ocr.cell_ocr import (
    _recognize_extracted_cells_with_rules,
    recognize_table_cells,
    save_extracted_cell_images,
)
from src.ocr.column_rules import ColumnOcrRule
from src.table.cell_extraction import ExtractedCell
from src.table.grid_reconstruction import GridCell, GridStructure


class _SequenceEngine:
    def __init__(self, values: list[OcrText]) -> None:
        self.values = values

    def recognize(self, image: np.ndarray) -> OcrText:
        return self.values.pop(0)


def _grid() -> GridStructure:
    return GridStructure(
        row_coords=[0, 5, 10],
        col_coords=[0, 5, 10],
        cells=[
            GridCell(row=0, col=0, bbox=(0, 0, 5, 5)),
            GridCell(row=0, col=1, bbox=(5, 0, 5, 5)),
            GridCell(row=1, col=0, bbox=(0, 5, 5, 5)),
            GridCell(row=1, col=1, bbox=(5, 5, 5, 5)),
        ],
    )


class TestCellOcrCoverage(unittest.TestCase):
    def test_recognize_extracted_cells_with_rules_infers_table_size(self) -> None:
        cells = [ExtractedCell(1, 2, (0, 0, 1, 1), np.zeros((1, 1), dtype=np.uint8))]
        engine = _SequenceEngine([OcrText(text="42", confidence=91)])
        rule = ColumnOcrRule(index=2, key="temp", value_type="temperature")

        table = _recognize_extracted_cells_with_rules(
            cells,
            engine=engine,
            rules_by_col={2: rule},
        )

        self.assertEqual(table.row_count, 2)
        self.assertEqual(table.col_count, 3)
        self.assertEqual(table.text_matrix()[1][2], "42")

    def test_save_extracted_cell_images_raises_when_cv_write_fails(self) -> None:
        cell = ExtractedCell(0, 0, (0, 0, 1, 1), np.zeros((1, 1), dtype=np.uint8))

        with TemporaryDirectory() as tmpdir:
            with patch("src.ocr.cell_ocr.cv2.imwrite", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "Could not write cell image"):
                    save_extracted_cell_images([cell], Path(tmpdir))

    def test_known_column_names_save_cell_images_before_ocr(self) -> None:
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        engine = _SequenceEngine([OcrText(text="A1"), OcrText(text="101")])

        with TemporaryDirectory() as tmpdir:
            cell_dir = Path(tmpdir) / "cells"
            table = recognize_table_cells(
                image,
                _grid(),
                engine=engine,
                padding=0,
                column_names=["Pen", "HI"],
                column_keys=["pen key", "hi"],
                cell_image_dir=cell_dir,
            )
            files = sorted(path.name for path in cell_dir.glob("*.png"))

        self.assertEqual(table.text_matrix(), [["Pen", "HI"], ["A1", "101"]])
        self.assertEqual(len(files), 4)
        self.assertTrue(any(name.startswith("pen_key_row_000") for name in files))

    def test_unfiltered_branch_saves_images_and_uses_plain_recognition(self) -> None:
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        engine = _SequenceEngine(
            [
                OcrText(text="h0"),
                OcrText(text="h1"),
                OcrText(text="d0"),
                OcrText(text="d1"),
            ]
        )

        with TemporaryDirectory() as tmpdir:
            table = recognize_table_cells(
                image,
                _grid(),
                engine=engine,
                padding=0,
                filter_out_columns=set(),
                cell_image_dir=Path(tmpdir) / "cells",
            )

        self.assertEqual(table.text_matrix(), [["h0", "h1"], ["d0", "d1"]])

    def test_unfiltered_branch_uses_column_rules_when_present(self) -> None:
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        engine = _SequenceEngine(
            [
                OcrText(text="h0"),
                OcrText(text="h1"),
                OcrText(text="101"),
                OcrText(text="102"),
            ]
        )
        rule = ColumnOcrRule(index=1, key="hi", value_type="temperature")

        table = recognize_table_cells(
            image,
            _grid(),
            engine=engine,
            padding=0,
            filter_out_columns=set(),
            column_ocr_rules=[rule],
        )

        self.assertEqual(table.text_matrix(), [["h0", "h1"], ["101", "102"]])

    def test_header_filter_branch_recognizes_headers_and_remaining_data(self) -> None:
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        engine = _SequenceEngine(
            [
                OcrText(text="Drop"),
                OcrText(text="Keep"),
                OcrText(text="kept-data"),
            ]
        )

        with TemporaryDirectory() as tmpdir:
            table = recognize_table_cells(
                image,
                _grid(),
                engine=engine,
                padding=0,
                filter_out_columns={"drop"},
                cell_image_dir=Path(tmpdir) / "cells",
            )

        self.assertEqual(table.text_matrix(), [["Keep"], ["kept-data"]])


if __name__ == "__main__":
    unittest.main()
