from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.export.csv_export import table_to_csv_string
from src.ocr.cell_ocr import (
    OcrCell,
    OcrTable,
    _get_ocr_engine,
    recognize_extracted_cells,
    recognize_table_cells,
)
from src.ocr.table_ocr import export_table_ocr
from src.ocr.tesseract_engine import OcrText, TesseractConfig
from src.table.cell_extraction import ExtractedCell
from src.table.grid_reconstruction import GridCell, GridStructure


class FakeEngine:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image: np.ndarray) -> OcrText:
        self.calls += 1
        return OcrText(text=f"cell-{self.calls}", confidence=90.0)


class SequenceEngine:
    def __init__(self, values: list[str]) -> None:
        self.values = values
        self.calls = 0

    def recognize(self, image: np.ndarray) -> OcrText:
        value = self.values[self.calls]
        self.calls += 1
        return OcrText(text=value, confidence=90.0)


class TestCellOcr(unittest.TestCase):
    def test_recognize_extracted_cells_creates_default_engine(self) -> None:
        cell = ExtractedCell(
            row=2,
            col=3,
            bbox=(0, 0, 1, 1),
            image=np.zeros((1, 1), dtype=np.uint8),
        )
        engine = FakeEngine()

        with patch("src.ocr.registry.create_ocr_engine", return_value=engine):
            table = recognize_extracted_cells([cell])

        self.assertEqual(engine.calls, 1)
        self.assertEqual(table.row_count, 3)
        self.assertEqual(table.col_count, 4)
        self.assertEqual(table.text_matrix()[2][3], "cell-1")

    def test_get_ocr_engine_creates_default_engine(self) -> None:
        engine = FakeEngine()

        with patch("src.ocr.registry.create_ocr_engine", return_value=engine):
            self.assertIs(_get_ocr_engine(None), engine)

    def test_recognize_table_cells_returns_structured_ocr_table(self) -> None:
        image = np.zeros((10, 10), dtype=np.uint8)
        grid = GridStructure(
            row_coords=[0, 5, 10],
            col_coords=[0, 5],
            cells=[
                GridCell(row=0, col=0, bbox=(0, 0, 5, 5)),
                GridCell(row=1, col=0, bbox=(0, 5, 5, 5)),
            ],
        )
        engine = FakeEngine()

        table = recognize_table_cells(image, grid, engine=engine, padding=0)

        self.assertEqual(table.row_count, 2)
        self.assertEqual(table.col_count, 1)
        self.assertEqual([cell.text for cell in table.cells], ["cell-1", "cell-2"])
        self.assertEqual(table.text_matrix(), [["cell-1"], ["cell-2"]])
        self.assertEqual(engine.calls, 2)

    def test_recognize_table_cells_filters_matching_header_columns(self) -> None:
        image = np.zeros((10, 15), dtype=np.uint8)
        grid = GridStructure(
            row_coords=[0, 5, 10],
            col_coords=[0, 5, 10, 15],
            cells=[
                GridCell(row=0, col=0, bbox=(0, 0, 5, 5)),
                GridCell(row=0, col=1, bbox=(5, 0, 5, 5)),
                GridCell(row=0, col=2, bbox=(10, 0, 5, 5)),
                GridCell(row=1, col=0, bbox=(0, 5, 5, 5)),
                GridCell(row=1, col=1, bbox=(5, 5, 5, 5)),
                GridCell(row=1, col=2, bbox=(10, 5, 5, 5)),
            ],
        )
        engine = SequenceEngine(
            ["Keep A", " Remove   Me ", "Keep B", "value-a", "value-b"]
        )

        table = recognize_table_cells(
            image,
            grid,
            engine=engine,
            padding=0,
            filter_out_columns={"remove me"},
        )

        self.assertEqual(engine.calls, 5)
        self.assertEqual(table.row_count, 2)
        self.assertEqual(table.col_count, 2)
        self.assertEqual(
            table.text_matrix(),
            [["Keep A", "Keep B"], ["value-a", "value-b"]],
        )
        self.assertEqual(
            table_to_csv_string(table),
            "Keep A,Keep B\nvalue-a,value-b\n",
        )
        self.assertEqual([cell.col for cell in table.cells], [0, 1, 0, 1])

    def test_tesseract_config_builds_config_string(self) -> None:
        config = TesseractConfig(
            psm=7, oem=1, extra_config="-c preserve_interword_spaces=1"
        )

        self.assertEqual(
            config.to_config_string(),
            "--oem 1 --psm 7 -c preserve_interword_spaces=1",
        )

    def test_table_to_csv_string_quotes_values_and_fills_empty_cells(self) -> None:
        table = OcrTable(
            cells=[OcrCell(row=0, col=0, bbox=(0, 0, 1, 1), text="a,b")],
            row_count=2,
            col_count=2,
        )

        self.assertEqual(table_to_csv_string(table, fill_value="-"), '"a,b",-\n-,-\n')

    def test_export_table_ocr_writes_requested_outputs(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        image = np.zeros((5, 5), dtype=np.uint8)
        grid = GridStructure(
            row_coords=[0, 5],
            col_coords=[0, 5],
            cells=[GridCell(row=0, col=0, bbox=(0, 0, 5, 5))],
        )

        with TemporaryDirectory() as tmpdir:
            result = export_table_ocr(
                image,
                grid,
                csv_path=Path(tmpdir) / "table.csv",
                json_path=Path(tmpdir) / "table.json",
                engine=FakeEngine(),
                padding=0,
                filter_out_columns=set(),
            )

            self.assertEqual(result.table.text_matrix(), [["cell-1"]])
            self.assertTrue(result.csv_path is not None and result.csv_path.exists())
            self.assertTrue(result.json_path is not None and result.json_path.exists())

    def test_export_table_ocr_saves_cropped_cell_images(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        grid = GridStructure(
            row_coords=[0, 5, 10],
            col_coords=[0, 5],
            cells=[
                GridCell(row=0, col=0, bbox=(0, 0, 5, 5)),
                GridCell(row=1, col=0, bbox=(0, 5, 5, 5)),
            ],
        )

        with TemporaryDirectory() as tmpdir:
            cell_dir = Path(tmpdir) / "record_cells"
            result = export_table_ocr(
                image,
                grid,
                cell_image_dir=cell_dir,
                engine=FakeEngine(),
                padding=0,
                context_padding=1,
                filter_out_columns=set(),
            )

            files = sorted(path.name for path in cell_dir.glob("*.png"))

        self.assertEqual(result.table.text_matrix(), [["cell-1"], ["cell-2"]])
        self.assertEqual(
            files,
            [
                "row_000_col_000_x0000_y0000_w0006_h0006.png",
                "row_001_col_000_x0000_y0004_w0006_h0006.png",
            ],
        )


if __name__ == "__main__":
    unittest.main()
