from __future__ import annotations

import unittest

import numpy as np

from src.ocr.cell_ocr import recognize_table_cells
from src.ocr.table_ocr import export_table_ocr
from src.ocr.tesseract_engine import OcrText, TesseractConfig
from src.table.grid_reconstruction import GridCell, GridStructure


class FakeEngine:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image: np.ndarray) -> OcrText:
        self.calls += 1
        return OcrText(text=f"cell-{self.calls}", confidence=90.0)


class TestCellOcr(unittest.TestCase):
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

    def test_tesseract_config_builds_config_string(self) -> None:
        config = TesseractConfig(
            psm=7, oem=1, extra_config="-c preserve_interword_spaces=1"
        )

        self.assertEqual(
            config.to_config_string(),
            "--oem 1 --psm 7 -c preserve_interword_spaces=1",
        )

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
            )

            self.assertEqual(result.table.text_matrix(), [["cell-1"]])
            self.assertTrue(result.csv_path is not None and result.csv_path.exists())
            self.assertTrue(result.json_path is not None and result.json_path.exists())


if __name__ == "__main__":
    unittest.main()
