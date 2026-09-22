from __future__ import annotations

import unittest

import numpy as np

from src.ocr.base import OcrText
from src.ocr.cell_ocr import recognize_table_cells
from src.table.grid_reconstruction import GridCell, GridStructure


class _Engine:
    def recognize(self, image: np.ndarray) -> OcrText:
        return OcrText(text="70")


class TestCellProgress(unittest.TestCase):
    def test_known_template_reports_data_cells_only(self) -> None:
        grid = GridStructure(
            row_coords=[0, 10, 20],
            col_coords=[0, 10],
            cells=[
                GridCell(row=0, col=0, bbox=(0, 0, 10, 10)),
                GridCell(row=1, col=0, bbox=(0, 10, 10, 10)),
            ],
        )
        progress: list[tuple[int, int]] = []

        recognize_table_cells(
            np.full((20, 10), 255, dtype=np.uint8),
            grid,
            engine=_Engine(),
            column_names=["Temperature"],
            column_keys=["temperature"],
            progress_callback=lambda completed, total: progress.append(
                (completed, total)
            ),
        )

        self.assertEqual(progress, [(1, 1)])


if __name__ == "__main__":
    unittest.main()
