from __future__ import annotations

import unittest

import numpy as np

from src.ocr.base import OcrText
from src.ocr.cell_ocr import (
    _recognize_extracted_cells_with_rules,
    recognize_extracted_cells,
    recognize_table_cells,
)
from src.table.cell_extraction import ExtractedCell
from src.table.grid_reconstruction import GridCell, GridStructure


class _Engine:
    def recognize(self, image: np.ndarray) -> OcrText:
        return OcrText(text="70")


class TestCellProgress(unittest.TestCase):
    def test_extracted_and_header_progress_callbacks(self) -> None:
        image = np.full((10, 10), 255, dtype=np.uint8)
        cell = ExtractedCell(row=0, col=0, bbox=(0, 0, 10, 10), image=image)
        progress: list[tuple[int, int]] = []
        recognize_extracted_cells(
            [cell],
            engine=_Engine(),
            progress_callback=lambda done, total: progress.append((done, total)),
        )
        _recognize_extracted_cells_with_rules(
            [cell],
            engine=_Engine(),
            rules_by_col={},
            progress_callback=lambda done, total: progress.append((done, total)),
        )
        self.assertEqual(progress, [(1, 1), (1, 1)])
        grid = GridStructure(
            row_coords=[0, 10, 20],
            col_coords=[0, 10],
            cells=[GridCell(0, 0, (0, 0, 10, 10)), GridCell(1, 0, (0, 10, 10, 10))],
        )
        progress.clear()
        recognize_table_cells(
            np.full((20, 10), 255, dtype=np.uint8),
            grid,
            engine=_Engine(),
            filter_out_columns=["Other"],
            progress_callback=lambda done, total: progress.append((done, total)),
        )
        self.assertEqual(progress, [(1, 2), (2, 2)])

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
