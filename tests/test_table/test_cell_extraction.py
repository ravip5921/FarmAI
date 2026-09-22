from __future__ import annotations

import unittest

import numpy as np

from src.table.cell_extraction import crop_cell, extract_cell_images
from src.table.grid_reconstruction import GridCell, GridStructure


class TestCellExtraction(unittest.TestCase):
    def test_crop_cell_clips_bbox_and_applies_padding(self) -> None:
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        cell = GridCell(row=1, col=2, bbox=(-2, 1, 7, 6))

        extracted = crop_cell(image, cell, padding=1)

        self.assertIsNotNone(extracted)
        assert extracted is not None
        self.assertEqual(extracted.row, 1)
        self.assertEqual(extracted.col, 2)
        self.assertEqual(extracted.bbox, (0, 2, 4, 4))
        self.assertTrue(np.array_equal(extracted.image, image[2:6, 0:4]))

    def test_crop_cell_returns_none_when_padding_removes_cell(self) -> None:
        image = np.zeros((5, 5), dtype=np.uint8)
        cell = GridCell(row=0, col=0, bbox=(1, 1, 2, 2))

        self.assertIsNone(crop_cell(image, cell, padding=2))

    def test_crop_cell_expands_with_context_padding(self) -> None:
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        cell = GridCell(row=0, col=1, bbox=(2, 2, 4, 4))

        extracted = crop_cell(image, cell, padding=1, context_padding=3)

        self.assertIsNotNone(extracted)
        assert extracted is not None
        self.assertEqual(extracted.bbox, (0, 0, 8, 8))
        self.assertTrue(np.array_equal(extracted.image, image[0:8, 0:8]))

    def test_extract_cell_images_returns_row_major_cells(self) -> None:
        image = np.zeros((10, 10), dtype=np.uint8)
        grid = GridStructure(
            row_coords=[0, 5, 10],
            col_coords=[0, 5],
            cells=[
                GridCell(row=1, col=0, bbox=(0, 5, 5, 5)),
                GridCell(row=0, col=0, bbox=(0, 0, 5, 5)),
            ],
        )

        cells = extract_cell_images(image, grid, padding=0)

        self.assertEqual([(cell.row, cell.col) for cell in cells], [(0, 0), (1, 0)])


if __name__ == "__main__":
    unittest.main()
