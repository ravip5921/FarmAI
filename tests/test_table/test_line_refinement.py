from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.table.line_refinement import refine_grid_with_projection_profiles


class TestLineRefinement(unittest.TestCase):
    def test_refinement_fills_missing_regular_row_and_filters_weak_column(self) -> None:
        horizontal = np.zeros((24, 24), dtype=np.uint8)
        vertical = np.zeros((24, 24), dtype=np.uint8)

        cv2.line(horizontal, (1, 1), (20, 1), color=255, thickness=1)
        for y in (4, 8, 16, 20):
            cv2.line(horizontal, (2, y), (18, y), color=255, thickness=1)

        cv2.line(vertical, (3, 4), (3, 20), color=255, thickness=1)
        cv2.line(vertical, (4, 4), (4, 20), color=255, thickness=1)
        cv2.line(vertical, (12, 4), (12, 20), color=255, thickness=1)
        cv2.line(vertical, (19, 2), (19, 5), color=255, thickness=1)

        result = refine_grid_with_projection_profiles(horizontal, vertical, horizontal.shape)

        self.assertEqual(result.grid.row_coords, [4, 8, 12, 16, 20])
        self.assertEqual(result.grid.col_coords, [4, 12])
        self.assertEqual(len(result.grid.cells), 4)
        self.assertEqual(result.estimated_row_spacing, 4)


if __name__ == "__main__":
    unittest.main()
