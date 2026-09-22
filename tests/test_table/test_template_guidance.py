from __future__ import annotations

import unittest

from src.table.grid_reconstruction import GridCell, GridStructure
from src.table.template_guidance import apply_template_to_grid
from src.templates import load_template


class TestTemplateGuidance(unittest.TestCase):
    def test_apply_template_to_grid_repairs_columns_from_proportions(self) -> None:
        template = load_template("boar_room")
        grid = GridStructure(
            row_coords=[10, 24, 39],
            col_coords=[5, 20, 40, 80],
            cells=[GridCell(row=0, col=0, bbox=(5, 10, 15, 14))],
        )

        result = apply_template_to_grid(grid, template, (50, 85))

        self.assertTrue(result.repaired_columns)
        self.assertFalse(result.repaired_rows)
        self.assertEqual(len(result.grid.col_coords), 11)
        self.assertEqual(result.grid.col_coords[0], 5)
        self.assertEqual(result.grid.col_coords[-1], 80)
        self.assertEqual(result.grid.row_coords, [10, 24, 39])
        self.assertEqual(len(result.grid.cells), 20)


if __name__ == "__main__":
    unittest.main()
