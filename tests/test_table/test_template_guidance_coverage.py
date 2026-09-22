from __future__ import annotations

import unittest

from src.table.grid_reconstruction import GridStructure
from src.table.template_guidance import (
    _column_count_score,
    _grid_from_axis_coordinates,
    _proportional_columns,
    _regular_rows,
    apply_template_to_grid,
)
from src.templates import FormTemplate


def _template(*, widths=(1.0, 1.0), uniform_row_height=True) -> FormTemplate:
    return FormTemplate(
        id="t",
        name="Template",
        description="",
        column_widths=tuple(widths),
        uniform_row_height=uniform_row_height,
        columns=(),
    )


class TestTemplateGuidanceCoverage(unittest.TestCase):
    def test_grid_builder_skips_cells_starting_outside_image(self) -> None:
        grid = _grid_from_axis_coordinates([0, 5, 10], [0, 5, 20, 30], (6, 6))

        self.assertEqual(len(grid.cells), 4)

    def test_low_level_helpers_cover_degenerate_inputs(self) -> None:
        self.assertEqual(_proportional_columns(1, 5, (0.0, 0.0)), [1, 5])
        self.assertEqual(_regular_rows([3, 1]), [3, 1])
        self.assertEqual(_regular_rows([5, 4, 3]), [5, 4, 3])
        self.assertEqual(_regular_rows([0, 4, 10]), [0, 5, 10])
        self.assertEqual(_column_count_score(2, -1), 0.0)

    def test_apply_template_returns_original_for_empty_axes(self) -> None:
        grid = GridStructure(row_coords=[0], col_coords=[0, 10], cells=[])

        result = apply_template_to_grid(grid, _template(), (10, 10))

        self.assertIs(result.grid, grid)
        self.assertEqual(result.confidence, 0.0)
        self.assertFalse(result.repaired_columns)

    def test_apply_template_scores_uniform_row_variation(self) -> None:
        grid = GridStructure(
            row_coords=[0, 4, 10],
            col_coords=[0, 5, 10],
            cells=[],
        )

        result = apply_template_to_grid(grid, _template(), (10, 10))

        self.assertTrue(result.repaired_rows)
        self.assertLess(result.confidence, 1.0)
        self.assertEqual(result.grid.row_coords, [0, 5, 10])


if __name__ == "__main__":
    unittest.main()
