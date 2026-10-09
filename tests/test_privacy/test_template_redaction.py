from __future__ import annotations

import unittest

import numpy as np

from src.privacy import (
    redact_filtered_template_columns,
    redacted_template_column_indices,
)
from src.table.grid_reconstruction import GridStructure
from src.templates import load_template


class TestTemplateRedaction(unittest.TestCase):
    def test_empty_and_outside_grid_are_unchanged(self) -> None:
        template = load_template("boar_room")
        image = np.full((10, 10, 3), 255, dtype=np.uint8)
        grid = GridStructure(row_coords=[0, 10], col_coords=[0, 2], cells=[])
        self.assertEqual(
            redacted_template_column_indices(
                template, extra_filtered_columns={"comments"}
            ),
            {5, 6, 7, 9},
        )
        redacted = redact_filtered_template_columns(image, grid, template)
        self.assertTrue(np.array_equal(redacted, image))
        grid = GridStructure(row_coords=[0], col_coords=[0, 10], cells=[])
        self.assertTrue(
            np.array_equal(
                redact_filtered_template_columns(image, grid, template), image
            )
        )
        grid = GridStructure(
            row_coords=[0, 10], col_coords=[0, 2, 4, 6, 8, 10, 12], cells=[]
        )
        redacted = redact_filtered_template_columns(image, grid, template)
        self.assertTrue(np.array_equal(redacted, image))
        grid = GridStructure(row_coords=[10, 12], col_coords=list(range(11)), cells=[])
        redacted = redact_filtered_template_columns(image, grid, template)
        self.assertTrue(np.array_equal(redacted, image))

    def test_redacted_indices_skip_only_dividers(self) -> None:
        template = load_template("boar_room")

        self.assertEqual(redacted_template_column_indices(template), {5, 7, 9})

    def test_redacts_entire_sensitive_column_span(self) -> None:
        template = load_template("boar_room")
        image = np.full((20, 40, 3), 255, dtype=np.uint8)
        grid = GridStructure(
            row_coords=[2, 10, 18],
            col_coords=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
            cells=[],
        )

        redacted = redact_filtered_template_columns(image, grid, template)

        self.assertTrue(np.all(redacted[2:18, 20:24] == 0))
        self.assertTrue(np.all(redacted[2:18, 28:32] == 0))
        self.assertTrue(np.all(redacted[2:18, 36:40] == 0))
        self.assertTrue(np.all(redacted[2:18, 16:20] == 255))


if __name__ == "__main__":
    unittest.main()
