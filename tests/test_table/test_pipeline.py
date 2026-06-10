from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.table import (
    GridCell,
    GridStructure,
    TablePipelineResult,
    process_table_image,
    render_grid_overlay,
)


class TestTablePipeline(unittest.TestCase):
    def test_process_table_image_returns_grid_result(self) -> None:
        bitmap = np.array(
            [
                [0, 255, 0, 255],
                [0, 255, 0, 255],
                [0, 255, 0, 255],
                [0, 255, 0, 255],
            ],
            dtype=np.uint8,
        )

        result = process_table_image(bitmap, image_name="sample.jpg")

        self.assertIsInstance(result, TablePipelineResult)
        self.assertTrue(hasattr(result.line_detection, "horizontal_mask"))
        self.assertTrue(hasattr(result.intersections, "centroids"))
        self.assertTrue(hasattr(result.grid, "cells"))

    def test_process_table_image_saves_intermediate_images(self) -> None:
        bitmap = np.array(
            [
                [0, 255, 0, 255],
                [0, 255, 0, 255],
                [0, 255, 0, 255],
                [0, 255, 0, 255],
            ],
            dtype=np.uint8,
        )

        with TemporaryDirectory() as tmpdir:
            debug_dir = Path(tmpdir)
            result = process_table_image(
                bitmap,
                image_name="sample.jpg",
                debug_dir=debug_dir,
                save_line_detection=True,
                save_intersections=True,
            )

            self.assertIsInstance(result, TablePipelineResult)
            self.assertTrue((debug_dir / "line_detection_sample.png").exists())
            self.assertTrue((debug_dir / "intersection_sample.png").exists())

    def test_render_grid_overlay_draws_grid_on_image_copy(self) -> None:
        image = np.full((12, 12), 255, dtype=np.uint8)
        grid = GridStructure(
            row_coords=[2, 8],
            col_coords=[3, 9],
            cells=[GridCell(row=0, col=0, bbox=(3, 2, 6, 6))],
        )

        overlay = render_grid_overlay(image, grid, color=(0, 0, 255), thickness=1)

        self.assertEqual(overlay.shape, (12, 12, 3))
        self.assertTrue(np.array_equal(image, np.full((12, 12), 255, dtype=np.uint8)))
        self.assertLess(int(overlay[2, 3, 0]), 255)

    def test_render_grid_overlay_rejects_unsupported_dimensions(self) -> None:
        grid = GridStructure(row_coords=[], col_coords=[], cells=[])

        with self.assertRaisesRegex(
            ValueError,
            "render_grid_overlay expects a 2D or 3D image",
        ):
            render_grid_overlay(np.zeros((2, 2, 2, 2), dtype=np.uint8), grid)


if __name__ == "__main__":
    unittest.main()
