from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.table import TablePipelineResult, process_table_image


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


if __name__ == "__main__":
    unittest.main()
