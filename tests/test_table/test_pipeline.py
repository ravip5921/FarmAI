from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from src.ocr.base import OcrText
from src.ocr.cell_ocr import recognize_table_cells
from src.table import (
    GridCell,
    GridStructure,
    TablePipelineResult,
    process_table_image,
    render_grid_overlay,
)
from src.table.line_detection import LineDetectionResult
from src.templates import load_template


class TestTablePipeline(unittest.TestCase):
    def test_farrowing_template_keeps_first_date_through_grid_and_ocr(self) -> None:
        class DateEngine:
            def recognize(self, image: np.ndarray) -> OcrText:
                return OcrText(text=f"{int(image[0, 0]):02d}-May")

        template = load_template("farrowing_room_4")
        for header_height in (14, 36):
            for scale, offset in ((1, 0), (2, 40)):
                with self.subTest(
                    header_height=header_height, scale=scale, offset=offset
                ):
                    horizontal = np.zeros((600 * scale, 600), dtype=np.uint8)
                    vertical = np.zeros_like(horizontal)
                    rows = [20] + [20 + header_height + i * 36 for i in range(6)]
                    rows = [offset + scale * y for y in rows]
                    for y in rows:
                        cv2.line(horizontal, (20, y), (550, y), 255)
                    for x in (20, 100, 200, 350, 550):
                        cv2.line(vertical, (x, rows[0]), (x, rows[-1]), 255)
                    source = np.full_like(horizontal, 255)
                    for row, (top, bottom) in enumerate(zip(rows, rows[1:])):
                        source[top:bottom, :] = row

                    with patch(
                        "src.table.detect_lines",
                        return_value=LineDetectionResult(horizontal, vertical),
                    ):
                        result = process_table_image(source, template=template)
                    table = recognize_table_cells(
                        source,
                        result.grid,
                        engine=DateEngine(),
                        column_names=template.column_names,
                        filter_out_column_indices=template.filtered_column_indices,
                    )

                    self.assertEqual(result.grid.row_coords, rows)
                    self.assertEqual(
                        [row[0] for row in table.text_matrix()],
                        ["Date", "01-May", "02-May", "03-May", "04-May", "05-May"],
                    )

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
        self.assertEqual(int(overlay[0, 3, 0]), 255)
        self.assertEqual(int(overlay[2, 0, 0]), 255)

    def test_render_grid_overlay_rejects_unsupported_dimensions(self) -> None:
        grid = GridStructure(row_coords=[], col_coords=[], cells=[])

        with self.assertRaisesRegex(
            ValueError,
            "render_grid_overlay expects a 2D or 3D image",
        ):
            render_grid_overlay(np.zeros((2, 2, 2, 2), dtype=np.uint8), grid)


if __name__ == "__main__":
    unittest.main()
