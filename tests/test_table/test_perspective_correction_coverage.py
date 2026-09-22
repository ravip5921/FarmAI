from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.table.perspective_correction import (
    find_table_corners,
    render_perspective_corners,
    warp_image_to_corners,
)


class TestPerspectiveCorrectionCoverage(unittest.TestCase):
    def test_find_table_corners_accepts_float_masks_and_rejects_non_2d_masks(
        self,
    ) -> None:
        self.assertIsNone(
            find_table_corners(
                np.zeros((10, 10), dtype=np.float32),
                np.zeros((10, 10), dtype=np.float32),
            )
        )

        with self.assertRaisesRegex(ValueError, "2D masks"):
            find_table_corners(
                np.zeros((1, 10, 10), dtype=np.uint8),
                np.zeros((10, 10), dtype=np.uint8),
            )

    def test_find_table_corners_skips_small_and_degenerate_contours(self) -> None:
        mask = np.zeros((30, 30), dtype=np.uint8)
        cv2.rectangle(mask, (1, 1), (2, 2), 255, thickness=-1)

        self.assertIsNone(find_table_corners(mask, mask, min_area_ratio=0.5))

        degenerate = np.array([[[0, 0]], [[0, 0]], [[0, 0]]], dtype=np.int32)
        with (
            patch("src.table.perspective_correction.cv2.findContours") as contours,
            patch("src.table.perspective_correction.cv2.contourArea", return_value=50),
            patch("src.table.perspective_correction.cv2.arcLength", return_value=0),
        ):
            contours.return_value = ([degenerate], None)
            self.assertIsNone(
                find_table_corners(
                    np.ones((10, 10), dtype=np.uint8),
                    np.ones((10, 10), dtype=np.uint8),
                    min_area_ratio=0.01,
                )
            )

    def test_warp_color_image_uses_linear_path_and_render_handles_color_no_corners(
        self,
    ) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        corners = np.array([[1, 1], [6, 1], [6, 6], [1, 6]], dtype=np.float32)

        warped = warp_image_to_corners(image, corners, output_size=(4, 4))
        overlay = render_perspective_corners(image, corners=None)

        self.assertTrue(warped.corrected)
        self.assertEqual(warped.image.shape, (4, 4, 3))
        self.assertEqual(overlay.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
