from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.table.perspective_correction import (
    correct_table_perspective,
    expand_corners,
    find_table_corners,
    render_perspective_corners,
)


class TestPerspectiveCorrection(unittest.TestCase):
    def test_expand_corners_moves_points_outward_and_clips_to_image(self) -> None:
        corners = np.array([[10, 10], [40, 10], [40, 30], [10, 30]], dtype=np.float32)

        expanded = expand_corners(corners, (35, 45), padding=10)

        self.assertLessEqual(expanded[0, 0], corners[0, 0])
        self.assertLessEqual(expanded[0, 1], corners[0, 1])
        self.assertGreaterEqual(expanded[2, 0], corners[2, 0])
        self.assertGreaterEqual(expanded[2, 1], corners[2, 1])
        self.assertGreaterEqual(float(expanded.min()), 0.0)
        self.assertLessEqual(float(expanded[:, 0].max()), 44.0)
        self.assertLessEqual(float(expanded[:, 1].max()), 34.0)

    def test_find_table_corners_from_closed_table_border(self) -> None:
        horizontal = np.zeros((100, 140), dtype=np.uint8)
        vertical = np.zeros_like(horizontal)
        corners = np.array([[20, 10], [120, 15], [110, 90], [15, 80]], np.int32)
        cv2.polylines(horizontal, [corners], isClosed=True, color=255, thickness=3)
        cv2.polylines(vertical, [corners], isClosed=True, color=255, thickness=3)

        found = find_table_corners(horizontal, vertical, min_area_ratio=0.01)

        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.shape, (4, 2))
        self.assertLess(found[0, 0], found[1, 0])
        self.assertLess(found[0, 1], found[3, 1])

    def test_correct_table_perspective_warps_with_padded_corners(self) -> None:
        image = np.full((100, 140), 255, dtype=np.uint8)
        horizontal = np.zeros_like(image)
        vertical = np.zeros_like(image)
        corners = np.array([[20, 10], [120, 15], [110, 90], [15, 80]], np.int32)
        cv2.polylines(horizontal, [corners], isClosed=True, color=255, thickness=3)
        cv2.polylines(vertical, [corners], isClosed=True, color=255, thickness=3)

        unpadded = correct_table_perspective(image, horizontal, vertical, padding=0)
        padded = correct_table_perspective(image, horizontal, vertical, padding=12)

        self.assertTrue(padded.corrected)
        self.assertIsNotNone(padded.transform)
        self.assertIsNotNone(padded.output_size)
        self.assertIsNotNone(padded.padded_corners)
        self.assertGreaterEqual(padded.image.shape[1], unpadded.image.shape[1])
        self.assertGreaterEqual(padded.image.shape[0], unpadded.image.shape[0])

    def test_correct_table_perspective_keeps_image_when_no_corners_are_found(
        self,
    ) -> None:
        image = np.full((20, 20), 255, dtype=np.uint8)

        result = correct_table_perspective(
            image,
            np.zeros_like(image),
            np.zeros_like(image),
        )

        self.assertFalse(result.corrected)
        self.assertIs(result.image, image)

    def test_render_perspective_corners_returns_color_overlay(self) -> None:
        image = np.full((20, 20), 255, dtype=np.uint8)
        corners = np.array([[2, 2], [17, 2], [17, 17], [2, 17]], dtype=np.float32)
        padded = expand_corners(corners, image.shape, padding=2)

        overlay = render_perspective_corners(image, corners, padded_corners=padded)

        self.assertEqual(overlay.shape, (20, 20, 3))


if __name__ == "__main__":
    unittest.main()
