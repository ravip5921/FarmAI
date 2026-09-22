from __future__ import annotations

import math
import unittest

import cv2
import numpy as np
from unittest.mock import patch

from src.core.image import DocumentImage
from src.preprocessing.skew import SkewCorrectionStage

from . import make_tiny_gray_image


class TestSkewStage(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = SkewCorrectionStage()

    def test_no_lines_returns_unchanged(self) -> None:
        doc = make_tiny_gray_image()

        result = self.stage.process(doc)

        self.assertFalse(result.metadata.get("deskewed"))
        self.assertEqual(result.metadata.get("skew_angle"), 0.0)
        np.testing.assert_array_equal(result.image, doc.image)

    def _make_skewed_lines(
        self, angle_deg: float, height: int = 200, width: int = 400, n: int = 5
    ) -> np.ndarray:
        """Create a white image with several parallel black lines at angle_deg."""
        img = np.full((height, width), 255, dtype=np.uint8)
        theta = math.radians(angle_deg)
        L = int(width * 0.9)
        dx = int(L * math.cos(theta))
        dy = int(L * math.sin(theta))
        start_x = 10
        ys = np.linspace(20, height - 20 - abs(dy), n, dtype=int)
        for y in ys:
            x1 = start_x
            y1 = int(y)
            x2 = start_x + dx
            y2 = y1 + dy
            cv2.line(img, (x1, y1), (x2, y2), color=0, thickness=3)
        return img

    def test_estimate_angle_detects_small_skew(self) -> None:
        # Create skewed near-horizontal lines at +5 degrees and ensure detection
        angle = 5.0
        img = self._make_skewed_lines(angle_deg=angle)
        doc = DocumentImage(image=img, metadata={})

        # use a more permissive hough threshold for test reliability
        stage = SkewCorrectionStage(hough_threshold=20)
        result = stage.process(doc)

        self.assertTrue(result.metadata.get("deskewed"))
        detected = float(result.metadata.get("skew_angle"))
        # Allow some tolerance due to discretization and Hough variability
        self.assertAlmostEqual(detected, angle, delta=1.0)
        self.assertEqual(result.metadata.get("skew_interpolation"), "nearest")
        self.assertTrue(set(np.unique(result.image).tolist()).issubset({0, 255}))

    def test_process_handles_color_image(self) -> None:
        # Same as above but with 3-channel BGR image to exercise color->gray branch
        angle = -3.0
        gray = self._make_skewed_lines(angle_deg=angle)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        doc = DocumentImage(image=bgr, metadata={})

        stage = SkewCorrectionStage(hough_threshold=20)
        result = stage.process(doc)

        self.assertTrue(result.metadata.get("deskewed"))
        detected = float(result.metadata.get("skew_angle"))
        self.assertAlmostEqual(detected, angle, delta=1.5)

    def test_vertical_lines_yield_no_valid_angles(self) -> None:
        # Draw long vertical lines so Hough finds lines but angles are ~90deg
        img = np.full((200, 400), 255, dtype=np.uint8)
        for x in range(50, 350, 50):
            cv2.line(img, (x, 10), (x, 190), color=0, thickness=3)

        doc = DocumentImage(image=img, metadata={})
        stage = SkewCorrectionStage(hough_threshold=20)
        result = stage.process(doc)

        # vertical-only lines should be ignored by angle filter and return deskewed False
        self.assertFalse(result.metadata.get("deskewed"))
        self.assertEqual(result.metadata.get("skew_angle"), 0.0)

    def test_steep_cell_marks_do_not_outvote_horizontal_table_lines(self) -> None:
        img = np.full((500, 800), 255, dtype=np.uint8)
        for y in range(50, 451, 50):
            cv2.line(img, (25, y), (775, y), color=0, thickness=2)
        for offset in range(20):
            start_x = 50 + offset * 20
            cv2.line(
                img,
                (start_x, 400),
                (start_x + 300, 100),
                color=0,
                thickness=2,
            )

        stage = SkewCorrectionStage(hough_threshold=20)

        self.assertAlmostEqual(stage.estimate_angle(img), 0.0, delta=0.5)

    def test_estimate_angle_accepts_flat_hough_line_array(self) -> None:
        img = np.full((100, 200), 255, dtype=np.uint8)
        lines = np.array([[10, 20, 190, 29], [10, 50, 190, 59]], dtype=np.int32)

        with patch("src.preprocessing.skew.cv2.HoughLinesP", return_value=lines):
            detected = self.stage.estimate_angle(img)

        self.assertAlmostEqual(detected, 2.86, delta=0.1)


if __name__ == "__main__":
    unittest.main()
