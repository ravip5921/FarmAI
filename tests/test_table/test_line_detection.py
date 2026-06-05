from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.core.image import DocumentImage
from src.table.line_detection import LineDetectionStage, detect_lines


class TestLineDetection(unittest.TestCase):
    def test_process_extracts_horizontal_and_vertical_masks(self) -> None:
        image = np.full((60, 60), 255, dtype=np.uint8)
        cv2.line(image, (5, 20), (55, 20), color=0, thickness=2)
        cv2.line(image, (30, 5), (30, 55), color=0, thickness=2)

        doc = DocumentImage(image=image, metadata={})
        stage = LineDetectionStage(min_line_scale=1.0)
        result = stage.process(doc)

        self.assertEqual(result.image.shape, image.shape)
        self.assertIn("line_detection", result.metadata)
        self.assertIn("horizontal_mask", result.metadata)
        self.assertIn("vertical_mask", result.metadata)
        self.assertGreater(int(np.count_nonzero(result.metadata["horizontal_mask"])), 0)
        self.assertGreater(int(np.count_nonzero(result.metadata["vertical_mask"])), 0)

    def test_process_converts_color_input_to_binary(self) -> None:
        gray = np.full((40, 40), 255, dtype=np.uint8)
        cv2.line(gray, (5, 10), (35, 10), color=0, thickness=2)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        doc = DocumentImage(image=bgr, metadata={})
        stage = LineDetectionStage(min_line_scale=1.0)
        result = stage.process(doc)

        self.assertEqual(result.image.dtype, np.uint8)
        self.assertEqual(result.image.shape, gray.shape)
        self.assertTrue(
            np.array_equal(np.unique(result.image), np.array([0, 255], dtype=np.uint8))
        )

    def test_process_converts_boolean_input_to_binary(self) -> None:
        binary = np.array(
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ],
            dtype=bool,
        )

        doc = DocumentImage(image=binary, metadata={})
        stage = LineDetectionStage(min_line_scale=1.0)
        result = stage.process(doc)

        self.assertEqual(result.image.dtype, np.uint8)
        self.assertTrue(
            np.array_equal(np.unique(result.image), np.array([0, 255], dtype=np.uint8))
        )

    def test_detect_lines_returns_result_dataclass(self) -> None:
        image = np.full((50, 50), 255, dtype=np.uint8)
        cv2.line(image, (5, 25), (45, 25), color=0, thickness=2)

        result = detect_lines(image, min_line_scale=1.0)

        self.assertIsInstance(result.horizontal_mask, np.ndarray)
        self.assertIsInstance(result.vertical_mask, np.ndarray)
        self.assertEqual(result.horizontal_mask.shape, image.shape)

    def test_close_gaps_horizontal_fills_small_gap(self) -> None:
        image = np.zeros((5, 5), dtype=np.uint8)
        image[2, 1] = 255
        image[2, 3] = 255

        stage = LineDetectionStage()

        result = stage._close_gaps(
            image,
            orientation="horizontal",
            gap=3,
        )

        self.assertEqual(int(result[2, 2]), 255)

    def test_close_gaps_vertical_fills_small_gap(self) -> None:
        image = np.zeros((5, 5), dtype=np.uint8)
        image[1, 2] = 255
        image[3, 2] = 255

        stage = LineDetectionStage()

        result = stage._close_gaps(
            image,
            orientation="vertical",
            gap=3,
        )

        self.assertEqual(int(result[2, 2]), 255)


if __name__ == "__main__":
    unittest.main()
