from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.core.image import DocumentImage
from src.table.line_detection import LineDetectionStage


class TestLineDetectionHelpers(unittest.TestCase):
    def test_kernel_size_uses_average_character_height(self) -> None:
        stage = LineDetectionStage(min_line_scale=1.8)
        binary = np.zeros((10, 10), dtype=np.uint8)
        report = SimpleNamespace(average_height=7, median_height=4)

        with patch(
            "src.table.line_detection.estimate_character_size", return_value=report
        ):
            self.assertEqual(stage._kernel_size(binary), 12)

    def test_kernel_size_falls_back_to_median_height(self) -> None:
        stage = LineDetectionStage(min_line_scale=2.0)
        binary = np.zeros((10, 10), dtype=np.uint8)
        report = SimpleNamespace(average_height=0, median_height=5)

        with patch(
            "src.table.line_detection.estimate_character_size", return_value=report
        ):
            self.assertEqual(stage._kernel_size(binary), 10)

    def test_hough_vertical_mask_filters_non_vertical_segments(self) -> None:
        stage = LineDetectionStage()
        binary = np.zeros((20, 20), dtype=np.uint8)
        lines = np.array(
            [
                [[2, 1, 2, 15]],
                [[1, 1, 10, 2]],
            ],
            dtype=np.int32,
        )

        with patch("src.table.line_detection.cv2.HoughLinesP", return_value=lines):
            mask = stage._hough_vertical_mask(binary, min_line_length=5)

        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertEqual(int(mask[2:15, 2].sum()), 255 * 13)

    def test_hough_vertical_mask_returns_empty_when_no_lines(self) -> None:
        stage = LineDetectionStage()
        binary = np.zeros((20, 20), dtype=np.uint8)

        with patch("src.table.line_detection.cv2.HoughLinesP", return_value=None):
            mask = stage._hough_vertical_mask(binary, min_line_length=5)

        self.assertEqual(int(np.count_nonzero(mask)), 0)

    def test_process_recovers_sparse_verticals_from_hough(self) -> None:
        stage = LineDetectionStage(min_line_scale=1.55)
        binary = np.zeros((20, 20), dtype=np.uint8)
        report = SimpleNamespace(average_height=0, median_height=0, component_count=0)
        hough_mask = np.zeros((20, 20), dtype=np.uint8)
        hough_mask[:, 10] = 255

        with (
            patch(
                "src.table.line_detection.estimate_character_size", return_value=report
            ),
            patch.object(
                LineDetectionStage,
                "_hough_vertical_mask",
                return_value=hough_mask,
            ),
            patch.object(
                LineDetectionStage,
                "_retain_vertical_components_with_crossings",
                return_value=hough_mask,
            ),
        ):
            result = stage.process(DocumentImage(binary, {}))

        self.assertGreater(int(np.count_nonzero(result.metadata["vertical_mask"])), 0)

    def test_retain_horizontal_components_requires_vertical_crossings(self) -> None:
        stage = LineDetectionStage()
        horizontal = np.zeros((30, 40), dtype=np.uint8)
        vertical = np.zeros((30, 40), dtype=np.uint8)

        horizontal[10, 2:32] = 255
        horizontal[20, 22:34] = 255
        for x in (4, 16, 28):
            vertical[4:26, x] = 255

        filtered = stage._retain_horizontal_components_with_crossings(
            horizontal,
            vertical,
            min_crossings=2,
        )

        self.assertGreater(int(np.count_nonzero(filtered[10])), 0)
        self.assertEqual(int(np.count_nonzero(filtered[20])), 0)

    def test_retain_horizontal_components_keeps_top_border_with_endpoint_gap(
        self,
    ) -> None:
        stage = LineDetectionStage()
        horizontal = np.zeros((40, 50), dtype=np.uint8)
        vertical = np.zeros((40, 50), dtype=np.uint8)

        horizontal[10, 5:46] = 255
        for x in (5, 25, 45):
            vertical[12:35, x] = 255

        filtered = stage._retain_horizontal_components_with_crossings(
            horizontal,
            vertical,
            min_crossings=2,
        )

        self.assertGreater(int(np.count_nonzero(filtered[10])), 0)


if __name__ == "__main__":
    unittest.main()
