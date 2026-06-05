from __future__ import annotations

import unittest

import numpy as np

from src.analysis.character_size import estimate_character_size


class TestCharacterSize(unittest.TestCase):
    def test_estimate_character_size_returns_zero_report_when_empty(self) -> None:
        binary = np.zeros((6, 6), dtype=np.uint8)

        report = estimate_character_size(binary)

        self.assertEqual(report.average_height, 0)
        self.assertEqual(report.median_height, 0)
        self.assertEqual(report.component_count, 0)

    def test_estimate_character_size_tracks_heights(self) -> None:
        binary = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 255, 255, 0, 255, 0],
                [0, 255, 255, 0, 255, 0],
                [0, 255, 255, 0, 255, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        report = estimate_character_size(binary, min_area=1)

        self.assertEqual(report.component_count, 2)
        self.assertEqual(report.average_height, 3)
        self.assertEqual(report.median_height, 3)

    def test_estimate_character_size_ignores_small_components(self) -> None:
        binary = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 255, 255, 0, 255],
                [0, 255, 255, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        report = estimate_character_size(binary, min_area=2)

        self.assertEqual(report.component_count, 1)
        self.assertEqual(report.average_height, 2)
        self.assertEqual(report.median_height, 2)

    def test_estimate_character_size_converts_boolean_input(self) -> None:
        binary = np.array(
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ],
            dtype=bool,
        )

        report = estimate_character_size(binary, min_area=1)

        self.assertEqual(report.component_count, 1)
        self.assertEqual(report.average_height, 2)
        self.assertEqual(report.median_height, 2)


if __name__ == "__main__":
    unittest.main()
