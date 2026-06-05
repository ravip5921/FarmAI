from __future__ import annotations

import unittest

import numpy as np

from src.analysis.connected_components import (
    connected_components,
    estimate_average_character_height,
    filter_components_by_area,
)


class TestConnectedComponents(unittest.TestCase):
    def test_connected_components_converts_boolean_input(self) -> None:
        binary = np.array(
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ],
            dtype=bool,
        )

        num_labels, labels, stats, centroids = connected_components(binary)

        self.assertEqual(num_labels, 2)
        self.assertEqual(labels.shape, binary.shape)
        self.assertEqual(stats.shape[0], 2)
        self.assertEqual(centroids.shape[0], 2)

    def test_filter_components_by_area_keeps_large_component_only(self) -> None:
        binary = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 255, 255, 0, 255, 0],
                [0, 255, 255, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        mask = filter_components_by_area(binary, min_area=3)

        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 255, 255, 0, 0, 0],
                [0, 255, 255, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(mask, expected)

    def test_estimate_average_character_height_returns_zero_when_empty(self) -> None:
        binary = np.zeros((5, 5), dtype=np.uint8)

        height = estimate_average_character_height(binary)

        self.assertEqual(height, 0)

    def test_estimate_average_character_height_ignores_small_components(self) -> None:
        binary = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 255, 255, 0, 255],
                [0, 255, 255, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        height = estimate_average_character_height(binary, min_area=2)

        self.assertEqual(height, 2)


if __name__ == "__main__":
    unittest.main()
