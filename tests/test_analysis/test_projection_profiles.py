from __future__ import annotations

import unittest

import numpy as np

from src.analysis.projection_profiles import (
    find_projection_peaks,
    peak_positions,
    projection_profile,
    smooth_profile,
)


class TestProjectionProfiles(unittest.TestCase):
    def test_projection_profile_counts_rows_and_columns(self) -> None:
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1, 1:4] = 255
        mask[2, 3] = 255

        horizontal = projection_profile(mask, "horizontal")
        vertical = projection_profile(mask, "vertical")

        self.assertEqual(horizontal.tolist(), [0, 3, 1, 0])
        self.assertEqual(vertical.tolist(), [0, 1, 1, 2, 0])

    def test_find_projection_peaks_clusters_adjacent_bins(self) -> None:
        profile = np.array([0, 0, 5, 8, 6, 0, 0, 7, 0], dtype=np.int32)

        peaks = find_projection_peaks(profile, min_peak_ratio=0.5, max_gap=1)

        self.assertEqual(peak_positions(peaks), [3, 7])
        self.assertEqual((peaks[0].start, peaks[0].end), (2, 4))

    def test_smooth_profile_returns_moving_average(self) -> None:
        profile = np.array([0, 3, 0], dtype=np.int32)

        smoothed = smooth_profile(profile, window_size=3)

        self.assertTrue(
            np.allclose(smoothed, np.array([1.0, 1.0, 1.0], dtype=np.float32))
        )

    def test_projection_profile_rejects_non_2d_input(self) -> None:
        with self.assertRaises(ValueError):
            projection_profile(np.zeros(5, dtype=np.uint8), "horizontal")

    def test_projection_profile_rejects_invalid_axis(self) -> None:
        mask = np.zeros((2, 2), dtype=np.uint8)

        with self.assertRaises(ValueError):
            projection_profile(mask, "diagonal")  # type: ignore[arg-type]

    def test_smooth_profile_rejects_non_1d_input(self) -> None:
        with self.assertRaises(ValueError):
            smooth_profile(np.zeros((2, 2), dtype=np.int32))

    def test_smooth_profile_returns_copy_for_window_size_one(self) -> None:
        profile = np.array([1, 2, 3], dtype=np.int32)

        smoothed = smooth_profile(profile, window_size=1)

        self.assertTrue(np.array_equal(smoothed, profile))
        self.assertIsNot(smoothed, profile)

    def test_find_projection_peaks_rejects_non_1d_input(self) -> None:
        with self.assertRaises(ValueError):
            find_projection_peaks(np.zeros((2, 2), dtype=np.int32))

    def test_find_projection_peaks_empty_profile(self) -> None:
        peaks = find_projection_peaks(np.array([], dtype=np.int32))

        self.assertEqual(peaks, [])

    def test_find_projection_peaks_returns_empty_when_threshold_filters_all(
        self,
    ) -> None:
        profile = np.array([1, 2, 3], dtype=np.int32)

        peaks = find_projection_peaks(
            profile,
            min_peak_value=10,
        )

        self.assertEqual(peaks, [])


if __name__ == "__main__":
    unittest.main()
