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

        self.assertTrue(np.allclose(smoothed, np.array([1.0, 1.0, 1.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
