from __future__ import annotations

import unittest

from src.table.grid_reconstruction import (
    GridCell,
    _cluster_sorted_values,
    reconstruct_grid,
)


class TestGridReconstruction(unittest.TestCase):
    def test_reconstruct_grid_returns_empty_structure_for_no_centroids(self) -> None:
        result = reconstruct_grid([], image_shape=(100, 100))

        self.assertEqual(result.row_coords, [])
        self.assertEqual(result.col_coords, [])
        self.assertEqual(result.cells, [])

    def test_reconstruct_grid_clusters_centroids_and_builds_cells(self) -> None:
        centroids = [
            (10, 10),
            (11, 12),
            (50, 11),
            (51, 12),
            (10, 50),
            (12, 51),
            (50, 50),
            (52, 51),
        ]

        result = reconstruct_grid(centroids, image_shape=(100, 100), tolerance=5)

        self.assertEqual(result.row_coords, [11, 50])
        self.assertEqual(result.col_coords, [11, 51])
        self.assertEqual(len(result.cells), 1)
        self.assertEqual(result.cells[0], GridCell(row=0, col=0, bbox=(11, 11, 40, 39)))

    def test_reconstruct_grid_clips_cells_to_image_bounds(self) -> None:
        centroids = [
            (5, 5),
            (95, 5),
            (5, 95),
            (95, 95),
        ]

        result = reconstruct_grid(centroids, image_shape=(50, 50), tolerance=5)

        self.assertEqual(len(result.cells), 1)
        self.assertEqual(result.cells[0].bbox, (5, 5, 45, 45))

    def test_reconstruct_grid_handles_negative_coordinates(self) -> None:
        centroids = [
            (-5, -5),
            (20, -5),
            (-5, 20),
            (20, 20),
        ]

        result = reconstruct_grid(centroids, image_shape=(50, 50), tolerance=5)

        self.assertEqual(result.cells[0].bbox, (0, 0, 25, 25))

    def test_cluster_sorted_values_returns_empty_list(self) -> None:
        self.assertEqual(_cluster_sorted_values([]), [])

    def test_reconstruct_grid_skips_cells_outside_image_bounds(self) -> None:
        centroids = [
            (60, 60),
            (70, 60),
            (60, 70),
            (70, 70),
        ]

        result = reconstruct_grid(centroids, image_shape=(50, 50), tolerance=5)

        self.assertEqual(result.row_coords, [60, 70])
        self.assertEqual(result.col_coords, [60, 70])
        self.assertEqual(result.cells, [])

    def test_no_tolerance(self) -> None:
        centroids = [
            (60, 60),
            (70, 60),
            (60, 70),
            (70, 70),
        ]

        result = reconstruct_grid(centroids, image_shape=(50, 50))

        self.assertEqual(result.row_coords, [65])
        self.assertEqual(result.col_coords, [65])
        self.assertEqual(result.cells, [])


if __name__ == "__main__":
    unittest.main()
