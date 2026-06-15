from __future__ import annotations

import unittest

import cv2
import numpy as np

import src.table.line_refinement as line_refinement


class LateStructuralHeaderSupport:
    def __init__(self, values: list[int], late_index: int) -> None:
        self.values = values
        self.late_index = late_index
        self.calls_by_index = [0 for _ in values]

    def __getitem__(self, index: int) -> int:
        self.calls_by_index[index] += 1
        if index == self.late_index and self.calls_by_index[index] > 1:
            return 2
        return self.values[index]


class DiffEmptyColumns(list[int]):
    def __array__(self, dtype: object | None = None) -> np.ndarray:
        return np.array([], dtype=np.int64)


class TestLineRefinement(unittest.TestCase):
    def test_refinement_fills_missing_regular_row_and_recovers_endpoint_column(
        self,
    ) -> None:
        horizontal = np.zeros((24, 24), dtype=np.uint8)
        vertical = np.zeros((24, 24), dtype=np.uint8)

        cv2.line(horizontal, (1, 1), (20, 1), color=255, thickness=1)
        for y in (4, 8, 16, 20):
            cv2.line(horizontal, (2, y), (18, y), color=255, thickness=1)

        cv2.line(vertical, (3, 4), (3, 20), color=255, thickness=1)
        cv2.line(vertical, (4, 4), (4, 20), color=255, thickness=1)
        cv2.line(vertical, (12, 4), (12, 20), color=255, thickness=1)
        cv2.line(vertical, (19, 2), (19, 5), color=255, thickness=1)

        result = line_refinement.refine_grid_with_projection_profiles(
            horizontal, vertical, horizontal.shape
        )

        self.assertEqual(result.grid.row_coords, [4, 8, 12, 16, 20])
        self.assertEqual(result.grid.col_coords, [4, 12, 18])
        self.assertEqual(len(result.grid.cells), 8)
        self.assertEqual(result.estimated_row_spacing, 4)

    def test_grid_from_axis_coordinates_skips_out_of_bounds_cells(self) -> None:

        # Image is small
        shape = (10, 10)

        # Force one valid cell and one out-of-bounds cell
        row_coords = [12, 12]  # bottom row is within bounds, top edge is fine
        col_coords = [12, 12]  # right edge is out of bounds

        grid = line_refinement._grid_from_axis_coordinates(
            row_coords, col_coords, shape
        )

        # No cell should be created
        self.assertEqual(len(grid.cells), 0)

    def test_estimate_regular_spacing_returns_zero_with_fewer_than_three_coords(
        self,
    ) -> None:
        self.assertEqual(
            line_refinement._estimate_regular_spacing([10, 20], axis_length=100),
            0,
        )

    def test_estimate_regular_spacing_returns_zero_when_no_gaps_are_usable(
        self,
    ) -> None:
        # For axis_length=120:
        # min_spacing = max(3, 120 // 120) = 3
        # max_spacing = max(4, 120 // 12) = 10
        #
        # Gaps here are 20 and 20, so none fall within [3, 10].
        self.assertEqual(
            line_refinement._estimate_regular_spacing([0, 20, 40], axis_length=120),
            0,
        )

    def test_merge_close_positions_returns_empty_for_no_coords(self) -> None:
        self.assertEqual(
            line_refinement._merge_close_positions([], spacing=5),
            [],
        )

    def test_merge_close_positions_returns_sorted_coords_when_spacing_is_not_positive(
        self,
    ) -> None:
        self.assertEqual(
            line_refinement._merge_close_positions([10, 2, 6], spacing=0),
            [2, 6, 10],
        )

    def test_merge_axis_candidates_returns_empty_for_no_coords(self) -> None:
        self.assertEqual(
            line_refinement._merge_axis_candidates([], tolerance=3),
            [],
        )

    def test_merge_axis_candidates_clusters_within_tolerance_and_splits_others(
        self,
    ) -> None:
        self.assertEqual(
            line_refinement._merge_axis_candidates([10, 12, 30, 31], tolerance=3),
            [11, 30],
        )

    def test_fill_missing_regular_positions_returns_sorted_coords_for_early_exit_cases(
        self,
    ) -> None:
        cases = [
            ([8], 4, [8]),  # len(coords) < 2
            ([10, 2, 6], 0, [2, 6, 10]),  # spacing <= 0
        ]

        for coords, spacing, expected in cases:
            with self.subTest(coords=coords, spacing=spacing):
                self.assertEqual(
                    line_refinement._fill_missing_regular_positions(coords, spacing),
                    expected,
                )

    def test_trim_row_candidates_returns_original_rows_when_trimmed_table_has_less_than_two_rows(
        self,
    ) -> None:
        rows, spacing = line_refinement._trim_row_candidates(
            rows=[0, 4],
            row_support=[0, 2],
            axis_length=48,
        )

        self.assertEqual(rows, [0, 4])
        self.assertEqual(spacing, 0)

    def test_trim_row_candidates_stops_at_large_gap(self) -> None:
        rows, spacing = line_refinement._trim_row_candidates(
            rows=[0, 4, 8, 12, 40, 44],
            row_support=[2, 2, 2, 2, 2, 2],
            axis_length=120,
        )

        self.assertEqual(rows, [0, 4, 8, 12])
        self.assertEqual(spacing, 4)

    def test_filter_column_candidates_returns_original_cols_when_filtered_has_less_than_two(
        self,
    ) -> None:
        cols = [2, 6, 10]
        filtered = line_refinement._filter_column_candidates(
            cols=cols,
            col_support=[2, 0, 0],
            row_count=12,
        )

        self.assertEqual(filtered, cols)

    def test_line_refinement_helper_early_exits_and_supported_subset(self) -> None:
        self.assertEqual(
            line_refinement._filter_column_candidates([4], [0], row_count=12),
            [4],
        )
        self.assertEqual(
            line_refinement._filter_column_candidates(
                [4, 12, 20], [2, 0, 3], row_count=12
            ),
            [4, 20],
        )

        vertical = np.zeros((5, 5), dtype=np.uint8)
        self.assertEqual(
            line_refinement._column_segment_support(
                vertical,
                rows=[0, 1],
                cols=[10],
                radius=1,
            ),
            [0],
        )
        self.assertEqual(
            line_refinement._column_segment_support(
                vertical,
                rows=[6, 7],
                cols=[2],
                radius=1,
            ),
            [0],
        )
        self.assertEqual(
            line_refinement._dedupe_close_columns(
                cols=[4],
                crossing_support=[1],
                segment_support=[1],
                endpoint_support=[1],
                min_separation=3,
            ),
            [4],
        )
        self.assertEqual(
            line_refinement._horizontal_endpoint_candidates(
                np.zeros((5, 8), dtype=np.uint8),
                rows=[10],
                radius=1,
                min_run_length=2,
                tolerance=1,
            ),
            ([], []),
        )

    def test_column_segment_support_ignores_intersection_only_marks(self) -> None:
        vertical = np.zeros((30, 24), dtype=np.uint8)
        rows = [2, 8, 14, 20, 26]

        cv2.line(vertical, (4, 2), (4, 26), color=255, thickness=1)
        for y in rows:
            cv2.line(vertical, (12, y - 1), (12, y + 1), color=255, thickness=1)

        support = line_refinement._column_segment_support(
            vertical,
            rows,
            [4, 12],
            radius=2,
        )

        self.assertEqual(support, [4, 0])

    def test_top_row_column_support_distinguishes_border_from_header_text(
        self,
    ) -> None:
        horizontal = np.zeros((20, 30), dtype=np.uint8)
        vertical = np.zeros((20, 30), dtype=np.uint8)

        for y in (2, 14):
            cv2.line(horizontal, (2, y), (24, y), color=255, thickness=1)
        cv2.line(vertical, (4, 2), (4, 14), color=255, thickness=1)
        cv2.line(vertical, (12, 5), (12, 11), color=255, thickness=1)

        support = line_refinement._top_row_column_support(
            horizontal,
            vertical,
            rows=[2, 14],
            cols=[4, 12, 40],
            radius=1,
        )

        self.assertEqual(support, [3, 0, 0])
        self.assertEqual(
            line_refinement._top_row_column_support(
                horizontal,
                vertical,
                rows=[2],
                cols=[4],
                radius=1,
            ),
            [0],
        )
        self.assertEqual(
            line_refinement._top_row_column_support(
                np.zeros((0, 30), dtype=np.uint8),
                np.zeros((0, 30), dtype=np.uint8),
                rows=[0, 1],
                cols=[4],
                radius=1,
            ),
            [0],
        )

    def test_top_row_column_candidates_recover_profile_and_endpoint_axes(
        self,
    ) -> None:
        horizontal = np.zeros((20, 30), dtype=np.uint8)
        vertical = np.zeros((20, 30), dtype=np.uint8)

        for y in (2, 14):
            cv2.line(horizontal, (3, y), (24, y), color=255, thickness=1)
        cv2.line(vertical, (10, 2), (10, 14), color=255, thickness=1)

        candidates, support = line_refinement._top_row_column_candidates(
            vertical,
            horizontal,
            rows=[2, 14],
            radius=1,
            tolerance=1,
        )

        self.assertEqual(candidates, [3, 10, 24])
        self.assertEqual(support, [0, 3, 0])
        self.assertEqual(
            line_refinement._top_row_column_candidates(
                vertical,
                horizontal,
                rows=[2],
                radius=1,
                tolerance=1,
            ),
            ([], []),
        )
        self.assertEqual(
            line_refinement._top_row_column_candidates(
                np.zeros((0, 30), dtype=np.uint8),
                np.zeros((0, 30), dtype=np.uint8),
                rows=[0, 1],
                radius=1,
                tolerance=1,
            ),
            ([], []),
        )

    def test_filter_columns_by_table_support_uses_header_gate(
        self,
    ) -> None:
        filtered = line_refinement._filter_columns_by_table_support(
            cols=[4, 12, 20],
            crossing_support=[4, 0, 4],
            segment_support=[4, 1, 4],
            endpoint_support=[0, 0, 0],
            header_support=[3, 0, 3],
            row_count=5,
        )

        self.assertEqual(filtered, [4, 20])

    def test_filter_columns_by_table_support_keeps_legacy_evidence_without_header(
        self,
    ) -> None:
        filtered = line_refinement._filter_columns_by_table_support(
            cols=[4, 12, 20],
            crossing_support=[1, 0, 0],
            segment_support=[0, 1, 0],
            endpoint_support=[0, 0, 1],
            header_support=[0, 0, 0],
            row_count=5,
        )

        self.assertEqual(filtered, [4, 12, 20])
        self.assertEqual(
            line_refinement._filter_columns_by_table_support(
                cols=[4, 12, 20],
                crossing_support=[2, 0, 0],
                segment_support=[0, 0, 0],
                endpoint_support=[0, 0, 0],
                header_support=[0, 0, 0],
                row_count=12,
            ),
            [4, 12, 20],
        )

    def test_refinement_filters_lower_row_text_vertical_with_header_anchors(
        self,
    ) -> None:
        horizontal = np.zeros((30, 40), dtype=np.uint8)
        vertical = np.zeros((30, 40), dtype=np.uint8)

        for y in (2, 8, 14, 20, 26):
            cv2.line(horizontal, (4, y), (34, y), color=255, thickness=1)
        for x in (4, 20, 34):
            cv2.line(vertical, (x, 2), (x, 26), color=255, thickness=1)
        cv2.line(vertical, (12, 10), (12, 12), color=255, thickness=1)

        result = line_refinement.refine_grid_with_projection_profiles(
            horizontal, vertical, horizontal.shape
        )

        self.assertEqual(result.grid.col_coords, [4, 20, 34])
        self.assertNotIn(12, result.grid.col_coords)
        self.assertEqual(
            result.col_segment_support[result.col_candidates.index(12)],
            1,
        )
        self.assertLess(
            result.col_header_support[result.col_candidates.index(12)],
            2,
        )

    def test_dedupe_close_columns_keeps_best_supported_axis(self) -> None:
        deduped = line_refinement._dedupe_close_columns(
            cols=[4, 11, 13, 24],
            crossing_support=[5, 2, 5, 4],
            segment_support=[4, 1, 4, 3],
            endpoint_support=[4, 1, 4, 3],
            min_separation=4,
        )

        self.assertEqual(deduped, [4, 13, 24])

    def test_dedupe_close_columns_prefers_header_supported_axis(self) -> None:
        deduped = line_refinement._dedupe_close_columns(
            cols=[100, 118, 140],
            crossing_support=[8, 9, 5],
            segment_support=[8, 8, 4],
            endpoint_support=[0, 0, 0],
            header_support=[2, 0, 1],
            min_separation=20,
        )

        self.assertEqual(deduped, [100, 140])

    def test_prune_columns_inside_wide_spans_removes_handwriting_axes(self) -> None:
        pruned = line_refinement._prune_columns_inside_wide_spans(
            cols=[330, 445, 545, 680, 720, 820, 930, 1050, 2390, 2480, 2560],
            crossing_support=[5, 7, 4, 10, 2, 12, 15, 6, 11, 4, 2],
            segment_support=[10, 26, 4, 26, 4, 26, 26, 2, 9, 3, 0],
            endpoint_support=[5, 2, 1, 1, 0, 10, 1, 0, 1, 0, 8],
            header_support=[1, 1, 2, 2, 0, 1, 1, 0, 2, 0, 0],
            image_width=3072,
            row_spacing=79,
        )

        self.assertEqual(pruned, [330, 445, 545, 680, 2390, 2480, 2560])

    def test_prune_columns_inside_wide_spans_returns_original_for_edge_cases(
        self,
    ) -> None:
        cases = [
            {
                "cols": DiffEmptyColumns([10, 60, 110, 160]),
                "header_support": [0, 0, 0, 0],
                "image_width": 300,
                "row_spacing": 20,
            },
            {
                "cols": [10, 180, 220, 260],
                "header_support": [2, 0, 0, 0],
                "image_width": 300,
                "row_spacing": 20,
            },
            {
                "cols": [10, 60, 110, 240],
                "header_support": [0, 0, 0, 0],
                "image_width": 300,
                "row_spacing": 20,
            },
        ]

        for case in cases:
            with self.subTest(cols=list(case["cols"])):
                self.assertEqual(
                    line_refinement._prune_columns_inside_wide_spans(
                        cols=case["cols"],
                        crossing_support=[0 for _ in case["cols"]],
                        segment_support=[0 for _ in case["cols"]],
                        endpoint_support=[0 for _ in case["cols"]],
                        header_support=case["header_support"],
                        image_width=case["image_width"],
                        row_spacing=case["row_spacing"],
                    ),
                    case["cols"],
                )

    def test_prune_columns_inside_wide_spans_keeps_structural_inner_axis(
        self,
    ) -> None:
        pruned = line_refinement._prune_columns_inside_wide_spans(
            cols=[10, 50, 90, 240],
            crossing_support=[0, 0, 0, 0],
            segment_support=[0, 0, 0, 0],
            endpoint_support=[0, 0, 0, 0],
            header_support=LateStructuralHeaderSupport([2, 0, 0, 0], late_index=2),
            image_width=300,
            row_spacing=20,
        )

        self.assertEqual(pruned, [10, 90, 240])

    def test_horizontal_endpoint_candidates_recover_repeated_column_edges(self) -> None:
        horizontal = np.zeros((24, 30), dtype=np.uint8)
        rows = [4, 8, 12, 16]
        for y in rows:
            cv2.line(horizontal, (3, y), (10, y), color=255, thickness=1)
            cv2.line(horizontal, (14, y), (24, y), color=255, thickness=1)

        candidates, support = line_refinement._horizontal_endpoint_candidates(
            horizontal,
            rows,
            radius=1,
            min_run_length=4,
            tolerance=1,
        )

        self.assertEqual(candidates, [3, 10, 14, 24])
        self.assertEqual(support, [4, 4, 4, 4])

    def test_refinement_recovers_missing_verticals_from_horizontal_endpoints(
        self,
    ) -> None:
        horizontal = np.zeros((30, 30), dtype=np.uint8)
        vertical = np.zeros((30, 30), dtype=np.uint8)

        for y in (4, 8, 12, 16, 20):
            cv2.line(horizontal, (3, y), (10, y), color=255, thickness=1)
            cv2.line(horizontal, (14, y), (24, y), color=255, thickness=1)

        cv2.line(vertical, (3, 4), (3, 20), color=255, thickness=1)
        cv2.line(vertical, (24, 4), (24, 20), color=255, thickness=1)

        result = line_refinement.refine_grid_with_projection_profiles(
            horizontal, vertical, horizontal.shape
        )

        self.assertEqual(result.grid.col_coords, [3, 10, 14, 24])
        self.assertEqual(result.col_endpoint_support, [5, 5, 5, 5])

    def test_refinement_recovers_outer_verticals_from_horizontal_extents(
        self,
    ) -> None:
        horizontal = np.zeros((30, 30), dtype=np.uint8)
        vertical = np.zeros((30, 30), dtype=np.uint8)

        for y in (4, 8, 12, 16, 20):
            cv2.line(horizontal, (3, y), (24, y), color=255, thickness=1)

        result = line_refinement.refine_grid_with_projection_profiles(
            horizontal, vertical, horizontal.shape
        )

        self.assertEqual(result.grid.col_coords, [3, 24])
        self.assertEqual(result.col_endpoint_support, [5, 5])

    def test_refinement_keeps_weak_columns_for_recall_first_pass(
        self,
    ) -> None:
        horizontal = np.zeros((30, 28), dtype=np.uint8)
        vertical = np.zeros((30, 28), dtype=np.uint8)

        for y in (2, 8, 14, 20, 26):
            cv2.line(horizontal, (2, y), (24, y), color=255, thickness=1)

        cv2.line(vertical, (4, 2), (4, 26), color=255, thickness=1)
        cv2.line(vertical, (20, 2), (20, 26), color=255, thickness=1)
        for y in (2, 8, 14, 20, 26):
            cv2.line(vertical, (12, y - 1), (12, y + 1), color=255, thickness=1)

        result = line_refinement.refine_grid_with_projection_profiles(
            horizontal, vertical, horizontal.shape
        )

        self.assertEqual(result.grid.col_coords, [4, 12, 20, 24])
        self.assertIn(12, result.grid.col_coords)
        self.assertEqual(
            result.col_segment_support[result.col_candidates.index(12)],
            0,
        )

    def test_refine_grid_with_projection_profiles_raises_for_non_2d_masks(self) -> None:
        horizontal = np.zeros((10, 10, 3), dtype=np.uint8)
        vertical = np.zeros((10, 10), dtype=np.uint8)

        with self.assertRaisesRegex(
            ValueError,
            "refine_grid_with_projection_profiles expects 2D masks",
        ):
            line_refinement.refine_grid_with_projection_profiles(
                horizontal,
                vertical,
                horizontal.shape,
            )

    def test_refine_grid_with_projection_profiles_merges_intersection_centroid_columns(
        self,
    ) -> None:
        horizontal = np.zeros((24, 24), dtype=np.uint8)
        vertical = np.zeros((24, 24), dtype=np.uint8)

        for y in (4, 8, 12):
            cv2.line(horizontal, (2, y), (18, y), color=255, thickness=1)

        cv2.line(vertical, (4, 4), (4, 12), color=255, thickness=1)
        cv2.line(vertical, (12, 4), (12, 12), color=255, thickness=1)

        result = line_refinement.refine_grid_with_projection_profiles(
            horizontal,
            vertical,
            horizontal.shape,
            intersection_centroids=[(13, 8)],
        )

        self.assertEqual(result.col_candidates, [2, 4, 12, 18])


if __name__ == "__main__":
    unittest.main()
