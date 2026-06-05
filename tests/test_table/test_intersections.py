from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.core.image import DocumentImage
from src.table.intersections import (
    IntersectionDetectionStage,
    _filter_intersection_components,
    _intersection_thresholds,
    detect_intersections,
)


class TestIntersections(unittest.TestCase):
    def test_stage_requires_masks_in_metadata(self) -> None:
        stage = IntersectionDetectionStage()
        doc = DocumentImage(image=np.zeros((10, 10), dtype=np.uint8), metadata={})

        with self.assertRaises(ValueError):
            stage.process(doc)

    def test_stage_detects_centroids_and_stores_mask(self) -> None:
        horizontal = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 255, 255, 255, 0],
                [0, 0, 0, 0, 0],
                [0, 255, 255, 255, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        vertical = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 255, 0, 0],
                [0, 0, 255, 0, 0],
                [0, 0, 255, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        doc = DocumentImage(
            image=np.zeros((5, 5), dtype=np.uint8),
            metadata={
                "horizontal_mask": horizontal,
                "vertical_mask": vertical,
            },
        )

        stage = IntersectionDetectionStage()
        result = stage.process(doc)

        self.assertIn("intersection_mask", result.metadata)
        self.assertIn("intersection_centroids", result.metadata)
        self.assertEqual(result.metadata["intersection_centroids"], [(2, 1), (2, 3)])
        self.assertEqual(int(np.count_nonzero(result.metadata["intersection_mask"])), 2)

    def test_detect_intersections_returns_result_dataclass(self) -> None:
        horizontal = np.array([[0, 255], [0, 255]], dtype=np.uint8)
        vertical = np.array([[0, 0], [255, 255]], dtype=np.uint8)

        result = detect_intersections(horizontal, vertical)

        self.assertEqual(result.intersection_mask.shape, horizontal.shape)
        self.assertEqual(result.centroids, [(1, 1)])

    def test_intersection_thresholds_use_large_image_branch(self) -> None:
        self.assertEqual(_intersection_thresholds((200, 200)), (2, 200, 3.0))

    def test_filter_intersection_components_rejects_bad_components(self) -> None:
        intersection_mask = np.zeros((6, 6), dtype=np.uint8)
        labels = np.array(
            [
                [0, 1, 1, 2, 2, 3],
                [0, 1, 1, 2, 2, 3],
                [0, 4, 4, 4, 4, 5],
                [0, 4, 4, 4, 4, 5],
                [0, 0, 0, 0, 0, 5],
                [0, 0, 0, 0, 0, 5],
            ],
            dtype=np.int32,
        )
        stats = np.array(
            [
                [0, 0, 6, 6, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 50],
                [0, 0, 0, 2, 5],
                [0, 0, 4, 4, 4],
                [0, 0, 4, 1, 4],
            ],
            dtype=np.int32,
        )
        centroids = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [3.0, 1.0],
                [5.0, 1.0],
                [2.0, 3.0],
                [4.0, 4.0],
            ],
            dtype=np.float64,
        )

        with patch(
            "src.table.intersections.cv2.connectedComponentsWithStats",
            return_value=(6, labels, stats, centroids),
        ):
            filtered_mask, centroid_list = _filter_intersection_components(
                intersection_mask,
                min_area=2,
                max_area=10,
                max_aspect_ratio=3.0,
            )

        self.assertGreater(int(np.count_nonzero(filtered_mask)), 0)
        self.assertEqual(centroid_list, [(2, 3)])


if __name__ == "__main__":
    unittest.main()
