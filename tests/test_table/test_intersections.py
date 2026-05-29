from __future__ import annotations

import unittest

import numpy as np

from src.core.image import DocumentImage
from src.table.intersections import IntersectionDetectionStage, detect_intersections


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
		doc = DocumentImage(image=np.zeros((5, 5), dtype=np.uint8), metadata={
			"horizontal_mask": horizontal,
			"vertical_mask": vertical,
		})

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


if __name__ == "__main__":
	unittest.main()
