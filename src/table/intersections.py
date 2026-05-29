from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.core.image import DocumentImage
from src.core.stage import PipelineStage

@dataclass
class IntersectionResult:
	intersection_mask: np.ndarray
	centroids: list[tuple[int, int]]

class IntersectionDetectionStage(PipelineStage):
	"""Detect intersections by combining horizontal and vertical masks."""

	def _extract_masks(self, doc: DocumentImage) -> tuple[np.ndarray, np.ndarray]:
		horizontal = doc.metadata.get("horizontal_mask")
		vertical = doc.metadata.get("vertical_mask")
		if horizontal is None or vertical is None:
			raise ValueError("IntersectionDetectionStage requires horizontal_mask and vertical_mask in metadata")
		return horizontal, vertical

	def process(self, doc: DocumentImage) -> DocumentImage:
		horizontal, vertical = self._extract_masks(doc)
		intersection_mask = cv2.bitwise_and(horizontal, vertical)

		num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersection_mask, connectivity=8)
		centroid_list: list[tuple[int, int]] = []
		for label in range(1, num_labels):
			x = int(centroids[label][0])
			y = int(centroids[label][1])
			centroid_list.append((x, y))

		metadata = dict(doc.metadata)
		metadata["intersection_mask"] = intersection_mask
		metadata["intersection_centroids"] = centroid_list

		return DocumentImage(doc.image, metadata)

def detect_intersections(horizontal_mask: np.ndarray, vertical_mask: np.ndarray) -> IntersectionResult:
	intersection_mask = cv2.bitwise_and(horizontal_mask, vertical_mask)
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersection_mask, connectivity=8)
	centroid_list: list[tuple[int, int]] = []
	for label in range(1, num_labels):
		x = int(centroids[label][0])
		y = int(centroids[label][1])
		centroid_list.append((x, y))
	return IntersectionResult(intersection_mask=intersection_mask, centroids=centroid_list)
