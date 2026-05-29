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


def _filter_intersection_components(
	intersection_mask: np.ndarray,
	*,
	min_area: int,
	max_area: int,
	max_aspect_ratio: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
		intersection_mask,
		connectivity=8,
	)
	filtered_mask = np.zeros_like(intersection_mask)
	centroid_list: list[tuple[int, int]] = []

	for label in range(1, num_labels):
		area = int(stats[label, cv2.CC_STAT_AREA])
		if area < min_area or area > max_area:
			continue

		w = int(stats[label, cv2.CC_STAT_WIDTH])
		h = int(stats[label, cv2.CC_STAT_HEIGHT])
		if w <= 0 or h <= 0:
			continue

		aspect = max(w, h) / max(1, min(w, h))
		if aspect > max_aspect_ratio:
			continue

		filtered_mask[labels == label] = 255
		x = int(centroids[label][0])
		y = int(centroids[label][1])
		centroid_list.append((x, y))

	return filtered_mask, centroid_list


def _intersection_thresholds(shape: tuple[int, int]) -> tuple[int, int, float]:
	h, w = shape[:2]
	image_area = h * w
	# Keep tiny intersections for small synthetic images, stricter for real pages.
	if image_area < 10_000:
		return 1, max(4, int(0.02 * image_area)), 4.0
	return 2, max(25, int(0.005 * image_area)), 3.0

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
		min_area, max_area, max_aspect_ratio = _intersection_thresholds(intersection_mask.shape)
		intersection_mask, centroid_list = _filter_intersection_components(
			intersection_mask,
			min_area=min_area,
			max_area=max_area,
			max_aspect_ratio=max_aspect_ratio,
		)

		metadata = dict(doc.metadata)
		metadata["intersection_mask"] = intersection_mask
		metadata["intersection_centroids"] = centroid_list

		return DocumentImage(doc.image, metadata)

def detect_intersections(horizontal_mask: np.ndarray, vertical_mask: np.ndarray) -> IntersectionResult:
	intersection_mask = cv2.bitwise_and(horizontal_mask, vertical_mask)
	min_area, max_area, max_aspect_ratio = _intersection_thresholds(intersection_mask.shape)
	intersection_mask, centroid_list = _filter_intersection_components(
		intersection_mask,
		min_area=min_area,
		max_area=max_area,
		max_aspect_ratio=max_aspect_ratio,
	)
	return IntersectionResult(intersection_mask=intersection_mask, centroids=centroid_list)
