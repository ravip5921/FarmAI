from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.analysis.character_size import estimate_character_size
from src.core.image import DocumentImage
from src.core.stage import PipelineStage

@dataclass
class LineDetectionResult:
	horizontal_mask: np.ndarray
	vertical_mask: np.ndarray

class LineDetectionStage(PipelineStage):
	"""Extract horizontal and vertical line masks from a binary image."""

	def __init__(self, min_line_scale: float = 1.55):
		self.min_line_scale = min_line_scale

	def _ensure_binary(self, image: np.ndarray) -> np.ndarray:
		if image.dtype != np.uint8:
			image = (image > 0).astype(np.uint8) * 255
		if image.ndim == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# Use adaptive polarity: keep the sparser foreground mask as line candidates.
		_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
		_, binary_inv = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
		fg_ratio = float(np.count_nonzero(binary)) / float(binary.size)
		fg_inv_ratio = float(np.count_nonzero(binary_inv)) / float(binary_inv.size)

		# Table lines are typically sparse relative to the page area.
		if fg_inv_ratio < fg_ratio:
			return binary_inv
		return binary

	def _filter_line_components(
		self,
		mask: np.ndarray,
		*,
		orientation: str,
		min_length: int,
		max_thickness: int,
	) -> np.ndarray:
		num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
		filtered = np.zeros_like(mask)

		for label in range(1, num_labels):
			w = int(stats[label, cv2.CC_STAT_WIDTH])
			h = int(stats[label, cv2.CC_STAT_HEIGHT])
			if orientation == "horizontal":
				length = w
				thickness = h
			else:
				length = h
				thickness = w

			if length >= min_length and thickness <= max_thickness:
				filtered[labels == label] = 255

		return filtered

	def _kernel_size(self, binary: np.ndarray) -> int:
		report = estimate_character_size(binary, min_area=10)
		char_height = max(1, report.average_height or report.median_height or 10)
		return max(3, int(char_height * self.min_line_scale))

	def _hough_vertical_mask(self, binary: np.ndarray, min_line_length: int) -> np.ndarray:
		"""Recover weak vertical rules using a near-vertical Hough detector."""
		edges = cv2.Canny(binary, 50, 150)
		lines = cv2.HoughLinesP(
			edges,
			1,
			np.pi / 180,
			threshold=45,
			minLineLength=max(10, min_line_length),
			maxLineGap=10,
		)
		mask = np.zeros_like(binary)
		if lines is None:
			return mask

		for line in lines[:, 0]:
			x1, y1, x2, y2 = line
			dx = abs(x2 - x1)
			dy = abs(y2 - y1)
			# Keep near-vertical segments only.
			if dy >= max(8, dx * 2):
				cv2.line(mask, (x1, y1), (x2, y2), color=255, thickness=1)

		# Bridge tiny breaks in hand-drawn strokes.
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		return mask

	def _retain_vertical_components_with_crossings(
		self,
		vertical_mask: np.ndarray,
		horizontal_mask: np.ndarray,
		*,
		min_crossings: int,
	) -> np.ndarray:
		"""Keep vertical components that intersect horizontal rules enough times."""
		num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vertical_mask, connectivity=8)
		if num_labels <= 1:
			return vertical_mask

		horizontal_hits = cv2.dilate(
			horizontal_mask,
			cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
		)
		filtered = np.zeros_like(vertical_mask)

		for label in range(1, num_labels):
			component = labels == label
			crossings = int(np.count_nonzero(np.any(component & (horizontal_hits > 0), axis=1)))
			if crossings >= min_crossings:
				filtered[component] = 255

		return filtered

	def process(self, doc: DocumentImage) -> DocumentImage:
		binary = self._ensure_binary(doc.image)
		report = estimate_character_size(binary, min_area=10)
		char_height = max(1, report.average_height or report.median_height or 10)
		size = max(3, int(char_height * self.min_line_scale))

		horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
		vertical_size = max(3, int(size * 0.6))
		vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

		horizontal_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
		vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

		# Keep only long/thin structures to suppress text blobs.
		# Use character-size-driven thresholds so small/partial tables in large
		# images do not lose all vertical rules.
		# Cap page-relative minima by character-relative maxima to avoid
		# over-pruning true rules on large pages while still rejecting text.
		min_h_length = max(18, min(int(binary.shape[1] * 0.12), int(char_height * 10.0)))
		min_v_length = max(14, min(int(binary.shape[0] * 0.1), int(char_height * 6.0)))
		max_thickness = max(2, int(char_height * 0.75))
		horizontal_mask = self._filter_line_components(
			horizontal_mask,
			orientation="horizontal",
			min_length=min_h_length,
			max_thickness=max_thickness,
		)
		vertical_mask = self._filter_line_components(
			vertical_mask,
			orientation="vertical",
			min_length=min_v_length,
			max_thickness=max_thickness,
		)

		# Fallback for weak/fragmented hand-drawn vertical rules.
		if int(np.count_nonzero(vertical_mask)) == 0:
			fallback_v_size = max(3, int(vertical_size * 0.8))
			fallback_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, fallback_v_size))
			fallback_min_v_length = max(14, int(min_v_length * 0.55))
			fallback_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, fallback_kernel)
			vertical_mask = self._filter_line_components(
				fallback_v,
				orientation="vertical",
				min_length=fallback_min_v_length,
				max_thickness=max_thickness,
			)

		# Additional recovery: merge with Hough near-vertical segments only when
		# morphology result is sparse, then re-filter.
		vertical_count = int(np.count_nonzero(vertical_mask))
		horizontal_count = int(np.count_nonzero(horizontal_mask))
		need_vertical_recovery = (
			vertical_count == 0
			or vertical_count < max(10, int(0.02 * max(1, horizontal_count)))
		)
		if need_vertical_recovery:
			hough_v = self._hough_vertical_mask(binary, min_line_length=max(10, int(min_v_length * 0.7)))
			if int(np.count_nonzero(hough_v)) > 0:
				vertical_mask = cv2.bitwise_or(vertical_mask, hough_v)
				vertical_mask = self._filter_line_components(
					vertical_mask,
					orientation="vertical",
					min_length=max(10, int(min_v_length * 0.7)),
					max_thickness=max_thickness,
				)

		# Final gate: real table vertical rules should cross horizontal rules.
		crossing_threshold = 1 if binary.shape[0] < 200 else 2
		vertical_mask = self._retain_vertical_components_with_crossings(
			vertical_mask,
			horizontal_mask,
			min_crossings=crossing_threshold,
		)

		metadata = dict(doc.metadata)
		metadata["line_detection"] = {
			"kernel_size": size,
			"vertical_kernel_size": vertical_size,
			"estimated_char_height": report.average_height,
			"min_h_length": min_h_length,
			"min_v_length": min_v_length,
			"max_thickness": max_thickness,
			"vertical_crossings_min": crossing_threshold,
		}

		result = DocumentImage(binary, metadata)
		result.metadata["horizontal_mask"] = horizontal_mask
		result.metadata["vertical_mask"] = vertical_mask
		return result

def detect_lines(binary: np.ndarray, min_line_scale: float = 1.7) -> LineDetectionResult:
	"""Convenience function for line extraction without pipeline wiring."""
	doc = DocumentImage(binary)
	stage = LineDetectionStage(min_line_scale=min_line_scale)
	result = stage.process(doc)
	return LineDetectionResult(
		horizontal_mask=result.metadata["horizontal_mask"],
		vertical_mask=result.metadata["vertical_mask"],
	)
