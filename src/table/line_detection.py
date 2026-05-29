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

	def __init__(self, min_line_scale: float = 2.0):
		self.min_line_scale = min_line_scale

	def _ensure_binary(self, image: np.ndarray) -> np.ndarray:
		if image.dtype != np.uint8:
			image = (image > 0).astype(np.uint8) * 255
		if image.ndim == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
		return binary

	def _kernel_size(self, binary: np.ndarray) -> int:
		report = estimate_character_size(binary, min_area=10)
		char_height = max(1, report.average_height or report.median_height or 10)
		return max(3, int(char_height * self.min_line_scale))

	def process(self, doc: DocumentImage) -> DocumentImage:
		binary = self._ensure_binary(doc.image)
		size = self._kernel_size(binary)

		horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
		vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))

		horizontal_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
		vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

		metadata = dict(doc.metadata)
		metadata["line_detection"] = {
			"kernel_size": size,
			"estimated_char_height": estimate_character_size(binary).average_height,
		}

		result = DocumentImage(binary, metadata)
		result.metadata["horizontal_mask"] = horizontal_mask
		result.metadata["vertical_mask"] = vertical_mask
		return result

def detect_lines(binary: np.ndarray, min_line_scale: float = 2.0) -> LineDetectionResult:
	"""Convenience function for line extraction without pipeline wiring."""
	doc = DocumentImage(binary)
	stage = LineDetectionStage(min_line_scale=min_line_scale)
	result = stage.process(doc)
	return LineDetectionResult(
		horizontal_mask=result.metadata["horizontal_mask"],
		vertical_mask=result.metadata["vertical_mask"],
	)
