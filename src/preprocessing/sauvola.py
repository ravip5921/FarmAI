from __future__ import annotations

import numpy as np
from skimage.filters import threshold_sauvola

from src.core.image import DocumentImage
from src.core.stage import PipelineStage


class SauvolaBinarizationStage(PipelineStage):
	def __init__(self, window_size: int = 25, k: float = 0.34):
		self.window_size = window_size
		self.k = k

	def process(self, doc: DocumentImage) -> DocumentImage:
		image = doc.image

		threshold = threshold_sauvola(image, window_size=self.window_size, k=self.k)
		binary = (image > threshold).astype(np.uint8) * 255

		metadata = dict(doc.metadata)
		metadata["binary"] = True

		return DocumentImage(binary, metadata)
