from __future__ import annotations

import numpy as np

from src.core.image import DocumentImage


def make_tiny_color_image() -> DocumentImage:
	"""Create a small 3x3 RGB image fixture for preprocessing tests."""
	image = np.array(
		[
			[[10, 20, 30], [40, 50, 60], [70, 80, 90]],
			[[15, 25, 35], [45, 55, 65], [75, 85, 95]],
			[[20, 30, 40], [50, 60, 70], [80, 90, 100]],
		],
		dtype=np.uint8,
	)
	return DocumentImage(image=image, metadata={"fixture": "tiny_color_3x3"})


def make_tiny_gray_image() -> DocumentImage:
	"""Create a small 3x3 grayscale image fixture for preprocessing tests."""
	image = np.array(
		[
			[0, 64, 128],
			[32, 96, 160],
			[48, 112, 192],
		],
		dtype=np.uint8,
	)
	return DocumentImage(image=image, metadata={"fixture": "tiny_gray_3x3"})
