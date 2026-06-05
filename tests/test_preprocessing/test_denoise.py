from __future__ import annotations

import unittest

import numpy as np

from src.core.image import DocumentImage
from src.preprocessing.denoise import MorphologicalDenoiseStage


class TestDenoiseStage(unittest.TestCase):
    def test_closes_small_hole(self) -> None:
        # 5x5 white image with a single black pixel in the center
        img = np.full((5, 5), 255, dtype=np.uint8)
        img[2, 2] = 0

        doc = DocumentImage(image=img, metadata={})

        stage = MorphologicalDenoiseStage(kernel_size=3)
        result = stage.process(doc)

        self.assertEqual(result.image.shape, (5, 5))
        self.assertTrue(result.metadata.get("denoised"))
        # center pixel should be closed (become white)
        self.assertEqual(int(result.image[2, 2]), 255)

    def test_preserves_thin_connected_lines(self) -> None:
        img = np.full((8, 8), 255, dtype=np.uint8)
        img[4, 1:7] = 0

        doc = DocumentImage(image=img, metadata={})
        stage = MorphologicalDenoiseStage(kernel_size=3)
        result = stage.process(doc)

        self.assertTrue(
            np.array_equal(result.image[4, 1:7], np.zeros(6, dtype=np.uint8))
        )
        self.assertEqual(result.metadata.get("denoise_method"), "connected_components")

    def test_non_binary_image(self) -> None:
        img = np.full((5, 5, 3), 128, dtype=np.uint8)

        doc = DocumentImage(image=img, metadata={})

        stage = MorphologicalDenoiseStage(kernel_size=3)
        result = stage.process(doc)

        self.assertTrue(result.metadata.get("denoised"))
        self.assertEqual(result.metadata.get("denoise_method"), "median_blur")
        self.assertEqual(result.image.shape, (5, 5, 3))

    def test_invalid_kernal_size(self) -> None:
        img = np.full((8, 8), 255, dtype=np.uint8)
        img[4, 1:7] = 0

        doc = DocumentImage(image=img, metadata={})
        stage = MorphologicalDenoiseStage(kernel_size=0)
        result = stage.process(doc)

        self.assertTrue(
            np.array_equal(result.image[4, 1:7], np.zeros(6, dtype=np.uint8))
        )
        self.assertEqual(result.metadata.get("denoise_method"), "none")


if __name__ == "__main__":
    unittest.main()
