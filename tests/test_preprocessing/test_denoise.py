from __future__ import annotations

import unittest

import numpy as np

from src.preprocessing.denoise import MorphologicalDenoiseStage
from src.core.image import DocumentImage


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


if __name__ == "__main__":
    unittest.main()
