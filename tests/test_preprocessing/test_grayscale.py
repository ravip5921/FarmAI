from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.preprocessing.grayscale import GrayscaleStage

from . import make_tiny_color_image, make_tiny_gray_image


class TestGrayscaleStage(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = GrayscaleStage()

    def test_process_converts_color_image_to_gray(self) -> None:
        doc = make_tiny_color_image()

        result = self.stage.process(doc)

        self.assertEqual(result.image.shape, (3, 3))
        self.assertEqual(result.image.dtype, np.uint8)
        self.assertTrue(result.metadata.get("grayscale"))

        expected = cv2.cvtColor(doc.image, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(result.image, expected)

    def test_process_copies_grayscale_image_unchanged(self) -> None:
        doc = make_tiny_gray_image()

        result = self.stage.process(doc)

        self.assertEqual(result.image.shape, (3, 3))
        self.assertEqual(result.image.dtype, np.uint8)
        self.assertTrue(result.metadata.get("grayscale"))
        np.testing.assert_array_equal(result.image, doc.image)


if __name__ == "__main__":
    unittest.main()
