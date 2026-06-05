from __future__ import annotations

import unittest

import numpy as np

from src.core.image import DocumentImage
from src.preprocessing.sauvola import SauvolaBinarizationStage


class TestSauvolaStage(unittest.TestCase):
    def setUp(self) -> None:
        # use a small window so tests run quickly on small fixtures
        self.stage = SauvolaBinarizationStage(window_size=3, k=0.2)

    def test_binarizes_to_0_255(self) -> None:
        # create a simple horizontal gradient 10x10
        img = np.tile(np.linspace(0, 255, 10, dtype=np.uint8), (10, 1))
        doc = DocumentImage(image=img, metadata={})

        result = self.stage.process(doc)

        unique = np.unique(result.image)
        # result should only contain 0 and/or 255 values
        self.assertTrue(set(unique.tolist()).issubset({0, 255}))
        self.assertTrue(result.metadata.get("binary"))


if __name__ == "__main__":
    unittest.main()
