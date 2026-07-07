from __future__ import annotations

import unittest

import numpy as np

from src.ocr.image_preprocessing import (
    CellImagePreprocessConfig,
    prepare_cell_image_for_ocr,
)


class TestImagePreprocessingCoverage(unittest.TestCase):
    def test_resize_skips_when_scale_is_already_close_to_one(self) -> None:
        image = np.full((100, 20), 255, dtype=np.uint8)

        prepared = prepare_cell_image_for_ocr(
            image,
            CellImagePreprocessConfig(
                crop_to_ink=False,
                border=0,
                target_height=102,
            ),
        )

        self.assertEqual(prepared.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
