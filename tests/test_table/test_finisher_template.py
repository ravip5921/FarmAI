from pathlib import Path
import unittest

import numpy as np

from src.application.processing import process_document
from src.application.result_models import ProcessingSettings
from src.ocr.base import OcrText
from src.templates import load_template
from src.core.image import DocumentImage
from src.preprocessing.template_image import prepare_template_image


class TestFinisherTemplate(unittest.TestCase):
    def test_large_photo_keeps_all_rows_and_hides_manager_to_last_row(self) -> None:
        source = (
            Path(__file__).resolve().parents[2]
            / "examples/Finisher 3 - barn - flash.jpg"
        )
        if not source.exists():
            self.skipTest("Bundled finisher photograph is unavailable")

        class Engine:
            def recognize(self, image: np.ndarray) -> OcrText:
                return OcrText(text="70")

        page = process_document(
            source,
            settings=ProcessingSettings(template_id="finisher_room_3"),
            engine=Engine(),
        ).pages[0]
        self.assertEqual(page.image_width, 1200)
        self.assertEqual(page.data_row_count, 32)
        self.assertEqual(len(page.cells), 32 * 9)
        for x, y in ((1100, 220), (1100, 1100), (1100, 1250), (1000, 1250)):
            self.assertTrue(np.all(page.source_image[y, x] == 0))

    def test_preprocessing_remains_unchanged_for_other_templates(self) -> None:
        from src.templates import get_template_ids

        page = DocumentImage(np.full((800, 1600, 3), 255, np.uint8))
        for template_id in get_template_ids():
            if template_id == "finisher_room_3":
                continue
            template = load_template(template_id)
            self.assertEqual(template.binarization_k, 0.34)
            self.assertIs(prepare_template_image(page, template), page)
            self.assertFalse(template.redaction_follow_curved_borders)

    def test_sample_masks_initials_and_preserves_comments_and_rows(self) -> None:
        source = Path(__file__).resolve().parents[2] / "examples/Finisher room 3.pdf"
        if not source.exists():
            self.skipTest("Bundled finisher PDF is unavailable")

        class Engine:
            def recognize(self, image: np.ndarray) -> OcrText:
                return OcrText(text="70")

        template = load_template("finisher_room_3")
        self.assertEqual(template.min_horizontal_line_length_ratio, 0.045)
        self.assertTrue(template.redaction_follow_column_borders)
        page = process_document(
            source,
            settings=ProcessingSettings(template_id=template.id),
            engine=Engine(),
        ).pages[0]
        self.assertEqual(page.data_row_count, 32)
        self.assertEqual(len(page.cells), 32 * 9)
        self.assertNotIn("initials", [column.key for column in page.columns])
        self.assertNotIn("manager_initials", [column.key for column in page.columns])
        for x in (1060, 1110, 1150, 1174):
            self.assertTrue(np.all(page.source_image[250, x] == 0))
        # The old proportional mask began at x=1019, inside the comments.
        self.assertTrue(np.all(page.source_image[250, 1020] > 200))
