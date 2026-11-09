from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from src.application.processing import process_document
from src.application.result_models import ProcessingSettings
from src.ocr.base import OcrText
from src.table import process_table_image
from src.templates import get_template_ids, load_template


def _nursery_form(scale: float) -> tuple[np.ndarray, list[int], list[int]]:
    image = np.full((round(1684 * scale), round(1190 * scale)), 255, np.uint8)
    columns = [
        round(x * scale)
        for x in (60, 120, 183, 243, 303, 315, 375, 462, 474, 990, 1080, 1092, 1152)
    ]
    rows = [round((240 + 38 * i) * scale) for i in range(34)]
    thickness = max(1, round(scale))
    # The manager field is a detached 60-pixel-wide table, separated by a gap.
    for first, last in ((0, 4), (5, 7), (8, 10), (11, 12)):
        for y in rows:
            cv2.line(image, (columns[first], y), (columns[last], y), 0, thickness)
        for x in columns[first : last + 1]:
            cv2.line(image, (x, rows[0]), (x, rows[-1]), 0, thickness)
    for y in rows[1:]:
        for text, x, size in (("All good", 490, 0.6), ("AB", 1100, 0.5)):
            cv2.putText(
                image,
                text,
                (round(x * scale), y - round(12 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                size * scale,
                0,
                thickness,
            )
    return image, columns, rows


class TestNurseryTemplate(unittest.TestCase):
    def test_recovers_detached_manager_column_at_different_scan_sizes(self) -> None:
        template = load_template("nursery_room_4")
        for scale in (0.75, 1.0, 1.5):
            with self.subTest(scale=scale):
                image, columns, rows = _nursery_form(scale)
                default = process_table_image(image)
                repaired = process_table_image(image, template=template)
                self.assertLess(default.grid.col_coords[-1], columns[-2])
                self.assertAlmostEqual(
                    repaired.grid.col_coords[-1], columns[-1], delta=3 * scale
                )
                self.assertAlmostEqual(
                    repaired.grid.col_coords[-2], columns[-2], delta=3 * scale
                )
                self.assertEqual(len(repaired.grid.row_coords), len(rows))
                self.assertEqual(len(repaired.grid.col_coords), 13)

    def test_detection_override_is_opt_in_for_nursery_and_finisher(self) -> None:
        for template_id in get_template_ids():
            if template_id in {"nursery_room_4", "finisher_room_3"}:
                continue
            with self.subTest(template=template_id):
                template = load_template(template_id)
                self.assertIsNone(template.min_horizontal_line_length_ratio)
                self.assertFalse(template.redaction_follow_column_borders)

        image, columns, _ = _nursery_form(1.0)
        disabled = replace(
            load_template("nursery_room_4"), min_horizontal_line_length_ratio=None
        )
        result = process_table_image(image, template=disabled)
        self.assertLess(result.grid.col_coords[-1], columns[-2])

    def test_sample_keeps_data_rows_and_excludes_both_initials_columns(self) -> None:
        source = Path(__file__).resolve().parents[2] / "examples/Nursery room 4.pdf"
        if not source.exists():
            self.skipTest("Bundled nursery PDF is unavailable")

        class Engine:
            calls = 0

            def recognize(self, image: np.ndarray) -> OcrText:
                self.calls += 1
                return OcrText(text="70")

        engine = Engine()
        page = process_document(
            source,
            settings=ProcessingSettings(template_id="nursery_room_4"),
            engine=engine,
        ).pages[0]
        self.assertEqual(page.data_row_count, 32)
        self.assertEqual(
            [column.key for column in page.columns],
            [
                "date",
                "current_temperature",
                "hi",
                "lo",
                "inventory_change",
                "inventory",
                "comments",
            ],
        )
        self.assertEqual(engine.calls, 32 * 7)
        # These points are inside the employee and manager fields, including
        # the top-right area missed by a rectangle based on average borders.
        for x, y in ((1000, 300), (1070, 300), (1110, 300), (1140, 300), (1100, 1450)):
            self.assertTrue(np.all(page.source_image[y, x] == 0))


if __name__ == "__main__":
    unittest.main()
