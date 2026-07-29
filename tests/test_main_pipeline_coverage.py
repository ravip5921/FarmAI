from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

import main
from src.core.image import DocumentImage
from src.ocr import OcrTable
from src.templates import FormTemplate, TemplateColumn


def _args(**overrides):
    values = {
        "save_line_detection": False,
        "save_intersections": False,
        "save_csv": False,
        "save_json": False,
        "save_cells": False,
        "save_image": False,
        "save_overlay": False,
        "save_all": False,
        "no_print_csv": True,
        "ocr_padding": 2,
        "ocr_context_padding": 0,
        "perspective_correct": False,
        "perspective_padding": 24,
        "llm_verify": "off",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestMainPipelineCoverage(unittest.TestCase):
    def test_run_page_saves_perspective_debug_images_when_save_all_is_enabled(
        self,
    ) -> None:
        page = DocumentImage(np.full((2, 2), 6, dtype=np.uint8))
        processed = DocumentImage(np.ones((2, 2), dtype=np.uint8))
        first_table_result = Mock()
        first_table_result.grid = "first-grid"
        first_table_result.line_detection.horizontal_mask = np.zeros(
            (2, 2), dtype=np.uint8
        )
        first_table_result.line_detection.vertical_mask = np.zeros(
            (2, 2), dtype=np.uint8
        )
        second_table_result = Mock()
        second_table_result.grid = "second-grid"
        ocr_result = Mock(csv_path=None, json_path=None)
        ocr_result.table = OcrTable(cells=[], row_count=0, col_count=0)
        correction = Mock(
            corrected=True,
            image=np.full((3, 4), 255, dtype=np.uint8),
            corners=np.array([[0, 0], [3, 0], [3, 2], [0, 2]], dtype=np.float32),
            padded_corners=np.array([[0, 0], [3, 0], [3, 2], [0, 2]], dtype=np.float32),
            output_size=(4, 3),
        )

        with TemporaryDirectory() as tmpdir:
            with (
                patch("main.process_image", return_value=processed),
                patch(
                    "main.process_table",
                    side_effect=[first_table_result, second_table_result],
                ),
                patch("main.correct_table_perspective", return_value=correction),
                patch(
                    "main.warp_image_to_corners",
                    return_value=Mock(image=np.full((3, 4), 8, dtype=np.uint8)),
                ),
                patch("main.render_perspective_corners", return_value=np.zeros((2, 2))),
                patch("main.process_ocr", return_value=ocr_result),
                patch("main.render_grid_structure", return_value=np.zeros((3, 4))),
                patch("main.render_grid_overlay", return_value=np.zeros((3, 4, 3))),
                patch("main.save_debug") as save_debug,
                patch("main.show"),
            ):
                main._run_page(
                    page,
                    pipeline=Mock(),
                    debug_dir=Path(tmpdir),
                    image_name="record",
                    args=_args(perspective_correct=True, save_all=True),
                    ocr_engine="engine",
                )

        saved_names = [call.args[0].name for call in save_debug.call_args_list]
        self.assertIn("perspective_corners_record.png", saved_names)
        self.assertIn("perspective_corrected_record.png", saved_names)

    def test_process_ocr_uses_template_filters_headers_and_rules(self) -> None:
        template = FormTemplate(
            id="room",
            name="Room",
            description="",
            column_widths=(1.0, 1.0),
            uniform_row_height=False,
            columns=(
                TemplateColumn(index=0, key="pen", name="Pen", filter_out=True),
                TemplateColumn(
                    index=1,
                    key="hi",
                    name="HI",
                    value_type="temperature",
                    range_min=80,
                    range_max=120,
                ),
            ),
        )
        table_result = Mock()
        table_result.grid = "grid"

        with patch("main.export_table_ocr", return_value="ocr") as export:
            result = main.process_ocr(
                np.zeros((1, 1), dtype=np.uint8),
                table_result,
                template=template,
            )

        self.assertEqual(result, "ocr")
        self.assertEqual(export.call_args.kwargs["filter_out_columns"], set())
        self.assertEqual(export.call_args.kwargs["filter_out_column_indices"], {0})
        self.assertEqual(export.call_args.kwargs["column_names"], ["Pen", "HI"])
        self.assertEqual(export.call_args.kwargs["column_keys"], ["pen", "hi"])
        self.assertEqual(len(export.call_args.kwargs["column_ocr_rules"]), 2)
        self.assertIsNone(export.call_args.kwargs["llm_verifier"])

    def test_process_ocr_creates_llm_verifier_when_requested(self) -> None:
        table_result = Mock()
        table_result.grid = "grid"

        with patch("main.export_table_ocr", return_value="ocr") as export:
            result = main.process_ocr(
                np.zeros((1, 1), dtype=np.uint8),
                table_result,
                llm_verify_mode="invalid",
            )

        self.assertEqual(result, "ocr")
        self.assertEqual(export.call_args.kwargs["llm_verifier"].mode, "invalid")


if __name__ == "__main__":
    unittest.main()
