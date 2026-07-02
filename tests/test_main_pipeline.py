from __future__ import annotations

import argparse
import io
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

import main
from src.core.image import DocumentImage
from src.core.io import LoadedDocument
from src.ocr import OcrCell, OcrTable
from src.table import GridCell, GridStructure


class _Stage:
    def __init__(self, name: str) -> None:
        self.name = name

    def process(self, doc: DocumentImage) -> DocumentImage:
        metadata = dict(doc.metadata)
        metadata[self.name] = True
        return DocumentImage(doc.image + 1, metadata)


class _Pipeline:
    def __init__(self) -> None:
        self.stages = [_Stage("a"), _Stage("b")]

    def run(self, doc: DocumentImage) -> DocumentImage:
        for stage in self.stages:
            doc = stage.process(doc)
        return doc


def _args(**overrides):
    values = {
        "save_line_detection": True,
        "save_intersections": True,
        "save_csv": True,
        "save_json": True,
        "save_cells": True,
        "save_image": True,
        "save_overlay": True,
        "save_all": False,
        "no_print_csv": False,
        "ocr_padding": 4,
        "ocr_context_padding": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestMainPipeline(unittest.TestCase):
    def test_parse_args_reads_cli_options(self) -> None:
        argv = [
            "main.py",
            "sample.jpg",
            "--no-debug",
            "--save-all",
            "--save-cells",
            "--ocr-padding",
            "9",
            "--ocr-context-padding",
            "11",
            "--ocr-engine",
            "tesseract",
            "--template",
            "boar_room",
        ]

        with patch.object(sys, "argv", argv):
            args = main.parse_args()

        self.assertEqual(args.image_path, Path("sample.jpg"))
        self.assertTrue(args.no_debug)
        self.assertTrue(args.save_all)
        self.assertTrue(args.save_cells)
        self.assertEqual(args.ocr_padding, 9)
        self.assertEqual(args.ocr_context_padding, 11)
        self.assertEqual(args.template, "boar_room")

    def test_build_pipeline_uses_requested_parameters(self) -> None:
        pipeline = main.build_pipeline(window_size=31, k=0.2, denoise_kernel=5)

        self.assertEqual(len(pipeline.stages), 4)
        self.assertEqual(pipeline.stages[1].window_size, 31)
        self.assertEqual(pipeline.stages[1].k, 0.2)
        self.assertEqual(pipeline.stages[2].kernel_size, 5)

    def test_process_image_runs_pipeline_without_debug(self) -> None:
        image = np.ones((2, 2), dtype=np.uint8)

        with patch("main.cv2.imread", return_value=image):
            result = main.process_image("sample.jpg", _Pipeline())

        self.assertEqual(int(result.image[0, 0]), 3)

    def test_process_image_raises_for_unreadable_path(self) -> None:
        with patch("main.cv2.imread", return_value=None):
            with self.assertRaises(FileNotFoundError):
                main.process_image("missing.jpg", _Pipeline())

    def test_process_image_saves_each_debug_stage(self) -> None:
        doc = DocumentImage(np.zeros((2, 2), dtype=np.uint8))

        with TemporaryDirectory() as tmpdir:
            with patch("main.save_debug") as save_debug:
                result = main.process_image(
                    doc,
                    _Pipeline(),
                    debug_dir=Path(tmpdir),
                    image_name="record.png",
                )

        self.assertEqual(int(result.image[0, 0]), 2)
        self.assertEqual(save_debug.call_count, 2)

    def test_process_table_delegates_to_table_pipeline(self) -> None:
        with patch("main.process_table_image", return_value="table") as process:
            result = main.process_table(
                np.zeros((1, 1), dtype=np.uint8),
                image_name="sample",
                debug_dir=Path("debug"),
                save_line_detection=True,
                save_intersections=True,
                template="template",
            )

        self.assertEqual(result, "table")
        process.assert_called_once()
        self.assertEqual(process.call_args.kwargs["template"], "template")

    def test_process_ocr_sets_requested_export_paths(self) -> None:
        table_result = Mock()
        table_result.grid = GridStructure(
            row_coords=[0, 1],
            col_coords=[0, 1],
            cells=[GridCell(row=0, col=0, bbox=(0, 0, 1, 1))],
        )

        with TemporaryDirectory() as tmpdir:
            with patch("main.export_table_ocr", return_value="ocr") as export:
                result = main.process_ocr(
                    np.zeros((1, 1), dtype=np.uint8),
                    table_result,
                    image_name="record.png",
                    debug_dir=Path(tmpdir),
                    save_csv=True,
                    save_json=True,
                    save_cells=True,
                    padding=7,
                    context_padding=8,
                    engine="engine",
                )

        self.assertEqual(result, "ocr")
        self.assertEqual(
            export.call_args.kwargs["csv_path"].name, "table_ocr_record.csv"
        )
        self.assertEqual(
            export.call_args.kwargs["json_path"].name, "table_ocr_record.json"
        )
        self.assertEqual(export.call_args.kwargs["cell_image_dir"].name, "record_cells")
        self.assertEqual(export.call_args.kwargs["padding"], 7)
        self.assertEqual(export.call_args.kwargs["context_padding"], 8)
        self.assertIsNone(export.call_args.kwargs["filter_out_column_indices"])

    def test_process_ocr_without_exports_keeps_paths_empty(self) -> None:
        table_result = Mock()
        table_result.grid = "grid"

        with patch("main.export_table_ocr", return_value="ocr") as export:
            result = main.process_ocr(
                np.zeros((1, 1), dtype=np.uint8),
                table_result,
                debug_dir=None,
            )

        self.assertEqual(result, "ocr")
        self.assertIsNone(export.call_args.kwargs["csv_path"])
        self.assertIsNone(export.call_args.kwargs["json_path"])
        self.assertIsNone(export.call_args.kwargs["cell_image_dir"])

    def test_run_page_saves_outputs_and_prints_csv(self) -> None:
        page = DocumentImage(np.zeros((2, 2), dtype=np.uint8))
        processed = DocumentImage(np.ones((2, 2), dtype=np.uint8))
        table_result = Mock()
        table_result.grid = "grid"
        ocr_table = OcrTable(
            cells=[OcrCell(row=0, col=0, bbox=(0, 0, 1, 1), text="Pen")],
            row_count=1,
            col_count=1,
        )
        ocr_result = Mock(csv_path=Path("out.csv"), json_path=Path("out.json"))
        ocr_result.table = ocr_table

        with TemporaryDirectory() as tmpdir:
            with (
                patch("main.process_image", return_value=processed),
                patch("main.process_table", return_value=table_result),
                patch("main.process_ocr", return_value=ocr_result),
                patch("main.render_grid_structure", return_value=np.zeros((2, 2))),
                patch("main.render_grid_overlay", return_value=np.zeros((2, 2, 3))),
                patch("main.save_debug") as save_debug,
                patch("main.show") as show,
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    main._run_page(
                        page,
                        pipeline=_Pipeline(),
                        debug_dir=Path(tmpdir),
                        image_name="record",
                        args=_args(),
                        ocr_engine="engine",
                    )

        self.assertEqual(save_debug.call_count, 2)
        self.assertEqual(show.call_count, 2)
        self.assertIn("Saved OCR CSV to: out.csv", output.getvalue())
        self.assertIn("--- OCR CSV: record ---", output.getvalue())

    def test_main_processes_loaded_document_pages(self) -> None:
        args = argparse.Namespace(
            image_path=Path("record.pdf"),
            no_debug=True,
            debug_dir=Path("debug"),
            ocr_engine="tesseract",
            trocr_model="model",
            save_cells=False,
            template=None,
        )
        page = DocumentImage(np.zeros((1, 1)), metadata={"page_index": 2})
        loaded = LoadedDocument([page], Path("record.pdf"))

        with (
            patch("main.parse_args", return_value=args),
            patch("main.build_pipeline", return_value="pipeline"),
            patch("main.create_ocr_engine", return_value="engine"),
            patch("main.load_document", return_value=loaded),
            patch("main._run_page") as run_page,
        ):
            main.main()

        run_page.assert_called_once()
        self.assertEqual(run_page.call_args.kwargs["image_name"], "record_p2")
        self.assertIsNone(run_page.call_args.kwargs["debug_dir"])

    def test_main_processes_single_image(self) -> None:
        args = argparse.Namespace(
            image_path=Path("record.jpg"),
            no_debug=False,
            debug_dir=Path("debug"),
            ocr_engine="tesseract",
            trocr_model="model",
            save_cells=False,
            template=None,
        )
        loaded = DocumentImage(np.zeros((1, 1)))

        with (
            patch("main.parse_args", return_value=args),
            patch("main.build_pipeline", return_value="pipeline"),
            patch("main.create_ocr_engine", return_value="engine"),
            patch("main.load_document", return_value=loaded),
            patch("main._run_page") as run_page,
        ):
            main.main()

        run_page.assert_called_once()
        self.assertEqual(run_page.call_args.kwargs["image_name"], "record")
        self.assertEqual(run_page.call_args.kwargs["debug_dir"], Path("debug"))

    def test_module_entrypoint_runs_main(self) -> None:
        with (
            patch.object(sys, "argv", ["main.py", "record.jpg", "--no-debug"]),
            patch("src.ocr.create_ocr_engine", return_value="engine"),
            patch(
                "src.core.io.load_document",
                return_value=DocumentImage(np.zeros((1, 1), dtype=np.uint8)),
            ),
            patch(
                "src.table.process_table_image",
                return_value=Mock(grid=GridStructure([], [], [])),
            ),
            patch(
                "src.ocr.export_table_ocr",
                return_value=Mock(
                    table=OcrTable(cells=[], row_count=0, col_count=0),
                    csv_path=None,
                    json_path=None,
                ),
            ),
            patch("src.table.render_grid_structure", return_value=np.zeros((1, 1))),
            patch("src.core.visualization.show"),
        ):
            runpy.run_module("main", run_name="__main__")


if __name__ == "__main__":
    unittest.main()
