from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.application.processing import (
    _detection_pipeline,
    _generic_columns,
    _notify,
    process_document,
)
from src.application.result_models import ProcessingSettings, UiCell, UiColumn
from src.core.image import DocumentImage
from src.ocr.base import OcrText
from src.table.grid_reconstruction import GridCell, GridStructure


class _Engine:
    def __init__(self) -> None:
        self.call_count = 0

    def recognize(self, image: np.ndarray) -> OcrText:
        self.call_count += 1
        return OcrText(text="70")


def _grid() -> GridStructure:
    row_coords = [0, 10, 20]
    col_coords = list(range(0, 101, 10))
    return GridStructure(
        row_coords=row_coords,
        col_coords=col_coords,
        cells=[
            GridCell(
                row=row,
                col=col,
                bbox=(col * 10, row * 10, 10, 10),
            )
            for row in range(2)
            for col in range(10)
        ],
    )


class TestApplicationProcessing(unittest.TestCase):
    def test_helpers_and_result_serialization(self) -> None:
        _notify(None, stage="ignored")
        self.assertEqual(len(_detection_pipeline().stages), 3)
        table = Mock()
        table.text_matrix.return_value = []
        table.col_count = 2
        self.assertEqual(
            [column.name for column in _generic_columns(table)],
            ["Column 1", "Column 2"],
        )
        self.assertEqual(UiColumn(0, 0, "x", "X").to_dict()["key"], "x")
        self.assertEqual(
            UiCell(1, 0, 0, "x", "X", (0, 0, 1, 1), "text").to_dict()["reviewed_text"],
            "text",
        )

    @patch("src.application.processing.process_table_image")
    @patch("src.application.processing._detection_pipeline")
    @patch("src.application.processing.load_document")
    def test_empty_table_reports_clear_error(
        self, load_document, detection_pipeline, process_table_image
    ) -> None:
        image = np.full((10, 10), 255, dtype=np.uint8)
        load_document.return_value = DocumentImage(image)
        detection_pipeline.return_value.run.return_value = DocumentImage(image)
        process_table_image.return_value = SimpleNamespace(
            grid=GridStructure([0, 10], [0, 10], [])
        )
        with patch(
            "src.application.processing.SkewCorrectionStage.estimate_angle",
            return_value=0,
        ):
            with self.assertRaisesRegex(ValueError, "could not find the table"):
                process_document("missing.png", engine=_Engine())

    @patch("src.application.processing.process_table_image")
    @patch("src.application.processing._detection_pipeline")
    @patch("src.application.processing.load_document")
    def test_returns_template_keys_and_cell_progress(
        self,
        load_document,
        detection_pipeline,
        process_table_image,
    ) -> None:
        source = np.full((20, 100, 3), 255, dtype=np.uint8)
        load_document.return_value = DocumentImage(source)
        pipeline = Mock()
        pipeline.run.return_value = DocumentImage(
            np.full((20, 100), 255, dtype=np.uint8),
            metadata={"binary": True},
        )
        detection_pipeline.return_value = pipeline
        process_table_image.return_value = SimpleNamespace(grid=_grid())
        engine = _Engine()
        progress = []

        with patch(
            "src.application.processing.SkewCorrectionStage.estimate_angle",
            return_value=2.0,
        ):
            result = process_document(
                "record.png",
                settings=ProcessingSettings(template_id="boar_room"),
                progress_callback=progress.append,
                engine=engine,
            )

        self.assertEqual(len(result.pages), 1)
        page = result.pages[0]
        self.assertEqual(
            [column.key for column in page.columns],
            ["date", "current_temperature", "hi", "lo", "comments"],
        )
        self.assertEqual(len(page.cells), 5)
        self.assertEqual(engine.call_count, 5)
        cell_progress = [item for item in progress if item.stage == "recognizing_cells"]
        self.assertEqual(cell_progress[-1].completed, 5)
        self.assertEqual(cell_progress[-1].total, 5)
        self.assertEqual(page.source_image.shape, source.shape)
        self.assertEqual(page.overlay_image.shape, source.shape)


if __name__ == "__main__":
    unittest.main()
