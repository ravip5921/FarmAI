from __future__ import annotations

import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import streamlit_app
from src.core.image import DocumentImage
from src.core.io import LoadedDocument
from src.ocr import OcrCell, OcrTable
from streamlit_app import (
    AppResult,
    bgr_to_rgb,
    build_pipeline,
    dataframe_to_csv,
    load_first_page,
    parse_filter_out_columns,
    render_empty_state,
    resolve_filter_out_columns,
    run_farmai,
    summarize_ocr,
    table_to_dataframe,
)


class _SessionState(dict):
    def __getattr__(self, name: str):
        return self[name]

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


class _Context:
    def __init__(self, st: "_FakeStreamlit") -> None:
        self.st = st

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def subheader(self, value: str) -> None:
        self.st.calls.append(("subheader", value))

    def caption(self, value: str) -> None:
        self.st.calls.append(("caption", value))

    def image(self, value, *, use_container_width: bool) -> None:
        self.st.calls.append(("image", use_container_width))

    def metric(self, label: str, value) -> None:
        self.st.calls.append(("metric", label, value))


class _FakeStreamlit(types.ModuleType):
    def __init__(
        self,
        *,
        uploaded_file=None,
        engine: str = "tesseract",
        template: str = "",
        text_values: list[str] | None = None,
        buttons: list[bool] | None = None,
        session_state: _SessionState | None = None,
    ) -> None:
        super().__init__("streamlit")
        self.uploaded_file = uploaded_file
        self.engine = engine
        self.template = template
        self.text_values = list(text_values or [])
        self.buttons = list(buttons or [])
        self.session_state = session_state or _SessionState()
        self.sidebar = _Context(self)
        self.calls: list[tuple] = []

    def set_page_config(self, **kwargs) -> None:
        self.calls.append(("set_page_config", kwargs))

    def markdown(self, value: str, **kwargs) -> None:
        self.calls.append(("markdown", value, kwargs))

    def title(self, value: str) -> None:
        self.calls.append(("title", value))

    def header(self, value: str) -> None:
        self.calls.append(("header", value))

    def file_uploader(self, *args, **kwargs):
        return self.uploaded_file

    def selectbox(self, *args, **kwargs):
        if args and args[0] == "Template":
            return self.template
        return self.engine

    def text_input(self, *args, **kwargs):
        if self.text_values:
            return self.text_values.pop(0)
        return kwargs.get("value", "")

    def button(self, *args, **kwargs):
        if self.buttons:
            return self.buttons.pop(0)
        return False

    def divider(self) -> None:
        self.calls.append(("divider",))

    def caption(self, value: str) -> None:
        self.calls.append(("caption", value))

    def info(self, value: str) -> None:
        self.calls.append(("info", value))

    def error(self, value: str) -> None:
        self.calls.append(("error", value))

    def spinner(self, value: str):
        self.calls.append(("spinner", value))
        return _Context(self)

    def columns(self, spec, gap: str | None = None):
        size = spec if isinstance(spec, int) else len(spec)
        return [_Context(self) for _ in range(size)]

    def tabs(self, labels: list[str]):
        self.calls.append(("tabs", labels))
        return [_Context(self) for _ in labels]

    def subheader(self, value: str) -> None:
        self.calls.append(("subheader", value))

    def image(self, value, *, use_container_width: bool) -> None:
        self.calls.append(("image", use_container_width))

    def data_editor(self, df, **kwargs):
        self.calls.append(("data_editor", kwargs))
        return df

    def download_button(self, *args, **kwargs) -> None:
        self.calls.append(("download_button", args, kwargs))


class _UploadedFile:
    name = "record.png"

    def getvalue(self) -> bytes:
        return b"data"


def _app_result() -> AppResult:
    table = OcrTable(
        cells=[OcrCell(row=0, col=0, bbox=(0, 0, 1, 1), text="Pen")],
        row_count=1,
        col_count=1,
    )
    df = pd.DataFrame([["A1"]], columns=["Pen"])
    image = np.zeros((2, 2), dtype=np.uint8)
    return AppResult(
        file_name="record.png",
        page=DocumentImage(image),
        processed=DocumentImage(image),
        table_result=Mock(),
        ocr_table=table,
        source_preview=image,
        overlay_preview=image,
        grid_preview=image,
        line_preview=image,
        original_df=df,
        ocr_engine_name="tesseract",
    )


class TestStreamlitColumnFilter(unittest.TestCase):
    def test_build_pipeline_returns_four_stage_pipeline(self) -> None:
        self.assertEqual(len(build_pipeline().stages), 4)

    def test_bgr_to_rgb_converts_color_and_keeps_grayscale(self) -> None:
        gray = np.zeros((2, 2), dtype=np.uint8)
        bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)

        self.assertIs(bgr_to_rgb(gray), gray)
        self.assertEqual(bgr_to_rgb(bgr).tolist(), [[[0, 0, 255]]])

    def test_table_to_dataframe_handles_empty_table(self) -> None:
        df = table_to_dataframe(OcrTable(cells=[], row_count=0, col_count=2))

        self.assertEqual(list(df.columns), ["Column 1", "Column 2"])
        self.assertEqual(df.values.tolist(), [])

    def test_table_to_dataframe_uses_first_ocr_row_as_headers(self) -> None:
        table = OcrTable(
            cells=[
                OcrCell(row=0, col=0, bbox=(0, 0, 1, 1), text="Pen"),
                OcrCell(row=0, col=1, bbox=(1, 0, 1, 1), text="Date"),
                OcrCell(row=1, col=0, bbox=(0, 1, 1, 1), text="A1"),
                OcrCell(row=1, col=1, bbox=(1, 1, 1, 1), text="06/03/26"),
            ],
            row_count=2,
            col_count=2,
        )

        df = table_to_dataframe(table)

        self.assertEqual(list(df.columns), ["Pen", "Date"])
        self.assertEqual(df.values.tolist(), [["A1", "06/03/26"]])
        self.assertEqual(dataframe_to_csv(df), "Pen,Date\nA1,06/03/26\n")

    def test_table_to_dataframe_falls_back_for_blank_or_duplicate_headers(
        self,
    ) -> None:
        table = OcrTable(
            cells=[
                OcrCell(row=0, col=0, bbox=(0, 0, 1, 1), text="Name"),
                OcrCell(row=0, col=1, bbox=(1, 0, 1, 1), text=" "),
                OcrCell(row=0, col=2, bbox=(2, 0, 1, 1), text="Name"),
            ],
            row_count=1,
            col_count=3,
        )

        df = table_to_dataframe(table)

        self.assertEqual(list(df.columns), ["Name", "Column 2", "Name (2)"])
        self.assertEqual(df.values.tolist(), [])

    def test_parse_filter_out_columns_reads_comma_separated_values(self) -> None:
        self.assertEqual(
            parse_filter_out_columns(" Pen, Date , , Notes "),
            {"Pen", "Date", "Notes"},
        )

    def test_resolve_filter_out_columns_uses_defaults_when_ui_is_empty(self) -> None:
        with patch("streamlit_app.FILTER_OUT_COLUMNS", {"Default"}):
            self.assertEqual(resolve_filter_out_columns(set()), {"Default"})

    def test_resolve_filter_out_columns_uses_ui_values_when_present(self) -> None:
        with patch("streamlit_app.FILTER_OUT_COLUMNS", {"Default"}):
            self.assertEqual(resolve_filter_out_columns({"UI"}), {"UI"})

    def test_summarize_ocr_counts_average_and_flagged_cells(self) -> None:
        table = OcrTable(
            cells=[
                OcrCell(row=0, col=0, bbox=(0, 0, 1, 1), text="ok", confidence=95),
                OcrCell(row=0, col=1, bbox=(0, 0, 1, 1), text="", confidence=80),
                OcrCell(row=0, col=2, bbox=(0, 0, 1, 1), text="x"),
            ],
            row_count=1,
            col_count=3,
        )

        self.assertEqual(summarize_ocr(table), (3, 88, 2))
        self.assertEqual(summarize_ocr(OcrTable([], 0, 0)), (0, 0, 0))

    def test_load_first_page_returns_first_pdf_page(self) -> None:
        first = DocumentImage(np.zeros((1, 1)))
        second = DocumentImage(np.ones((1, 1)))
        loaded = LoadedDocument([first, second], Path("record.pdf"))

        with patch("streamlit_app.load_document", return_value=loaded):
            self.assertIs(load_first_page(_UploadedFile()), first)

    def test_load_first_page_returns_image(self) -> None:
        image = DocumentImage(np.zeros((1, 1)))

        with patch("streamlit_app.load_document", return_value=image):
            self.assertIs(load_first_page(_UploadedFile()), image)

    def test_run_farmai_builds_previews_and_dataframe(self) -> None:
        page = DocumentImage(np.zeros((2, 2, 3), dtype=np.uint8))
        processed = DocumentImage(np.ones((2, 2), dtype=np.uint8))
        table = OcrTable(
            cells=[
                OcrCell(row=0, col=0, bbox=(0, 0, 1, 1), text="Pen"),
                OcrCell(row=1, col=0, bbox=(0, 1, 1, 1), text="A1"),
            ],
            row_count=2,
            col_count=1,
        )
        pipeline = Mock()
        pipeline.run.return_value = processed
        table_result = Mock()
        table_result.grid = "grid"
        table_result.line_detection.horizontal_mask = np.zeros((2, 2), dtype=np.uint8)
        table_result.line_detection.vertical_mask = np.ones((2, 2), dtype=np.uint8)

        with (
            patch("streamlit_app.load_first_page", return_value=page),
            patch("streamlit_app.build_pipeline", return_value=pipeline),
            patch("streamlit_app.load_template") as load_template,
            patch("streamlit_app.process_table_image", return_value=table_result) as process_table,
            patch("streamlit_app.create_ocr_engine", return_value="engine"),
            patch(
                "streamlit_app.export_table_ocr",
                return_value=Mock(table=table),
            ) as export,
            patch(
                "streamlit_app.render_grid_overlay",
                return_value=np.zeros((2, 2, 3), dtype=np.uint8),
            ),
            patch(
                "streamlit_app.render_grid_structure",
                return_value=np.zeros((2, 2), dtype=np.uint8),
            ),
        ):
            result = run_farmai(
                _UploadedFile(),
                ocr_engine_name="trocr-handwritten",
                trocr_model_name="custom/model",
                template_id=None,
                filter_out_columns={"Pen"},
            )

        self.assertEqual(result.original_df.values.tolist(), [["A1"]])
        self.assertEqual(result.line_preview.shape, (2, 4))
        load_template.assert_not_called()
        self.assertIsNone(process_table.call_args.kwargs["template"])
        self.assertEqual(export.call_args.kwargs["filter_out_columns"], {"Pen"})

    def test_apply_styles_and_empty_state_render_streamlit_calls(self) -> None:
        fake_st = _FakeStreamlit()

        with patch("streamlit_app.st", fake_st):
            streamlit_app.apply_styles()
            render_empty_state()

        self.assertTrue(any(call[0] == "markdown" for call in fake_st.calls))
        self.assertIn(
            ("info", "Upload a farm record image or single-page PDF, then press Go."),
            fake_st.calls,
        )

    def test_main_renders_empty_state_without_upload(self) -> None:
        fake_st = _FakeStreamlit(buttons=[False, False])

        with patch("streamlit_app.st", fake_st):
            streamlit_app.main()

        self.assertTrue(any(call[0] == "info" for call in fake_st.calls))

    def test_main_runs_farmai_and_renders_result(self) -> None:
        fake_st = _FakeStreamlit(
            uploaded_file=_UploadedFile(),
            engine="trocr-handwritten",
            text_values=["custom/model", "Pen, Date"],
            buttons=[True, False],
        )

        with (
            patch("streamlit_app.st", fake_st),
            patch("streamlit_app.run_farmai", return_value=_app_result()) as run,
        ):
            streamlit_app.main()

        self.assertEqual(run.call_args.kwargs["filter_out_columns"], {"Pen", "Date"})
        self.assertIsNone(run.call_args.kwargs["template_id"])
        self.assertIn("result", fake_st.session_state)
        self.assertTrue(any(call[0] == "download_button" for call in fake_st.calls))

    def test_main_reset_restores_original_dataframe(self) -> None:
        result = _app_result()
        fake_st = _FakeStreamlit(
            session_state=_SessionState(
                result=result,
                edited_df=pd.DataFrame([["changed"]], columns=["Pen"]),
            ),
            buttons=[False, True],
        )

        with patch("streamlit_app.st", fake_st):
            streamlit_app.main()

        self.assertEqual(fake_st.session_state.edited_df.values.tolist(), [["A1"]])

    def test_main_reports_run_errors(self) -> None:
        fake_st = _FakeStreamlit(
            uploaded_file=_UploadedFile(),
            buttons=[True, False],
        )

        with (
            patch("streamlit_app.st", fake_st),
            patch("streamlit_app.run_farmai", side_effect=RuntimeError("boom")),
        ):
            streamlit_app.main()

        self.assertIn(("error", "boom"), fake_st.calls)

    def test_module_entrypoint_runs_streamlit_main(self) -> None:
        fake_st = _FakeStreamlit(buttons=[False, False])

        with patch.dict(sys.modules, {"streamlit": fake_st}):
            runpy.run_module("streamlit_app", run_name="__main__")

        self.assertTrue(any(call[0] == "info" for call in fake_st.calls))


if __name__ == "__main__":
    unittest.main()
