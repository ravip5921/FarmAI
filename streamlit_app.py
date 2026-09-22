from __future__ import annotations

import csv
import io
import tempfile
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.core.image import DocumentImage
from src.core.io import LoadedDocument, load_document
from src.core.pipeline import Pipeline
from src.ocr import (
    DEFAULT_OCR_ENGINE,
    FILTER_OUT_COLUMNS,
    OcrTable,
    build_column_ocr_rules,
    create_ocr_engine,
    export_table_ocr,
    get_ocr_engine_specs,
)
from src.preprocessing.denoise import MorphologicalDenoiseStage
from src.preprocessing.grayscale import GrayscaleStage
from src.preprocessing.sauvola import SauvolaBinarizationStage
from src.preprocessing.skew import SkewCorrectionStage
from src.templates import FormTemplate, get_template_ids, load_template
from src.table import (
    TablePipelineResult,
    process_table_image,
    render_grid_overlay,
    render_grid_structure,
)


LOW_CONFIDENCE_THRESHOLD = 90.0
SUPPORTED_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "pdf"]


@dataclass
class AppResult:
    file_name: str
    page: DocumentImage
    processed: DocumentImage
    table_result: TablePipelineResult
    ocr_table: OcrTable
    source_preview: np.ndarray
    overlay_preview: np.ndarray
    grid_preview: np.ndarray
    line_preview: np.ndarray
    original_df: pd.DataFrame
    ocr_engine_name: str
    template_id: str | None = None


def build_pipeline(
    window_size: int = 25,
    k: float = 0.34,
    denoise_kernel: int = 2,
) -> Pipeline:
    return Pipeline(
        [
            GrayscaleStage(),
            SauvolaBinarizationStage(window_size=window_size, k=k),
            MorphologicalDenoiseStage(kernel_size=denoise_kernel),
            SkewCorrectionStage(),
        ]
    )


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def _make_unique_columns(headers: list[str]) -> list[str]:
    columns: list[str] = []
    counts: dict[str, int] = {}
    for index, header in enumerate(headers):
        base = header.strip() or f"Column {index + 1}"
        counts[base] = counts.get(base, 0) + 1
        if counts[base] == 1:
            columns.append(base)
        else:
            columns.append(f"{base} ({counts[base]})")
    return columns


def table_to_dataframe(table: OcrTable) -> pd.DataFrame:
    matrix = table.text_matrix(fill_value="")
    if not matrix:
        columns = [f"Column {index + 1}" for index in range(table.col_count)]
        return pd.DataFrame(columns=columns)

    columns = _make_unique_columns(matrix[0])
    return pd.DataFrame(matrix[1:], columns=columns)


def dataframe_to_csv(df: pd.DataFrame) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(list(df.columns))
    writer.writerows(df.fillna("").astype(str).values.tolist())
    return output.getvalue()


def parse_filter_out_columns(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def resolve_filter_out_columns(ui_columns: Collection[str] | None) -> Collection[str]:
    if ui_columns:
        return set(ui_columns)
    return FILTER_OUT_COLUMNS


def resolve_template_filter_indices(
    template: FormTemplate | None,
    ui_columns: Collection[str] | None,
) -> set[int] | None:
    if template is None:
        return None
    return template.filtered_column_indices | template.indices_for_column_names(
        set(ui_columns or ())
    )


def summarize_ocr(table: OcrTable) -> tuple[int, int, int]:
    confidences = [
        float(cell.confidence)
        for cell in table.cells
        if cell.confidence is not None
    ]
    average = round(sum(confidences) / len(confidences)) if confidences else 0
    flagged = sum(
        1
        for cell in table.cells
        if not cell.text
        or cell.confidence is None
        or cell.confidence < LOW_CONFIDENCE_THRESHOLD
    )
    return len(table.cells), average, flagged


def load_first_page(uploaded_file: Any) -> DocumentImage:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"uploaded{suffix}"
        path.write_bytes(uploaded_file.getvalue())
        loaded = load_document(path)

    if isinstance(loaded, LoadedDocument):
        return loaded.pages[0]
    return loaded


def run_farmai(
    uploaded_file: Any,
    *,
    ocr_engine_name: str = DEFAULT_OCR_ENGINE,
    trocr_model_name: str = "microsoft/trocr-base-handwritten",
    template_id: str | None = None,
    filter_out_columns: Collection[str] | None = None,
) -> AppResult:
    page = load_first_page(uploaded_file)
    pipeline = build_pipeline()
    processed = pipeline.run(page)
    template = load_template(template_id) if template_id else None
    table_result = process_table_image(
        processed.image,
        image_name=uploaded_file.name,
        template=template,
    )
    ocr_engine = create_ocr_engine(
        ocr_engine_name,
        model_name=trocr_model_name,
    )
    ocr_result = export_table_ocr(
        page.image,
        table_result.grid,
        engine=ocr_engine,
        padding=2,
        filter_out_columns=(
            set()
            if template is not None
            else resolve_filter_out_columns(filter_out_columns)
        ),
        filter_out_column_indices=resolve_template_filter_indices(
            template,
            filter_out_columns,
        ),
        column_names=template.column_names if template is not None else None,
        column_keys=template.column_keys if template is not None else None,
        column_ocr_rules=(
            build_column_ocr_rules(template.columns) if template is not None else None
        ),
    )

    overlay = render_grid_overlay(page.image, table_result.grid)
    grid = render_grid_structure(table_result.grid, processed.image.shape)
    line_preview = np.hstack(
        [
            table_result.line_detection.horizontal_mask,
            table_result.line_detection.vertical_mask,
        ]
    )
    df = table_to_dataframe(ocr_result.table)

    return AppResult(
        file_name=uploaded_file.name,
        page=page,
        processed=processed,
        table_result=table_result,
        ocr_table=ocr_result.table,
        source_preview=bgr_to_rgb(page.image),
        overlay_preview=bgr_to_rgb(overlay),
        grid_preview=grid,
        line_preview=line_preview,
        original_df=df,
        ocr_engine_name=ocr_engine_name,
        template_id=template.id if template is not None else None,
    )


def apply_styles() -> None:
    st.markdown(
        """
        <style>
          .block-container {
            max-width: 1500px;
            padding-top: 1.1rem;
            padding-bottom: 1.6rem;
          }
          [data-testid="stMetric"] {
            border: 1px solid #d8ded8;
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            background: #ffffff;
          }
          div[data-testid="stHorizontalBlock"] {
            align-items: stretch;
          }
          .farmai-subtitle {
            color: #5d6b62;
            margin-top: -0.6rem;
            margin-bottom: 1rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.info("Upload a farm record image or single-page PDF, then press Go.")


def main() -> None:
    st.set_page_config(page_title="FarmAI Review", layout="wide")
    apply_styles()

    st.title("FarmAI Review")
    st.markdown(
        '<p class="farmai-subtitle">Upload a scanned record, run table OCR, then correct the extracted cells before downloading CSV.</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Input")
        uploaded_file = st.file_uploader(
            "Image or PDF",
            type=SUPPORTED_TYPES,
            accept_multiple_files=False,
        )
        engine_specs = get_ocr_engine_specs()
        engine_names = [spec.name for spec in engine_specs]
        engine_labels = {spec.name: spec.label for spec in engine_specs}
        ocr_engine_name = st.selectbox(
            "OCR engine",
            options=engine_names,
            index=engine_names.index(DEFAULT_OCR_ENGINE),
            format_func=lambda name: engine_labels[name],
        )
        template_ids = get_template_ids()
        template_options = [""] + template_ids
        template_id = st.selectbox(
            "Template",
            options=template_options,
            format_func=lambda value: "None" if not value else value,
        )
        trocr_model_name = "microsoft/trocr-base-handwritten"
        if ocr_engine_name == "trocr-handwritten":
            trocr_model_name = st.text_input(
                "TrOCR model",
                value=trocr_model_name,
            )
        filter_out_columns_text = st.text_input(
            "Filter out columns",
            value="",
            placeholder="Pen, Date, Notes",
        )
        run_clicked = st.button(
            "Go",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )
        reset_clicked = st.button(
            "Reset table",
            use_container_width=True,
            disabled="result" not in st.session_state,
        )

        st.divider()
        st.caption("PDF support uses the first page only in this local version.")

    if run_clicked and uploaded_file is not None:
        try:
            with st.spinner("Detecting table and reading cells..."):
                result = run_farmai(
                    uploaded_file,
                    ocr_engine_name=ocr_engine_name,
                    trocr_model_name=trocr_model_name,
                    template_id=template_id or None,
                    filter_out_columns=parse_filter_out_columns(
                        filter_out_columns_text
                    ),
                )
            st.session_state.result = result
            st.session_state.edited_df = result.original_df.copy()
        except Exception as exc:
            st.error(str(exc))
            return

    if reset_clicked and "result" in st.session_state:
        st.session_state.edited_df = st.session_state.result.original_df.copy()

    if "result" not in st.session_state:
        render_empty_state()
        return

    result: AppResult = st.session_state.result
    left, right = st.columns([0.92, 1.08], gap="large")

    with left:
        st.subheader("Document Preview")
        st.caption(result.file_name)
        preview_tabs = st.tabs(["Source", "Overlay", "Grid", "Lines"])
        with preview_tabs[0]:
            st.image(result.source_preview, use_container_width=True)
        with preview_tabs[1]:
            st.image(result.overlay_preview, use_container_width=True)
        with preview_tabs[2]:
            st.image(result.grid_preview, use_container_width=True)
        with preview_tabs[3]:
            st.image(result.line_preview, use_container_width=True)

    with right:
        st.subheader("Editable Table Output")
        st.caption(f"OCR engine: {result.ocr_engine_name}")
        if result.template_id:
            st.caption(f"Template: {result.template_id}")
        cells, average, flagged = summarize_ocr(result.ocr_table)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Cells detected", cells)
        metric_cols[1].metric("Average confidence", f"{average}%")
        metric_cols[2].metric("Cells flagged", flagged)

        edited_df = st.data_editor(
            st.session_state.edited_df,
            key="editable_table",
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
        )
        st.session_state.edited_df = edited_df

        csv_data = dataframe_to_csv(edited_df)
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"{Path(result.file_name).stem}_farmai_table.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
