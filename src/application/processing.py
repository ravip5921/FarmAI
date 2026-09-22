from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.core.image import DocumentImage
from src.core.io import LoadedDocument, load_document
from src.core.pipeline import Pipeline
from src.ocr import build_column_ocr_rules, create_ocr_engine, run_table_ocr
from src.ocr.base import CellOcrEngine
from src.preprocessing.denoise import MorphologicalDenoiseStage
from src.preprocessing.grayscale import GrayscaleStage
from src.preprocessing.sauvola import SauvolaBinarizationStage
from src.preprocessing.skew import SkewCorrectionStage, rotate_image
from src.privacy import redact_filtered_template_columns
from src.table import process_table_image, render_grid_overlay
from src.templates import FormTemplate, TemplateColumn, load_template

from .result_models import (
    DocumentProcessingResult,
    PageProcessingResult,
    ProcessingProgress,
    ProcessingSettings,
    UiCell,
    UiColumn,
)

ProgressCallback = Callable[[ProcessingProgress], None]


def _notify(
    callback: ProgressCallback | None,
    *,
    stage: str,
    completed: int = 0,
    total: int = 0,
    page_number: int = 1,
    page_count: int = 1,
    message: str = "",
) -> None:
    if callback is None:
        return
    callback(
        ProcessingProgress(
            stage=stage,
            completed=completed,
            total=total,
            page_number=page_number,
            page_count=page_count,
            message=message,
        )
    )


def _detection_pipeline() -> Pipeline:
    return Pipeline(
        [
            GrayscaleStage(),
            SauvolaBinarizationStage(window_size=25, k=0.34),
            MorphologicalDenoiseStage(kernel_size=2),
        ]
    )


def _pages(loaded: DocumentImage | LoadedDocument) -> list[DocumentImage]:
    return loaded.pages if isinstance(loaded, LoadedDocument) else [loaded]


def _kept_template_columns(
    template: FormTemplate,
    extra_filtered_columns: tuple[str, ...],
) -> tuple[list[TemplateColumn], set[int]]:
    filtered = template.filtered_column_indices | template.indices_for_column_names(
        set(extra_filtered_columns)
    )
    kept = [
        column
        for column in sorted(template.columns, key=lambda item: item.index)
        if column.index not in filtered
    ]
    return kept, filtered


def _ui_columns_from_template(columns: list[TemplateColumn]) -> list[UiColumn]:
    return [
        UiColumn(
            index=index,
            source_index=column.index,
            key=column.key,
            name=column.name,
            value_type=column.value_type,
            format=column.format,
            range_min=column.range_min,
            range_max=column.range_max,
        )
        for index, column in enumerate(columns)
    ]


def _generic_columns(table) -> list[UiColumn]:
    matrix = table.text_matrix(fill_value="")
    headers = matrix[0] if matrix else []
    return [
        UiColumn(
            index=index,
            source_index=index,
            key=f"column_{index + 1}",
            name=(headers[index].strip() if index < len(headers) else "")
            or f"Column {index + 1}",
        )
        for index in range(table.col_count)
    ]


def _ui_cells(table, columns: list[UiColumn]) -> list[UiCell]:
    mapped: list[UiCell] = []
    for cell in table.cells:
        if cell.row == 0 or not (0 <= cell.col < len(columns)):
            continue
        column = columns[cell.col]
        mapped.append(
            UiCell(
                row=cell.row,
                column_index=cell.col,
                source_column_index=column.source_index,
                column_key=column.key,
                column_name=column.name,
                bbox=cell.bbox,
                ocr_text=cell.text,
                confidence=cell.confidence,
                raw_text=cell.raw_text,
                validation_error=cell.validation_error,
            )
        )
    return mapped


def process_document(
    source_path: str | Path,
    *,
    settings: ProcessingSettings | None = None,
    progress_callback: ProgressCallback | None = None,
    engine: CellOcrEngine | None = None,
) -> DocumentProcessingResult:
    """Process a document into UI-ready page images and stable cell metadata."""
    resolved_settings = settings or ProcessingSettings()
    source = Path(source_path)
    loaded = load_document(source)
    pages = _pages(loaded)
    template = (
        load_template(resolved_settings.template_id)
        if resolved_settings.template_id
        else None
    )
    kept_template_columns: list[TemplateColumn] = []
    filtered_indices: set[int] | None = None
    if template is not None:
        kept_template_columns, filtered_indices = _kept_template_columns(
            template,
            resolved_settings.extra_filtered_columns,
        )
    ocr_engine = engine or create_ocr_engine(resolved_settings.ocr_engine)
    result = DocumentProcessingResult(
        filename=source.name,
        template_id=template.id if template else None,
        template_name=template.name if template else None,
        ocr_engine=resolved_settings.ocr_engine,
    )

    page_count = len(pages)
    for page_number, page in enumerate(pages, start=1):
        _notify(
            progress_callback,
            stage="preparing",
            page_number=page_number,
            page_count=page_count,
            message="Preparing image",
        )
        detection_base = _detection_pipeline().run(page)
        skew_stage = SkewCorrectionStage()
        skew_angle = skew_stage.estimate_angle(detection_base.image)
        detection_image = rotate_image(
            detection_base.image,
            skew_angle,
            is_binary=True,
        )
        deskewed_source = rotate_image(page.image, skew_angle, is_binary=False)

        _notify(
            progress_callback,
            stage="detecting_table",
            page_number=page_number,
            page_count=page_count,
            message="Finding the table",
        )
        table_result = process_table_image(
            detection_image,
            image_name=f"{source.stem}_p{page_number}",
            template=template,
        )
        if not table_result.grid.cells:
            raise ValueError("FarmAI could not find the table in this record.")
        review_source = (
            redact_filtered_template_columns(
                deskewed_source,
                table_result.grid,
                template,
                extra_filtered_columns=set(resolved_settings.extra_filtered_columns),
            )
            if template is not None
            else deskewed_source
        )

        def on_cell_progress(completed: int, total: int) -> None:
            _notify(
                progress_callback,
                stage="recognizing_cells",
                completed=completed,
                total=total,
                page_number=page_number,
                page_count=page_count,
                message=f"Reading cells ({completed} of {total})",
            )

        table = run_table_ocr(
            review_source,
            table_result.grid,
            engine=ocr_engine,
            padding=resolved_settings.ocr_padding,
            context_padding=resolved_settings.ocr_context_padding,
            filter_out_columns=(
                set(resolved_settings.extra_filtered_columns)
                if template is None
                else set()
            ),
            filter_out_column_indices=filtered_indices,
            column_names=template.column_names if template else None,
            column_keys=template.column_keys if template else None,
            column_ocr_rules=(
                build_column_ocr_rules(template.columns) if template else None
            ),
            progress_callback=on_cell_progress,
        )
        columns = (
            _ui_columns_from_template(kept_template_columns)
            if template
            else _generic_columns(table)
        )
        overlay = render_grid_overlay(review_source, table_result.grid)
        height, width = review_source.shape[:2]
        result.pages.append(
            PageProcessingResult(
                page_number=page_number,
                source_image=review_source,
                overlay_image=overlay,
                image_width=width,
                image_height=height,
                skew_angle=skew_angle,
                columns=columns,
                cells=_ui_cells(table, columns),
                data_row_count=max(0, table.row_count - 1),
            )
        )
        _notify(
            progress_callback,
            stage="preparing_review",
            completed=page_number,
            total=page_count,
            page_number=page_number,
            page_count=page_count,
            message="Preparing review",
        )

    _notify(
        progress_callback,
        stage="completed",
        completed=page_count,
        total=page_count,
        page_number=page_count,
        page_count=page_count,
        message="Complete",
    )
    return result
