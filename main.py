from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2

from src.core.image import DocumentImage
from src.core.io import LoadedDocument, load_document
from src.core.pipeline import Pipeline
from src.core.visualization import save_debug, show
from src.export.csv_export import table_to_csv_string
from src.ocr import (
    CellOcrEngine,
    DEFAULT_OCR_ENGINE,
    FILTER_OUT_COLUMNS,
    create_ocr_engine,
    export_table_ocr,
    get_ocr_engine_names,
)
from src.preprocessing.denoise import MorphologicalDenoiseStage
from src.preprocessing.grayscale import GrayscaleStage
from src.templates import FormTemplate, get_template_ids, load_template
from src.preprocessing.sauvola import SauvolaBinarizationStage
from src.preprocessing.skew import SkewCorrectionStage
from src.table import (
    TablePipelineResult,
    correct_table_perspective,
    process_table_image,
    render_grid_overlay,
    render_grid_structure,
    render_perspective_corners,
    warp_image_to_corners,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Farm AI preprocessing and table extraction on an image."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "debug_outputs",
        help="Directory for intermediate debug images",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable saving intermediate preprocessing and table debug images",
    )
    parser.add_argument(
        "--save-line-detection",
        action="store_true",
        help="Save table line-detection previews",
    )
    parser.add_argument(
        "--save-intersections",
        action="store_true",
        help="Save table intersection previews",
    )
    parser.add_argument(
        "--save-image",
        action="store_true",
        help="Save the final detected table image",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Save the detected table grid overlaid on the original image",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save OCR output as a CSV file after table detection",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save OCR output as a JSON file after table detection",
    )
    parser.add_argument(
        "--save-cells",
        action="store_true",
        help="Save cropped OCR cell images under debug output",
    )
    parser.add_argument(
        "--no-print-csv",
        action="store_true",
        help="Do not print OCR output as CSV after processing",
    )
    parser.add_argument(
        "--ocr-padding",
        type=int,
        default=2,
        help="Padding to trim from each extracted cell before OCR",
    )
    parser.add_argument(
        "--ocr-context-padding",
        type=int,
        default=0,
        help="Extra pixels to expand each cell crop outward before OCR",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=get_ocr_engine_names(),
        default=DEFAULT_OCR_ENGINE,
        help="OCR backend to use for per-cell image-to-text recognition",
    )
    parser.add_argument(
        "--template",
        choices=get_template_ids(),
        default=None,
        help="Known form template used to repair the detected table grid",
    )
    parser.add_argument(
        "--perspective-correct",
        action="store_true",
        help="Warp the detected table to a rectangle before final table OCR",
    )
    parser.add_argument(
        "--perspective-padding",
        type=int,
        default=24,
        help="Pixels to expand detected table corners before perspective warp",
    )
    parser.add_argument(
        "--trocr-model",
        default="microsoft/trocr-base-handwritten",
        help="Hugging Face model name used when --ocr-engine=trocr-handwritten",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help=(
            "Save all debug images and OCR exports "
            "(line detection, intersections, final table, overlay, cells, CSV, JSON)"
        ),
    )
    return parser.parse_args()


def build_pipeline(
    window_size: int = 25,
    k: float = 0.34,
    denoise_kernel: int = 3,
) -> Pipeline:
    """Build a preprocessing pipeline with configurable parameters."""
    return Pipeline(
        [
            GrayscaleStage(),
            SauvolaBinarizationStage(window_size=window_size, k=k),
            MorphologicalDenoiseStage(kernel_size=denoise_kernel),
            SkewCorrectionStage(),
        ]
    )


def process_image(
    image_source: str | Path | DocumentImage,
    pipeline: Pipeline,
    debug_dir: Path | None = None,
    image_name: str | Path | None = None,
) -> DocumentImage:
    """Process a single image through the pipeline."""
    if isinstance(image_source, DocumentImage):
        doc = image_source
        source_label = str(image_name or "document")
    else:
        image = cv2.imread(str(image_source))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_source}")
        doc = DocumentImage(image)
        source_label = str(image_name or image_source)

    stem = Path(source_label).stem

    # Run the pipeline: output of stage N becomes input to stage N+1
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        current = doc
        for stage_index, stage in enumerate(pipeline.stages, start=1):
            current = stage.process(current)
            stage_name = stage.__class__.__name__
            filename = debug_dir / f"stage_{stage_index:02d}_{stage_name}_{stem}.png"
            save_debug(filename, stage_name, current)
        return current
    else:
        # If no debug, just run the whole pipeline at once
        return pipeline.run(doc)


def process_table(
    bitmap: Any,
    *,
    image_name: str | Path | None = None,
    debug_dir: Path | None = None,
    save_line_detection: bool = False,
    save_intersections: bool = False,
    template: FormTemplate | None = None,
) -> TablePipelineResult:
    """Run table-structure extraction after preprocessing."""
    return process_table_image(
        bitmap,
        image_name=image_name,
        debug_dir=debug_dir,
        save_line_detection=save_line_detection,
        save_intersections=save_intersections,
        template=template,
    )


def _run_page(
    page: DocumentImage,
    *,
    pipeline: Pipeline,
    debug_dir: Path | None,
    image_name: str,
    args: argparse.Namespace,
    ocr_engine: CellOcrEngine,
    template: FormTemplate | None = None,
    source_path: Path | None = None,
) -> None:
    file_debug_dir = (
        debug_dir / (Path(str(image_name)).stem or "table")
        if debug_dir is not None
        else None
    )
    result = process_image(
        page,
        pipeline,
        debug_dir=file_debug_dir,
        image_name=image_name,
    )
    ocr_image = page.image
    table_result = process_table(
        result.image,
        image_name=image_name,
        debug_dir=file_debug_dir,
        save_line_detection=args.save_line_detection or args.save_all,
        save_intersections=args.save_intersections or args.save_all,
        template=template,
    )
    if args.perspective_correct:
        correction = correct_table_perspective(
            result.image,
            table_result.line_detection.horizontal_mask,
            table_result.line_detection.vertical_mask,
            padding=args.perspective_padding,
        )
        if file_debug_dir is not None and args.save_all:
            file_debug_dir.mkdir(parents=True, exist_ok=True)
            save_debug(
                file_debug_dir / f"perspective_corners_{image_name}.png",
                "Perspective Corners",
                render_perspective_corners(
                    result.image,
                    correction.corners,
                    padded_corners=correction.padded_corners,
                ),
            )
        if correction.corrected:
            ocr_correction = warp_image_to_corners(
                page.image,
                correction.corners,
                padding=args.perspective_padding,
                output_size=correction.output_size,
            )
            ocr_image = ocr_correction.image
            result = DocumentImage(
                correction.image,
                {
                    **result.metadata,
                    "perspective_corrected": True,
                    "perspective_padding": args.perspective_padding,
                    "perspective_output_size": correction.output_size,
                },
            )
            if file_debug_dir is not None and args.save_all:
                save_debug(
                    file_debug_dir / f"perspective_corrected_{image_name}.png",
                    "Perspective Corrected Table",
                    result.image,
                )
            table_result = process_table(
                result.image,
                image_name=f"{image_name}_perspective",
                debug_dir=file_debug_dir,
                save_line_detection=args.save_line_detection or args.save_all,
                save_intersections=args.save_intersections or args.save_all,
                template=template,
            )
    ocr_result = process_ocr(
        ocr_image,
        table_result,
        image_name=image_name,
        debug_dir=file_debug_dir,
        save_csv=args.save_csv or args.save_all,
        save_json=args.save_json or args.save_all,
        save_cells=args.save_cells or args.save_all,
        padding=args.ocr_padding,
        context_padding=args.ocr_context_padding,
        engine=ocr_engine,
        template=template,
    )
    table_image = render_grid_structure(table_result.grid, result.image.shape)

    if file_debug_dir is not None and (args.save_image or args.save_all):
        file_debug_dir.mkdir(parents=True, exist_ok=True)
        print(
            "Saving final detected table image to: "
            f"{file_debug_dir / f'final_table_{image_name}.png'}"
        )
        save_debug(
            file_debug_dir / f"final_table_{image_name}.png",
            "Final Detected Table",
            table_image,
        )
    if file_debug_dir is not None and (args.save_overlay or args.save_all):
        overlay = render_grid_overlay(ocr_image, table_result.grid)
        print(
            "Saving table overlay image to: "
            f"{file_debug_dir / f'table_overlay_{image_name}.png'}"
        )
        save_debug(
            file_debug_dir / f"table_overlay_{image_name}.png",
            "Table Overlay",
            overlay,
        )
    if ocr_result.csv_path is not None:
        print(f"Saved OCR CSV to: {ocr_result.csv_path}")
    if ocr_result.json_path is not None:
        print(f"Saved OCR JSON to: {ocr_result.json_path}")
    if not args.no_print_csv:
        print_ocr_csv(ocr_result.table, image_name=image_name)
    show("Final Output", result)
    show("Detected Table", table_image)


def print_ocr_csv(table, *, image_name: str | Path | None = None) -> None:
    """Print OCR table output as CSV for command-line inspection."""
    label = Path(str(image_name or "table")).stem or "table"
    print(f"\n--- OCR CSV: {label} ---")
    print(table_to_csv_string(table), end="")
    print(f"--- END OCR CSV: {label} ---\n")


def process_ocr(
    image,
    table_result: TablePipelineResult,
    *,
    image_name: str | Path | None = None,
    debug_dir: Path | None = None,
    save_csv: bool = False,
    save_json: bool = False,
    save_cells: bool = False,
    padding: int = 2,
    context_padding: int = 0,
    engine: CellOcrEngine | None = None,
    template: FormTemplate | None = None,
):
    """Run OCR for the detected grid and optionally save CSV/JSON exports."""
    stem = Path(str(image_name or "table")).stem or "table"
    csv_path = json_path = None
    cell_image_dir = None
    if debug_dir is not None and (save_csv or save_json or save_cells):
        debug_dir.mkdir(parents=True, exist_ok=True)
        if save_csv:
            csv_path = debug_dir / f"table_ocr_{stem}.csv"
        if save_json:
            json_path = debug_dir / f"table_ocr_{stem}.json"
        if save_cells:
            cell_image_dir = debug_dir / f"{stem}_cells"
    if template is None:
        filter_out_columns = FILTER_OUT_COLUMNS
        filter_out_column_indices = None
        column_names = None
        column_keys = None
    else:
        filter_out_columns = set()
        filter_out_column_indices = (
            template.filtered_column_indices
            | template.indices_for_column_names(set(FILTER_OUT_COLUMNS))
        )
        column_names = template.column_names
        column_keys = template.column_keys
    return export_table_ocr(
        image,
        table_result.grid,
        csv_path=csv_path,
        json_path=json_path,
        engine=engine,
        padding=padding,
        context_padding=context_padding,
        filter_out_columns=filter_out_columns,
        filter_out_column_indices=filter_out_column_indices,
        column_names=column_names,
        column_keys=column_keys,
        cell_image_dir=cell_image_dir,
    )


def main() -> None:
    args = parse_args()
    image_path = args.image_path
    debug_dir = None if args.no_debug else args.debug_dir
    pipeline = build_pipeline(window_size=25, k=0.34, denoise_kernel=2)
    ocr_engine = create_ocr_engine(
        args.ocr_engine,
        model_name=args.trocr_model,
    )
    template = load_template(args.template) if args.template else None
    loaded = load_document(image_path)

    if isinstance(loaded, LoadedDocument):
        for page in loaded.pages:
            image_name = f"{image_path.stem}_p{page.metadata.get('page_index', 1)}"
            _run_page(
                page,
                pipeline=pipeline,
                debug_dir=debug_dir,
                image_name=image_name,
                args=args,
                ocr_engine=ocr_engine,
                template=template,
                source_path=image_path,
            )
    else:
        image_name = image_path.stem
        _run_page(
            loaded,
            pipeline=pipeline,
            debug_dir=debug_dir,
            image_name=image_name,
            args=args,
            ocr_engine=ocr_engine,
            template=template,
            source_path=image_path,
        )


if __name__ == "__main__":
    main()
