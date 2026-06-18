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
    create_ocr_engine,
    export_table_ocr,
    get_ocr_engine_names,
)
from src.preprocessing.denoise import MorphologicalDenoiseStage
from src.preprocessing.grayscale import GrayscaleStage
from src.preprocessing.sauvola import SauvolaBinarizationStage
from src.preprocessing.skew import SkewCorrectionStage
from src.table import (
    TablePipelineResult,
    process_table_image,
    render_grid_overlay,
    render_grid_structure,
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
        "--ocr-engine",
        choices=get_ocr_engine_names(),
        default=DEFAULT_OCR_ENGINE,
        help="OCR backend to use for per-cell image-to-text recognition",
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
            "(line detection, intersections, final table, overlay, CSV, JSON)"
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
) -> TablePipelineResult:
    """Run table-structure extraction after preprocessing."""
    return process_table_image(
        bitmap,
        image_name=image_name,
        debug_dir=debug_dir,
        save_line_detection=save_line_detection,
        save_intersections=save_intersections,
    )


def _run_page(
    page: DocumentImage,
    *,
    pipeline: Pipeline,
    debug_dir: Path | None,
    image_name: str,
    args: argparse.Namespace,
    ocr_engine: CellOcrEngine,
    source_path: Path | None = None,
) -> None:
    result = process_image(page, pipeline, debug_dir=debug_dir, image_name=image_name)
    table_result = process_table(
        result.image,
        image_name=image_name,
        debug_dir=debug_dir,
        save_line_detection=args.save_line_detection or args.save_all,
        save_intersections=args.save_intersections or args.save_all,
    )
    ocr_result = process_ocr(
        result.image,
        table_result,
        image_name=image_name,
        debug_dir=debug_dir,
        save_csv=args.save_csv or args.save_all,
        save_json=args.save_json or args.save_all,
        padding=args.ocr_padding,
        engine=ocr_engine,
    )
    table_image = render_grid_structure(table_result.grid, result.image.shape)

    if debug_dir is not None and (args.save_image or args.save_all):
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Saving final detected table image to: {debug_dir / f'final_table_{image_name}.png'}"
        )
        save_debug(
            debug_dir / f"final_table_{image_name}.png",
            "Final Detected Table",
            table_image,
        )
    if debug_dir is not None and (args.save_overlay or args.save_all):
        overlay = render_grid_overlay(page.image, table_result.grid)
        print(
            f"Saving table overlay image to: {debug_dir / f'table_overlay_{image_name}.png'}"
        )
        save_debug(
            debug_dir / f"table_overlay_{image_name}.png", "Table Overlay", overlay
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
    padding: int = 2,
    engine: CellOcrEngine | None = None,
):
    """Run OCR for the detected grid and optionally save CSV/JSON exports."""
    stem = Path(str(image_name or "table")).stem or "table"
    csv_path = json_path = None
    if debug_dir is not None and (save_csv or save_json):
        debug_dir.mkdir(parents=True, exist_ok=True)
        if save_csv:
            csv_path = debug_dir / f"table_ocr_{stem}.csv"
        if save_json:
            json_path = debug_dir / f"table_ocr_{stem}.json"
    return export_table_ocr(
        image,
        table_result.grid,
        csv_path=csv_path,
        json_path=json_path,
        engine=engine,
        padding=padding,
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
            source_path=image_path,
        )


if __name__ == "__main__":
    main()
