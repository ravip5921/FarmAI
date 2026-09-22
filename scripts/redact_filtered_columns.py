from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.core.image import DocumentImage
from src.core.io import LoadedDocument, load_document
from src.core.pipeline import Pipeline
from src.preprocessing.denoise import MorphologicalDenoiseStage
from src.preprocessing.grayscale import GrayscaleStage
from src.preprocessing.sauvola import SauvolaBinarizationStage
from src.preprocessing.skew import SkewCorrectionStage, rotate_image
from src.privacy import redact_filtered_template_columns
from src.table import process_table_image
from src.templates import FormTemplate, get_template_ids, load_template

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DOCUMENT_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save copies of farm record images with sensitive filtered template "
            "columns filled black."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input image/PDF paths or directories containing records.",
    )
    parser.add_argument(
        "--template",
        choices=get_template_ids(),
        default=None,
        help="Template to use for every input. Omit with --auto-template.",
    )
    parser.add_argument(
        "--auto-template",
        action="store_true",
        help="Infer the template from each filename.",
    )
    parser.add_argument(
        "--default-template",
        choices=get_template_ids(),
        default=None,
        help="Fallback template when --auto-template cannot infer one.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("redacted_outputs"),
        help="Directory where redacted PNG files will be saved.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input directories.",
    )
    return parser.parse_args()


def detection_pipeline() -> Pipeline:
    return Pipeline(
        [
            GrayscaleStage(),
            SauvolaBinarizationStage(window_size=25, k=0.34),
            MorphologicalDenoiseStage(kernel_size=2),
        ]
    )


def normalize(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def infer_template_id(path: Path) -> str | None:
    name = normalize(path.stem)
    aliases = {
        "boar_room": ("boar", "boarroom", "sample"),
        "breeding": ("breeding",),
        "farrowing_room_4": ("farrowingroom4", "farrowing4"),
        "finisher_room_3": ("finisherroom3", "finisher3"),
        "gestation": ("gestation",),
        "gilt_developer": ("giltdeveloper",),
        "load_out": ("loadout",),
        "metabolism_room_e": ("metabolismroome", "metabolismeroom", "metabolisme"),
        "nursery_room_4": ("nurseryroom4", "nursery4"),
    }
    for template_id, choices in aliases.items():
        if any(choice in name for choice in choices):
            return template_id
    return None


def iter_inputs(paths: list[Path], *, recursive: bool) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            files.extend(
                item
                for item in iterator
                if item.is_file() and item.suffix.lower() in DOCUMENT_SUFFIXES
            )
        elif path.suffix.lower() in DOCUMENT_SUFFIXES:
            files.append(path)
    return sorted(files)


def pages(loaded: DocumentImage | LoadedDocument) -> list[DocumentImage]:
    return loaded.pages if isinstance(loaded, LoadedDocument) else [loaded]


def output_path_for(source: Path, page_number: int, page_count: int, output_dir: Path) -> Path:
    suffix = f"_p{page_number}" if page_count > 1 or source.suffix.lower() == ".pdf" else ""
    return output_dir / f"{source.stem}{suffix}_redacted.png"


def redact_page(
    page: DocumentImage,
    *,
    template: FormTemplate,
    image_name: str,
) -> tuple[object, float]:
    base = detection_pipeline().run(page)
    skew_stage = SkewCorrectionStage()
    skew_angle = skew_stage.estimate_angle(base.image)
    detection_image = rotate_image(base.image, skew_angle, is_binary=True)
    deskewed_source = rotate_image(page.image, skew_angle, is_binary=False)
    table_result = process_table_image(
        detection_image,
        image_name=image_name,
        template=template,
    )
    if not table_result.grid.cells:
        raise ValueError("FarmAI could not find the table in this record.")
    return (
        redact_filtered_template_columns(
            deskewed_source,
            table_result.grid,
            template,
        ),
        skew_angle,
    )


def main() -> None:
    args = parse_args()
    if not args.template and not args.auto_template:
        raise SystemExit("Choose --template or --auto-template.")

    input_paths = iter_inputs(args.inputs, recursive=args.recursive)
    if not input_paths:
        raise SystemExit("No supported image or PDF files were found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in input_paths:
        template_id = args.template
        if args.auto_template:
            template_id = infer_template_id(source) or args.default_template
        if template_id is None:
            print(f"Skipping {source}: no matching template.")
            continue

        template = load_template(template_id)
        loaded = load_document(source)
        doc_pages = pages(loaded)
        for page_number, page in enumerate(doc_pages, start=1):
            image_name = f"{source.stem}_p{page_number}"
            try:
                redacted, _skew_angle = redact_page(
                    page,
                    template=template,
                    image_name=image_name,
                )
            except Exception as exc:
                print(f"Failed {source} page {page_number}: {exc}")
                continue

            out_path = output_path_for(
                source,
                page_number,
                len(doc_pages),
                args.output_dir,
            )
            if not cv2.imwrite(str(out_path), redacted):
                print(f"Failed {source} page {page_number}: could not save {out_path}")
                continue
            print(f"Saved {out_path} using template {template_id}.")


if __name__ == "__main__":
    main()
