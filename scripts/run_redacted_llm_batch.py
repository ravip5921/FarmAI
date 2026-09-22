from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.application import ProcessingProgress, ProcessingSettings, process_document
from src.templates import get_template_ids
from user_interface.backend.services.artifact_store import result_to_csv

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FarmAI llm-vision OCR on redacted example images and save one "
            "CSV per image for accuracy analysis."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("examples") / "redacted",
        help="Directory containing redacted images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out") / "redacted_llm_results",
        help="Directory where CSV outputs will be saved.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Also save the full structured OCR result JSON for each image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun images even when the output CSV already exists.",
    )
    parser.add_argument(
        "--ocr-padding",
        type=int,
        default=2,
        help="Pixels to trim from each detected cell before OCR.",
    )
    parser.add_argument(
        "--ocr-context-padding",
        type=int,
        default=0,
        help="Pixels to expand each cell crop before OCR.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional filename substrings to process, case-insensitive.",
    )
    return parser.parse_args()


def normalize(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def infer_template_id(path: Path) -> str | None:
    name = normalize(path.stem)
    aliases = {
        "boar_room": ("boar", "boarroom"),
        "breeding": ("breeding",),
        "farrowing_room_4": ("farrowingroom4", "farrowing4"),
        "finisher_room_3": ("finisherroom3", "finisher3"),
        "gestation": ("gestation",),
        "gilt_developer": ("giltdeveloper",),
        "load_out": ("loadout",),
        "metabolism_room_e": ("metabolismroome", "metabolisme"),
        "nursery_room_4": ("nurseryroom4", "nursery4"),
    }
    available = set(get_template_ids())
    for template_id, choices in aliases.items():
        if template_id in available and any(choice in name for choice in choices):
            return template_id
    return None


def iter_images(input_dir: Path, only: list[str] | None) -> list[Path]:
    filters = [item.casefold() for item in only or []]
    images = [
        path
        for path in input_dir.glob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    ]
    if filters:
        images = [
            path
            for path in images
            if any(filter_value in path.name.casefold() for filter_value in filters)
        ]
    return sorted(images)


def result_to_dict(result) -> dict:
    return {
        "filename": result.filename,
        "template_id": result.template_id,
        "template_name": result.template_name,
        "ocr_engine": result.ocr_engine,
        "warning_count": result.warning_count,
        "metrics": None,
        "pages": [
            page.to_dict(source_url="", overlay_url="")
            for page in result.pages
        ],
    }


def output_stem(path: Path) -> str:
    stem = path.stem
    return stem.removesuffix("_redacted")


def main() -> None:
    args = parse_args()
    images = iter_images(args.input_dir, args.only)
    if not images:
        raise SystemExit(f"No redacted images found in {args.input_dir}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[Path, str]] = []

    for index, image_path in enumerate(images, start=1):
        template_id = infer_template_id(image_path)
        if template_id is None:
            failures.append((image_path, "No matching template."))
            print(f"[{index}/{len(images)}] Skipping {image_path.name}: no template.")
            continue

        stem = output_stem(image_path)
        csv_path = args.output_dir / f"{stem}_{template_id}_llm.csv"
        json_path = args.output_dir / f"{stem}_{template_id}_llm.json"
        if csv_path.exists() and not args.overwrite:
            print(f"[{index}/{len(images)}] Skipping {image_path.name}: CSV exists.")
            continue

        print(f"[{index}/{len(images)}] Processing {image_path.name} ({template_id})")

        def progress(value: ProcessingProgress) -> None:
            if value.stage == "recognizing_cells":
                print(
                    f"  {value.message}",
                    end="\r",
                    flush=True,
                )

        try:
            result = process_document(
                image_path,
                settings=ProcessingSettings(
                    template_id=template_id,
                    ocr_engine="llm-vision",
                    ocr_padding=args.ocr_padding,
                    ocr_context_padding=args.ocr_context_padding,
                ),
                progress_callback=progress,
            )
            payload = result_to_dict(result)
            csv_path.write_text(result_to_csv(payload), encoding="utf-8")
            if args.save_json:
                json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\n  Saved {csv_path}")
        except Exception as exc:
            failures.append((image_path, str(exc)))
            print(f"\n  Failed {image_path.name}: {exc}")

    if failures:
        failure_path = args.output_dir / "failures.json"
        failure_path.write_text(
            json.dumps(
                [
                    {"image": str(path), "error": error}
                    for path, error in failures
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nCompleted with {len(failures)} failure(s). See {failure_path}.")
    else:
        print("\nCompleted without failures.")


if __name__ == "__main__":
    main()
