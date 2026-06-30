from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.ocr import DEFAULT_OCR_ENGINE, create_ocr_engine, get_ocr_engine_names


ASCII_RAMP = " .:-=+*#%@"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one cropped cell image through a FarmAI OCR backend."
    )
    parser.add_argument("image_path", type=Path, help="Path to one cropped cell image")
    parser.add_argument(
        "--ocr-engine",
        choices=get_ocr_engine_names(),
        default=DEFAULT_OCR_ENGINE,
        help="OCR backend to use",
    )
    parser.add_argument(
        "--trocr-model",
        default="microsoft/trocr-base-handwritten",
        help="Hugging Face model name used when --ocr-engine=trocr-handwritten",
    )
    parser.add_argument("--lang", default="eng", help="Tesseract language")
    parser.add_argument(
        "--psm",
        type=int,
        default=13,
        help="Tesseract page segmentation mode",
    )
    parser.add_argument("--oem", type=int, default=3, help="Tesseract OCR engine mode")
    parser.add_argument(
        "--extra-config",
        default="",
        help="Extra raw Tesseract config string",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for TrOCR, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens for TrOCR",
    )
    parser.add_argument(
        "--save-prepared-image",
        type=Path,
        default=None,
        help="Save the image after OCR-engine preprocessing",
    )
    parser.add_argument(
        "--print-image",
        action="store_true",
        help="Print a small ASCII preview of the loaded image",
    )
    parser.add_argument(
        "--print-prepared-image",
        action="store_true",
        help="Print a small ASCII preview after OCR-engine preprocessing",
    )
    parser.add_argument(
        "--ascii-width",
        type=int,
        default=80,
        help="Maximum character width for ASCII image previews",
    )
    return parser.parse_args(argv)


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("ASCII preview expects a 2D grayscale or 3/4-channel image")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def image_to_ascii(image: np.ndarray, *, width: int = 80) -> str:
    gray = _to_grayscale(image)
    max_width = max(8, int(width))
    scale = min(1.0, max_width / max(1, gray.shape[1]))
    preview_width = max(1, int(round(gray.shape[1] * scale)))
    preview_height = max(1, int(round(gray.shape[0] * scale * 0.5)))
    preview = cv2.resize(
        gray,
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA,
    )

    indices = np.clip(
        np.rint((255 - preview) / 255 * (len(ASCII_RAMP) - 1)),
        0,
        len(ASCII_RAMP) - 1,
    ).astype(np.int32)
    return "\n".join("".join(ASCII_RAMP[index] for index in row) for row in indices)


def _prepared_to_array(prepared: Any) -> np.ndarray:
    if isinstance(prepared, np.ndarray):
        return prepared
    return np.asarray(prepared)


def prepare_image_for_debug(engine: Any, image: np.ndarray) -> np.ndarray:
    prepare = getattr(engine, "_prepare_image", None)
    if prepare is None:
        return image
    return _prepared_to_array(prepare(image))


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def create_engine(args: argparse.Namespace):
    return create_ocr_engine(
        args.ocr_engine,
        model_name=args.trocr_model,
        lang=args.lang,
        psm=args.psm,
        oem=args.oem,
        extra_config=args.extra_config,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    image = load_image(args.image_path)
    engine = create_engine(args)

    if args.print_image:
        print("--- INPUT IMAGE ---")
        print(image_to_ascii(image, width=args.ascii_width))

    prepared = None
    if args.save_prepared_image is not None or args.print_prepared_image:
        prepared = prepare_image_for_debug(engine, image)
        if args.save_prepared_image is not None:
            save_image(args.save_prepared_image, prepared)
            print(f"Saved prepared image to: {args.save_prepared_image}")
        if args.print_prepared_image:
            print("--- PREPARED IMAGE ---")
            print(image_to_ascii(prepared, width=args.ascii_width))

    result = engine.recognize(image)
    confidence = (
        "n/a" if result.confidence is None else f"{float(result.confidence):.2f}"
    )

    print(f"Image: {args.image_path}")
    print(f"OCR engine: {args.ocr_engine}")
    print(f"Confidence: {confidence}")
    print("--- OCR TEXT ---")
    print(result.text)


if __name__ == "__main__":
    main()
