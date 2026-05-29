from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2

from src.core.image import DocumentImage
from src.core.pipeline import Pipeline
from src.core.visualization import save_debug, show
from src.preprocessing.denoise import MorphologicalDenoiseStage
from src.preprocessing.grayscale import GrayscaleStage
from src.preprocessing.sauvola import SauvolaBinarizationStage
from src.preprocessing.skew import SkewCorrectionStage
from src.table import process_table_image, render_grid_structure


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run Farm AI preprocessing and table extraction on an image.")
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
    image_path: str | Path,
    pipeline: Pipeline,
    debug_dir: Path | None = None,
	image_name: str | Path | None = None,
) -> DocumentImage:
	"""Process a single image through the pipeline."""
	image = cv2.imread(str(image_path))
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")

	doc = DocumentImage(image)
	stem = Path(str(image_name or image_path)).stem

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
):
	"""Run table-structure extraction after preprocessing."""
	return process_table_image(
		bitmap,
		image_name=image_name,
		debug_dir=debug_dir,
		save_line_detection=save_line_detection,
		save_intersections=save_intersections,
	)


def main() -> None:
	args = parse_args()
	image_path = args.image_path
	debug_dir = None if args.no_debug else args.debug_dir
	image_name = image_path.stem

	# Build pipeline with default parameters
	pipeline = build_pipeline(window_size=25, k=0.34, denoise_kernel=2)

	# Process single image
	result = process_image(image_path, pipeline, debug_dir=debug_dir, image_name=image_name)
	table_result = process_table(
		result.image,
		image_name=image_name,
		debug_dir=debug_dir,
		save_line_detection=args.save_line_detection,
		save_intersections=args.save_intersections,
	)
	table_image = render_grid_structure(table_result.grid, result.image.shape)

	show("Final Output", result)
	show("Detected Table", table_image)


if __name__ == "__main__":
	main()
