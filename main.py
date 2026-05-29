from __future__ import annotations

from pathlib import Path

import cv2

from src.core.image import DocumentImage
from src.core.pipeline import Pipeline
from src.core.visualization import save_debug, show
from src.preprocessing.denoise import MorphologicalDenoiseStage
from src.preprocessing.grayscale import GrayscaleStage
from src.preprocessing.sauvola import SauvolaBinarizationStage
from src.preprocessing.skew import SkewCorrectionStage


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
) -> DocumentImage:
	"""Process a single image through the pipeline."""
	image = cv2.imread(str(image_path))
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")

	doc = DocumentImage(image)

	# Run the pipeline: output of stage N becomes input to stage N+1
	if debug_dir:
		debug_dir.mkdir(parents=True, exist_ok=True)
		current = doc
		for stage_index, stage in enumerate(pipeline.stages, start=1):
			current = stage.process(current)
			stage_name = stage.__class__.__name__
			filename = debug_dir / f"stage_{stage_index:02d}_{stage_name}.png"
			save_debug(filename, stage_name, current)
		return current
	else:
		# If no debug, just run the whole pipeline at once
		return pipeline.run(doc)


def main() -> None:
	root = Path(__file__).resolve().parent
	sample_path = root / "examples" / "Boar - barn - flash.jpg"
	debug_dir = root / "debug_outputs"

	# Build pipeline with default parameters
	pipeline = build_pipeline(window_size=25, k=0.34, denoise_kernel=2)

	# Process single image
	result = process_image(sample_path, pipeline, debug_dir=debug_dir)

	show("Final Output", result)


if __name__ == "__main__":
	main()
