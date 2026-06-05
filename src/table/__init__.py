from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.image import DocumentImage
from src.core.visualization import save_debug

from .grid_reconstruction import GridCell, GridStructure, reconstruct_grid
from .intersections import IntersectionDetectionStage, IntersectionResult, detect_intersections
from .line_detection import LineDetectionResult, LineDetectionStage, detect_lines


@dataclass
class TablePipelineResult:
	line_detection: LineDetectionResult
	intersections: IntersectionResult
	grid: GridStructure


def _extract_bitmap(bitmap: Any) -> np.ndarray:
	if isinstance(bitmap, DocumentImage):
		return bitmap.image
	return np.asarray(bitmap)


def _resolve_stem(image_name: str | Path | None) -> str:
	if image_name is None:
		return "table"
	return Path(str(image_name)).stem or "table"


def _save_preview(debug_dir: Path, prefix: str, stem: str, image: np.ndarray) -> None:
	save_debug(debug_dir / f"{prefix}_{stem}.png", prefix.replace("_", " ").title(), image)


def render_grid_structure(grid: GridStructure, image_shape: tuple[int, int]) -> np.ndarray:
	"""Render reconstructed table cells as a simple black-on-white table map."""
	canvas = np.full(image_shape[:2], 255, dtype=np.uint8)
	for cell in grid.cells:
		x, y, w, h = cell.bbox
		cv2.rectangle(canvas, (x, y), (x + w, y + h), color=0, thickness=1)
	return canvas


def process_table_image(
	bitmap: Any,
	*,
	image_name: str | Path | None = None,
	debug_dir: Path | None = None,
	save_line_detection: bool = False,
	save_intersections: bool = False,
) -> TablePipelineResult:
	"""Run table-structure extraction on a binary bitmap image.

	The input is expected to be the binarized output from preprocessing. The
	function extracts line masks, finds intersections, reconstructs the grid,
	and optionally saves intermediate previews with names prefixed by the step
	name and the source image stem.
	"""
	image = _extract_bitmap(bitmap)
	if image.ndim != 2:
		raise ValueError("process_table_image expects a 2D binary bitmap image")

	stem = _resolve_stem(image_name)
	line_detection = detect_lines(image)
	if debug_dir is not None and save_line_detection:
		debug_dir.mkdir(parents=True, exist_ok=True)
		line_preview = np.hstack([line_detection.horizontal_mask, line_detection.vertical_mask])
		_save_preview(debug_dir, "line_detection", stem, line_preview)
		_save_preview(debug_dir, "line_detection_horizontal", stem, line_detection.horizontal_mask)
		_save_preview(debug_dir, "line_detection_vertical", stem, line_detection.vertical_mask)


	intersections = detect_intersections(
		line_detection.horizontal_mask,
		line_detection.vertical_mask,
	)
	if debug_dir is not None and save_intersections:
		debug_dir.mkdir(parents=True, exist_ok=True)
		_save_preview(debug_dir, "intersection", stem, intersections.intersection_mask)

	grid = reconstruct_grid(intersections.centroids, image.shape)
	return TablePipelineResult(
		line_detection=line_detection,
		intersections=intersections,
		grid=grid,
	)


__all__ = [
	"GridCell",
	"GridStructure",
	"IntersectionDetectionStage",
	"IntersectionResult",
	"LineDetectionResult",
	"LineDetectionStage",
	"TablePipelineResult",
	"detect_intersections",
	"detect_lines",
	"process_table_image",
	"render_grid_structure",
	"reconstruct_grid",
]
