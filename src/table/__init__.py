from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.image import DocumentImage
from src.core.visualization import save_debug

from .grid_reconstruction import GridCell, GridStructure, reconstruct_grid
from .intersections import (
    IntersectionDetectionStage,
    IntersectionResult,
    detect_intersections,
)
from .line_detection import LineDetectionResult, LineDetectionStage, detect_lines
from .line_refinement import GridRefinementResult, refine_grid_with_projection_profiles


@dataclass
class TablePipelineResult:
    line_detection: LineDetectionResult
    intersections: IntersectionResult
    grid: GridStructure
    grid_refinement: GridRefinementResult | None = None


def _extract_bitmap(bitmap: Any) -> np.ndarray:
    if isinstance(bitmap, DocumentImage):
        return bitmap.image
    return np.asarray(bitmap)


def _resolve_stem(image_name: str | Path | None) -> str:
    if image_name is None:
        return "table"
    return Path(str(image_name)).stem or "table"


def _save_preview(debug_dir: Path, prefix: str, stem: str, image: np.ndarray) -> None:
    save_debug(
        debug_dir / f"{prefix}_{stem}.png", prefix.replace("_", " ").title(), image
    )


def render_grid_structure(
    grid: GridStructure, image_shape: tuple[int, int]
) -> np.ndarray:
    """Render reconstructed table cells as a simple black-on-white table map."""
    canvas = np.full(image_shape[:2], 255, dtype=np.uint8)
    for cell in grid.cells:
        x, y, w, h = cell.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color=0, thickness=1)
    return canvas


def render_grid_overlay(
    image: np.ndarray,
    grid: GridStructure,
    *,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    alpha: float = 0.75,
) -> np.ndarray:
    """Draw the reconstructed table grid over an image for visual inspection."""
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        base = image.copy()
    else:
        raise ValueError("render_grid_overlay expects a 2D or 3D image")

    overlay = base.copy()
    max_h, max_w = base.shape[:2]
    line_thickness = max(1, int(thickness))

    for x in grid.col_coords:
        x_coord = int(np.clip(x, 0, max_w - 1))
        cv2.line(
            overlay,
            (x_coord, 0),
            (x_coord, max_h - 1),
            color=color,
            thickness=line_thickness,
        )
    for y in grid.row_coords:
        y_coord = int(np.clip(y, 0, max_h - 1))
        cv2.line(
            overlay,
            (0, y_coord),
            (max_w - 1, y_coord),
            color=color,
            thickness=line_thickness,
        )

    blend = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(overlay, blend, base, 1.0 - blend, 0.0)


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
        line_preview = np.hstack(
            [line_detection.horizontal_mask, line_detection.vertical_mask]
        )
        _save_preview(debug_dir, "line_detection", stem, line_preview)
        _save_preview(
            debug_dir, "line_detection_horizontal", stem, line_detection.horizontal_mask
        )
        _save_preview(
            debug_dir, "line_detection_vertical", stem, line_detection.vertical_mask
        )

    intersections = detect_intersections(
        line_detection.horizontal_mask,
        line_detection.vertical_mask,
    )
    if debug_dir is not None and save_intersections:
        debug_dir.mkdir(parents=True, exist_ok=True)
        _save_preview(debug_dir, "intersection", stem, intersections.intersection_mask)

    raw_grid = reconstruct_grid(intersections.centroids, image.shape)
    grid_refinement = refine_grid_with_projection_profiles(
        line_detection.horizontal_mask,
        line_detection.vertical_mask,
        image.shape,
        intersection_centroids=intersections.centroids,
    )
    grid = grid_refinement.grid if grid_refinement.grid.cells else raw_grid
    if debug_dir is not None and save_intersections:
        _save_preview(
            debug_dir, "grid_projection", stem, render_grid_structure(grid, image.shape)
        )

    return TablePipelineResult(
        line_detection=line_detection,
        intersections=intersections,
        grid=grid,
        grid_refinement=grid_refinement,
    )


__all__ = [
    "GridCell",
    "GridRefinementResult",
    "GridStructure",
    "IntersectionDetectionStage",
    "IntersectionResult",
    "LineDetectionResult",
    "LineDetectionStage",
    "TablePipelineResult",
    "detect_intersections",
    "detect_lines",
    "refine_grid_with_projection_profiles",
    "process_table_image",
    "render_grid_overlay",
    "render_grid_structure",
    "reconstruct_grid",
]
