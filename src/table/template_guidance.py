from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.templates import FormTemplate

from .grid_reconstruction import GridCell, GridStructure


@dataclass(frozen=True)
class TemplateGridResult:
    grid: GridStructure
    confidence: float
    repaired_columns: bool
    repaired_rows: bool


def _grid_from_axis_coordinates(
    row_coords: list[int],
    col_coords: list[int],
    image_shape: tuple[int, ...],
) -> GridStructure:
    max_h, max_w = image_shape[:2]
    cells: list[GridCell] = []

    for row_index, (top, bottom) in enumerate(zip(row_coords, row_coords[1:])):
        for col_index, (left, right) in enumerate(zip(col_coords, col_coords[1:])):
            x = max(0, int(left))
            y = max(0, int(top))
            if x >= max_w or y >= max_h:
                continue
            w = min(max(1, int(right) - int(left)), max_w - x)
            h = min(max(1, int(bottom) - int(top)), max_h - y)
            cells.append(GridCell(row=row_index, col=col_index, bbox=(x, y, w, h)))

    return GridStructure(row_coords=row_coords, col_coords=col_coords, cells=cells)


def _proportional_columns(
    left: int,
    right: int,
    widths: tuple[float, ...],
) -> list[int]:
    total = float(sum(widths))
    if total <= 0:
        return [left, right]

    coords = [int(left)]
    running = float(left)
    span = float(right - left)
    for width in widths[:-1]:
        running += span * float(width) / total
        coords.append(int(round(running)))
    coords.append(int(right))

    monotonic: list[int] = []
    for coord in coords:
        if monotonic and coord <= monotonic[-1]:
            coord = monotonic[-1] + 1
        monotonic.append(coord)
    return monotonic


def _regular_rows(rows: list[int]) -> list[int]:
    if len(rows) < 3:
        return rows
    top = int(rows[0])
    bottom = int(rows[-1])
    if bottom <= top:
        return rows
    return [
        int(round(value))
        for value in np.linspace(top, bottom, num=len(rows), endpoint=True)
    ]


def _column_count_score(detected_axis_count: int, expected_column_count: int) -> float:
    expected_axis_count = expected_column_count + 1
    if expected_axis_count <= 0:
        return 0.0
    difference = abs(detected_axis_count - expected_axis_count)
    return max(0.0, 1.0 - difference / expected_axis_count)


def apply_template_to_grid(
    grid: GridStructure,
    template: FormTemplate,
    image_shape: tuple[int, ...],
) -> TemplateGridResult:
    """Repair detected grid axes using known template column proportions."""
    if len(grid.row_coords) < 2 or len(grid.col_coords) < 2:
        return TemplateGridResult(
            grid=grid,
            confidence=0.0,
            repaired_columns=False,
            repaired_rows=False,
        )

    expected_columns = len(template.column_widths)
    left = min(grid.col_coords)
    right = max(grid.col_coords)
    col_coords = _proportional_columns(left, right, template.column_widths)
    row_coords = (
        _regular_rows(grid.row_coords)
        if template.uniform_row_height
        else list(grid.row_coords)
    )

    repaired = _grid_from_axis_coordinates(row_coords, col_coords, image_shape)
    confidence = _column_count_score(len(grid.col_coords), expected_columns)
    if template.uniform_row_height and len(grid.row_coords) >= 3:
        row_gaps = np.diff(grid.row_coords)
        if len(row_gaps) > 0 and float(np.mean(row_gaps)) > 0:
            variation = float(np.std(row_gaps) / np.mean(row_gaps))
            confidence = min(1.0, confidence * 0.8 + max(0.0, 1.0 - variation) * 0.2)

    return TemplateGridResult(
        grid=repaired,
        confidence=float(confidence),
        repaired_columns=True,
        repaired_rows=template.uniform_row_height,
    )
