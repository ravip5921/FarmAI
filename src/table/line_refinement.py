from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.analysis.projection_profiles import (
    find_projection_peaks,
    peak_positions,
    projection_profile,
)

from .grid_reconstruction import GridCell, GridStructure


@dataclass
class GridRefinementResult:
    grid: GridStructure
    row_candidates: list[int]
    col_candidates: list[int]
    row_support: list[int]
    col_support: list[int]
    estimated_row_spacing: int


def _grid_from_axis_coordinates(
    row_coords: list[int],
    col_coords: list[int],
    image_shape: tuple[int, int],
) -> GridStructure:
    cells: list[GridCell] = []
    max_h, max_w = image_shape[:2]

    for row_index in range(len(row_coords) - 1):
        top = row_coords[row_index]
        bottom = row_coords[row_index + 1]
        for col_index in range(len(col_coords) - 1):
            left = col_coords[col_index]
            right = col_coords[col_index + 1]

            x = max(0, left)
            y = max(0, top)
            if x >= max_w or y >= max_h:
                continue

            w = min(max(1, right - left), max_w - x)
            h = min(max(1, bottom - top), max_h - y)
            cells.append(GridCell(row=row_index, col=col_index, bbox=(x, y, w, h)))

    return GridStructure(row_coords=row_coords, col_coords=col_coords, cells=cells)


def _crossing_support(
    horizontal_mask: np.ndarray,
    vertical_mask: np.ndarray,
    rows: list[int],
    cols: list[int],
    *,
    radius: int,
) -> tuple[list[int], list[int]]:
    if not rows or not cols:
        return [0 for _ in rows], [0 for _ in cols]

    horizontal_hits = cv2.dilate(
        horizontal_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1)),
    ) > 0
    vertical_hits = cv2.dilate(
        vertical_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1)),
    ) > 0

    height, width = horizontal_mask.shape[:2]
    row_support = [0 for _ in rows]
    col_support = [0 for _ in cols]

    for row_index, y in enumerate(rows):
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        for col_index, x in enumerate(cols):
            x0 = max(0, x - radius)
            x1 = min(width, x + radius + 1)
            if np.any(horizontal_hits[y0:y1, x0:x1] & vertical_hits[y0:y1, x0:x1]):
                row_support[row_index] += 1
                col_support[col_index] += 1

    return row_support, col_support


def _estimate_regular_spacing(coords: list[int], axis_length: int) -> int:
    if len(coords) < 3:
        return 0

    min_spacing = max(3, axis_length // 120)
    max_spacing = max(min_spacing + 1, axis_length // 12)
    gaps = np.diff(sorted(coords))
    usable_gaps = [int(gap) for gap in gaps if min_spacing <= int(gap) <= max_spacing]
    if not usable_gaps:
        return 0

    return max(1, int(round(float(np.median(np.asarray(usable_gaps))))))


def _merge_close_positions(coords: list[int], spacing: int) -> list[int]:
    if not coords:
        return []
    if spacing <= 0:
        return sorted(coords)

    tolerance = max(2, int(round(spacing * 0.45)))
    merged: list[int] = []
    group: list[int] = []
    for coord in sorted(coords):
        if not group or coord - group[-1] <= tolerance:
            group.append(coord)
        else:
            merged.append(int(round(float(np.mean(group)))))
            group = [coord]

    if group:
        merged.append(int(round(float(np.mean(group)))))
    return merged


def _merge_axis_candidates(coords: list[int], tolerance: int) -> list[int]:
    if not coords:
        return []

    sorted_coords = sorted(coords)
    clusters: list[list[int]] = [[sorted_coords[0]]]
    for coord in sorted_coords[1:]:
        if coord - clusters[-1][-1] <= tolerance:
            clusters[-1].append(coord)
        else:
            clusters.append([coord])

    return [int(round(float(np.mean(cluster)))) for cluster in clusters]


def _fill_missing_regular_positions(coords: list[int], spacing: int) -> list[int]:
    if len(coords) < 2 or spacing <= 0:
        return sorted(coords)

    filled: list[int] = []
    sorted_coords = sorted(coords)
    for left, right in zip(sorted_coords, sorted_coords[1:]):
        filled.append(left)
        gap = right - left
        steps = int(round(gap / spacing))
        if steps > 1 and gap <= spacing * 3.25:
            for step in range(1, steps):
                filled.append(int(round(left + step * gap / steps)))

    filled.append(sorted_coords[-1])
    return filled


def _trim_row_candidates(
    rows: list[int],
    row_support: list[int],
    axis_length: int,
) -> tuple[list[int], int]:
    if len(rows) < 2:
        return rows, 0

    start_index = 0
    for required_support in (2, 1):
        supported = [index for index, support in enumerate(row_support) if support >= required_support]
        if supported:
            start_index = supported[0]
            break

    table_rows = rows[start_index:]
    spacing = _estimate_regular_spacing(table_rows, axis_length)
    if spacing > 0:
        end_index = len(table_rows)
        for index, gap in enumerate(np.diff(table_rows)):
            if int(gap) > spacing * 3.25:
                end_index = index + 1
                break
        table_rows = table_rows[:end_index]
        table_rows = _merge_close_positions(table_rows, spacing)
        table_rows = _fill_missing_regular_positions(table_rows, spacing)

    if len(table_rows) < 2:
        return rows, spacing
    return table_rows, spacing


def _filter_column_candidates(cols: list[int], col_support: list[int], row_count: int) -> list[int]:
    if len(cols) < 2 or not col_support:
        return cols

    min_support = max(2, min(5, row_count // 6))
    filtered = [col for col, support in zip(cols, col_support) if support >= min_support]
    if len(filtered) < 2:
        return cols
    return filtered


def refine_grid_with_projection_profiles(
    horizontal_mask: np.ndarray,
    vertical_mask: np.ndarray,
    image_shape: tuple[int, int],
    intersection_centroids: list[tuple[int, int]] | None = None,
) -> GridRefinementResult:
    """Snap grid axes to projection-profile peaks from line masks."""
    if horizontal_mask.ndim != 2 or vertical_mask.ndim != 2:
        raise ValueError("refine_grid_with_projection_profiles expects 2D masks")

    height, width = image_shape[:2]
    row_profile = projection_profile(horizontal_mask, "horizontal")
    col_profile = projection_profile(vertical_mask, "vertical")

    row_candidates = peak_positions(
        find_projection_peaks(
            row_profile,
            min_peak_ratio=0.1,
            min_peak_value=max(3, width * 0.02),
            max_gap=max(2, height // 900),
        )
    )
    col_candidates = peak_positions(
        find_projection_peaks(
            col_profile,
            min_peak_ratio=0.1,
            min_peak_value=max(3, height * 0.02),
            max_gap=max(2, width // 900),
        )
    )
    if intersection_centroids:
        centroid_cols = [point[0] for point in intersection_centroids]
        col_candidates = _merge_axis_candidates(
            col_candidates + centroid_cols,
            tolerance=max(5, width // 200),
        )

    radius = max(1, min(10, min(height, width) // 250))
    row_support, col_support = _crossing_support(
        horizontal_mask,
        vertical_mask,
        row_candidates,
        col_candidates,
        radius=radius,
    )
    row_coords, spacing = _trim_row_candidates(row_candidates, row_support, height)

    _, refined_col_support = _crossing_support(
        horizontal_mask,
        vertical_mask,
        row_coords,
        col_candidates,
        radius=radius,
    )
    col_coords = _filter_column_candidates(col_candidates, refined_col_support, len(row_coords))

    grid = _grid_from_axis_coordinates(row_coords, col_coords, image_shape)
    return GridRefinementResult(
        grid=grid,
        row_candidates=row_candidates,
        col_candidates=col_candidates,
        row_support=row_support,
        col_support=refined_col_support,
        estimated_row_spacing=spacing,
    )
