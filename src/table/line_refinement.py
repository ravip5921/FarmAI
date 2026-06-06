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
    col_segment_support: list[int]
    col_endpoint_support: list[int]
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

    horizontal_hits = (
        cv2.dilate(
            horizontal_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1)),
        )
        > 0
    )
    vertical_hits = (
        cv2.dilate(
            vertical_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1)),
        )
        > 0
    )

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
        supported = [
            index
            for index, support in enumerate(row_support)
            if support >= required_support
        ]
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


def _filter_column_candidates(
    cols: list[int], col_support: list[int], row_count: int
) -> list[int]:
    if len(cols) < 2 or not col_support:
        return cols

    min_support = max(2, min(5, row_count // 6))
    filtered = [
        col for col, support in zip(cols, col_support) if support >= min_support
    ]
    if len(filtered) < 2:
        return cols
    return filtered


def _column_segment_support(
    vertical_mask: np.ndarray,
    rows: list[int],
    cols: list[int],
    *,
    radius: int,
) -> list[int]:
    """Count row bands where a column has enough vertical ink to be a rule."""
    if len(rows) < 2 or not cols:
        return [0 for _ in cols]

    height, width = vertical_mask.shape[:2]
    foreground = vertical_mask > 0
    support = [0 for _ in cols]

    for col_index, x in enumerate(cols):
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        if x0 >= x1:
            continue

        for top, bottom in zip(rows, rows[1:]):
            y0 = max(0, min(top, bottom))
            y1 = min(height, max(top, bottom) + 1)
            band_height = y1 - y0
            if band_height <= 0:
                continue

            margin = min(max(0, band_height // 3), radius)
            y0 += margin
            y1 -= margin
            band_height = y1 - y0
            if band_height <= 0:
                continue

            rows_with_ink = int(
                np.count_nonzero(np.any(foreground[y0:y1, x0:x1], axis=1))
            )
            min_rows_with_ink = max(2, int(round(band_height * 0.30)))
            if rows_with_ink >= min_rows_with_ink:
                support[col_index] += 1

    return support


def _filter_columns_by_table_support(
    cols: list[int],
    crossing_support: list[int],
    segment_support: list[int],
    endpoint_support: list[int],
    row_count: int,
) -> list[int]:
    """Keep columns that behave like borders across the reconstructed row grid."""
    if len(cols) < 2:
        return cols

    filtered = [
        col
        for col, crossings, segments, endpoints in zip(
            cols, crossing_support, segment_support, endpoint_support
        )
        if crossings > 0 or segments > 0 or endpoints > 0
    ]
    if len(filtered) >= 2:
        return filtered

    return _filter_column_candidates(cols, crossing_support, row_count)


def _dedupe_close_columns(
    cols: list[int],
    crossing_support: list[int],
    segment_support: list[int],
    endpoint_support: list[int],
    *,
    min_separation: int,
) -> list[int]:
    """Collapse close column candidates, keeping the best-supported axis."""
    if len(cols) < 2 or min_separation <= 1:
        return cols

    sorted_items = sorted(
        zip(cols, crossing_support, segment_support, endpoint_support),
        key=lambda item: item[0],
    )
    clusters: list[list[tuple[int, int, int, int]]] = [[sorted_items[0]]]
    for item in sorted_items[1:]:
        if item[0] - clusters[-1][-1][0] < min_separation:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    deduped: list[int] = []
    for cluster in clusters:
        if len(cluster) == 1:
            deduped.append(cluster[0][0])
            continue

        best = max(
            cluster,
            key=lambda item: (
                item[1] + item[2] * 2,
                item[1],
                item[2],
                -item[3],
            ),
        )
        deduped.append(best[0])

    return deduped


def _foreground_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    indices = np.flatnonzero(values)
    if indices.size == 0:
        return runs

    start = int(indices[0])
    previous = start
    for index in indices[1:]:
        value = int(index)
        if value == previous + 1:
            previous = value
            continue
        runs.append((start, previous))
        start = value
        previous = value

    runs.append((start, previous))
    return runs


def _horizontal_endpoint_candidates(
    horizontal_mask: np.ndarray,
    rows: list[int],
    *,
    radius: int,
    min_run_length: int,
    tolerance: int,
) -> tuple[list[int], list[int]]:
    """Recover column axes from repeated horizontal segment starts and ends."""
    if not rows:
        return [], []

    height, width = horizontal_mask.shape[:2]
    foreground = horizontal_mask > 0
    endpoints: list[int] = []

    for y in rows:
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        if y0 >= y1:
            continue

        col_has_ink = np.any(foreground[y0:y1, :], axis=0)
        runs = _foreground_runs(col_has_ink)
        for start, end in runs:
            if end - start + 1 >= min_run_length:
                endpoints.extend([start, end])

    candidates = _merge_axis_candidates(endpoints, tolerance=tolerance)
    support = _axis_occurrence_support(
        endpoints,
        candidates,
        tolerance=tolerance,
    )
    return candidates, support


def _axis_occurrence_support(
    observations: list[int],
    candidates: list[int],
    *,
    tolerance: int,
) -> list[int]:
    if not candidates:
        return []

    support = [0 for _ in candidates]
    for observation in observations:
        distances = [abs(candidate - observation) for candidate in candidates]
        best_index = int(np.argmin(np.asarray(distances)))
        if distances[best_index] <= tolerance:
            support[best_index] += 1
    return support


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

    endpoint_tolerance = max(1, min(8, width // 220))
    endpoint_candidates, endpoint_candidate_support = _horizontal_endpoint_candidates(
        horizontal_mask,
        row_coords,
        radius=max(radius, 2),
        min_run_length=max(8, width // 90),
        tolerance=endpoint_tolerance,
    )
    if endpoint_candidates:
        col_candidates = _merge_axis_candidates(
            col_candidates + endpoint_candidates,
            tolerance=endpoint_tolerance,
        )

    _, refined_col_support = _crossing_support(
        horizontal_mask,
        vertical_mask,
        row_coords,
        col_candidates,
        radius=radius,
    )
    segment_radius = max(radius, 2)
    col_segment_support = _column_segment_support(
        vertical_mask,
        row_coords,
        col_candidates,
        radius=segment_radius,
    )
    col_endpoint_support = [
        sum(
            support
            for candidate, support in zip(
                endpoint_candidates, endpoint_candidate_support
            )
            if abs(candidate - col) <= endpoint_tolerance
        )
        for col in col_candidates
    ]
    col_coords = _filter_columns_by_table_support(
        col_candidates,
        refined_col_support,
        col_segment_support,
        col_endpoint_support,
        len(row_coords),
    )
    if len(col_coords) >= 3:
        min_column_separation = max(3, min(18, width // 250))
        filtered_support = [
            support
            for col, support in zip(col_candidates, refined_col_support)
            if col in col_coords
        ]
        filtered_segment_support = [
            support
            for col, support in zip(col_candidates, col_segment_support)
            if col in col_coords
        ]
        filtered_endpoint_support = [
            support
            for col, support in zip(col_candidates, col_endpoint_support)
            if col in col_coords
        ]
        col_coords = _dedupe_close_columns(
            col_coords,
            filtered_support,
            filtered_segment_support,
            filtered_endpoint_support,
            min_separation=min_column_separation,
        )

    grid = _grid_from_axis_coordinates(row_coords, col_coords, image_shape)
    return GridRefinementResult(
        grid=grid,
        row_candidates=row_candidates,
        col_candidates=col_candidates,
        row_support=row_support,
        col_support=refined_col_support,
        col_segment_support=col_segment_support,
        col_endpoint_support=col_endpoint_support,
        estimated_row_spacing=spacing,
    )
