from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridCell:
    row: int
    col: int
    bbox: tuple[int, int, int, int]


@dataclass
class GridStructure:
    row_coords: list[int]
    col_coords: list[int]
    cells: list[GridCell]


def _cluster_sorted_values(values: list[int], tolerance: int = 10) -> list[int]:
    if not values:
        return []

    sorted_values = sorted(values)
    clusters: list[list[int]] = [[sorted_values[0]]]

    for value in sorted_values[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])

    return [int(round(float(np.mean(cluster)))) for cluster in clusters]


def _extract_axis_coordinates(
    centroids: list[tuple[int, int]], axis: int, tolerance: int = 10
) -> list[int]:
    values = [point[axis] for point in centroids]
    return _cluster_sorted_values(values, tolerance=tolerance)


def reconstruct_grid(
    centroids: list[tuple[int, int]],
    image_shape: tuple[int, int],
    tolerance: int | None = None,
) -> GridStructure:
    """Reconstruct a table grid from intersection centroids.

    Args:
        centroids: List of intersection centroids as (x, y).
        image_shape: Image shape as (height, width).
        tolerance: Clustering tolerance in pixels. If omitted, a conservative
            image-size-scaled tolerance is used.

    Returns:
        GridStructure with clustered row/column coordinates and cell boxes.
    """
    if not centroids:
        return GridStructure(row_coords=[], col_coords=[], cells=[])

    if tolerance is None:
        tolerance = max(10, min(30, min(image_shape[:2]) // 120))

    col_coords = _extract_axis_coordinates(centroids, axis=0, tolerance=tolerance)
    row_coords = _extract_axis_coordinates(centroids, axis=1, tolerance=tolerance)

    cells: list[GridCell] = []
    for row_index in range(len(row_coords) - 1):
        top = row_coords[row_index]
        bottom = row_coords[row_index + 1]
        for col_index in range(len(col_coords) - 1):
            left = col_coords[col_index]
            right = col_coords[col_index + 1]

            x = max(0, left)
            y = max(0, top)
            w = max(1, right - left)
            h = max(1, bottom - top)

            # Clip to image bounds.
            max_h, max_w = image_shape[:2]
            if x >= max_w or y >= max_h:
                continue
            w = min(w, max_w - x)
            h = min(h, max_h - y)
            cells.append(GridCell(row=row_index, col=col_index, bbox=(x, y, w, h)))

    return GridStructure(row_coords=row_coords, col_coords=col_coords, cells=cells)
