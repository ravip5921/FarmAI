from __future__ import annotations

import cv2
import numpy as np

from src.table.grid_reconstruction import GridStructure
from src.templates import FormTemplate


def _is_divider_column(key: str) -> bool:
    return key.casefold().startswith("divider")


def redacted_template_column_indices(
    template: FormTemplate,
    *,
    extra_filtered_columns: set[str] | None = None,
) -> set[int]:
    """Return filtered template columns that should be blacked out.

    Divider columns are omitted because they are structural separators rather
    than data-bearing columns. Other filtered columns, including key="empty",
    are covered.
    """
    filtered = set(template.filtered_column_indices)
    if extra_filtered_columns:
        filtered |= template.indices_for_column_names(extra_filtered_columns)

    return {
        column.index
        for column in template.columns
        if column.index in filtered and not _is_divider_column(column.key)
    }


def _column_border_lines(
    image: np.ndarray,
    grid: GridStructure,
    top: int,
    bottom: int,
) -> dict[int, tuple[float, float]]:
    """Match long printed rules to template edges on slightly slanted scans.

    Return each matched edge's x positions at the top and bottom of the table.
    Short strokes in handwritten initials must not determine the mask edges.
    Unmatched edges keep their existing grid coordinates.
    """
    height = bottom - top
    if height < 8:
        return {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, foreground = cv2.threshold(
        gray[top:bottom], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    row_height = max(3, int(np.median(np.diff(grid.row_coords))))
    vertical = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, row_height * 2 // 3))),
    )
    min_length = max(8, height // 3)
    lines = cv2.HoughLinesP(
        vertical,
        1,
        np.pi / 720,
        threshold=max(10, height // 16),
        minLineLength=min_length,
        maxLineGap=row_height,
    )
    if lines is None:
        return {}

    candidates: list[tuple[float, float, int]] = []
    for x0, y0, x1, y1 in lines[:, 0]:
        dy = int(y1) - int(y0)
        dx = int(x1) - int(x0)
        if abs(dy) < min_length or abs(dx) > abs(dy) * 0.05:
            continue
        slope = dx / dy
        start = float(x0) - slope * float(y0)
        candidates.append((start, start + slope * height, abs(dy)))

    tolerance = max(3, (max(grid.col_coords) - min(grid.col_coords)) * 0.015)
    matched: dict[int, tuple[float, float]] = {}
    for index, x in enumerate(grid.col_coords):
        nearby = [
            line for line in candidates if abs((line[0] + line[1]) / 2 - x) <= tolerance
        ]
        if nearby:
            # Prefer the closest rule, using its length to break ties.
            start, end, _ = min(
                nearby,
                key=lambda line: (abs((line[0] + line[1]) / 2 - x), -line[2]),
            )
            matched[index] = (start, end)
    return matched


def _trace_column_border(
    image: np.ndarray,
    x: int,
    top: int,
    bottom: int,
    radius: int,
) -> np.ndarray | None:
    """Find a continuous printed border, allowing slow curvature in a photo.

    A path can move at most one pixel per row. This favors a sustained rule
    over the short strokes of handwriting beside it.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    left, right = max(0, x - radius), min(gray.shape[1], x + radius + 1)
    if right <= left or bottom - top < 8:
        return None
    _, ink = cv2.threshold(
        gray[top:bottom, left:right], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    cost = (ink == 0).astype(np.float64)
    cost += np.abs(np.arange(left, right) - x)[None, :] * (0.005 / max(1, radius))
    previous = cost[0].copy()
    back = np.zeros(cost.shape, dtype=np.int8)
    for row in range(1, len(cost)):
        choices = np.stack(
            (
                np.r_[np.inf, previous[:-1] + 0.15],
                previous,
                np.r_[previous[1:] + 0.15, np.inf],
            )
        )
        best = np.argmin(choices, axis=0)
        back[row] = best - 1
        previous = cost[row] + np.min(choices, axis=0)
    path: np.ndarray = np.empty(len(cost), dtype=np.int32)
    path[-1] = int(np.argmin(previous))
    for row in range(len(path) - 1, 0, -1):
        path[row - 1] = path[row] + back[row, path[row]]
    if np.mean(ink[np.arange(len(path)), path] > 0) < 0.55:
        return None
    return path + left


def redact_filtered_template_columns(
    image: np.ndarray,
    grid: GridStructure,
    template: FormTemplate,
    *,
    extra_filtered_columns: set[str] | None = None,
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Fill sensitive filtered template columns with a solid color."""
    redacted = image.copy()
    column_indices = redacted_template_column_indices(
        template,
        extra_filtered_columns=extra_filtered_columns,
    )
    if not column_indices or len(grid.row_coords) < 2 or len(grid.col_coords) < 2:
        return redacted

    max_h, max_w = redacted.shape[:2]
    top = max(0, min(grid.row_coords))
    bottom = min(max_h, max(grid.row_coords))
    if bottom <= top:
        return redacted
    border_lines = (
        _column_border_lines(image, grid, top, bottom)
        if template.redaction_follow_column_borders
        else {}
    )
    column_ranges: list[tuple[int, int]] = []
    divider_indices = {
        column.index
        for column in template.columns
        if column.filter_out and _is_divider_column(column.key)
    }
    for index in sorted(column_indices):
        if (
            template.redaction_follow_column_borders
            and column_ranges
            and all(
                gap in divider_indices for gap in range(column_ranges[-1][1], index)
            )
        ):
            # Initials often extend into the blank separator between the two
            # sensitive fields. Cover that separator along with both fields.
            column_ranges[-1] = (column_ranges[-1][0], index + 1)
        else:
            column_ranges.append((index, index + 1))

    for column_index, end_index in column_ranges:
        if end_index >= len(grid.col_coords):
            continue
        left = max(0, int(grid.col_coords[column_index]))
        right = min(max_w, int(grid.col_coords[end_index]))
        if right <= left:
            continue
        if template.redaction_follow_curved_borders:
            radius = max(
                5, round((max(grid.col_coords) - min(grid.col_coords)) * 0.035)
            )
            # Stay within the adjacent field's scale so a dark page edge or
            # another printed column cannot win over the intended border.
            left_radius = min(
                radius, max(3, round((grid.col_coords[column_index + 1] - left) * 0.4))
            )
            right_radius = min(
                radius, max(3, round((right - grid.col_coords[end_index - 1]) * 0.4))
            )
            left_path = _trace_column_border(image, left, top, bottom, left_radius)
            right_path = _trace_column_border(image, right, top, bottom, right_radius)
            if (
                left_path is not None
                and right_path is not None
                and np.all(left_path < right_path)
            ):
                for offset, (start, end) in enumerate(zip(left_path, right_path)):
                    redacted[
                        top + offset, max(0, int(start) - 2) : min(max_w, int(end) + 3)
                    ] = (fill_color if redacted.ndim == 3 else fill_color[0])
                continue
        if column_index in border_lines or end_index in border_lines:
            left_top, left_bottom = border_lines.get(column_index, (left, left))
            right_top, right_bottom = border_lines.get(end_index, (right, right))
            if left_top < right_top and left_bottom < right_bottom:
                # Cover the printed rule's thickness as well as its interior.
                polygon = np.array(
                    [
                        (int(np.floor(left_top)) - 1, top),
                        (int(np.ceil(right_top)) + 1, top),
                        (int(np.ceil(right_bottom)) + 1, bottom - 1),
                        (int(np.floor(left_bottom)) - 1, bottom - 1),
                    ],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(redacted, polygon, fill_color)
                continue
        cv2.rectangle(
            redacted,
            (left, top),
            (right - 1, bottom - 1),
            fill_color,
            thickness=-1,
        )
    return redacted
