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
        if column.index in filtered
        and not _is_divider_column(column.key)
    }


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
    for column_index in sorted(column_indices):
        if column_index + 1 >= len(grid.col_coords):
            continue
        left = max(0, int(grid.col_coords[column_index]))
        right = min(max_w, int(grid.col_coords[column_index + 1]))
        if right <= left or bottom <= top:
            continue
        cv2.rectangle(
            redacted,
            (left, top),
            (right - 1, bottom - 1),
            fill_color,
            thickness=-1,
        )
    return redacted
