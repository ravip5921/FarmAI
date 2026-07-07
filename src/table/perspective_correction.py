from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from typing import Any

@dataclass(frozen=True)
class PerspectiveCorrectionResult:
    image: np.ndarray
    corrected: bool
    corners: np.ndarray | None = None
    padded_corners: np.ndarray | None = None
    transform: np.ndarray | None = None
    output_size: tuple[int, int] | None = None


def _order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()

    ordered: np.ndarray = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[int(np.argmin(sums))]
    ordered[2] = pts[int(np.argmax(sums))]
    ordered[1] = pts[int(np.argmin(diffs))]
    ordered[3] = pts[int(np.argmax(diffs))]
    return ordered


def expand_corners(
    corners: np.ndarray,
    image_shape: tuple[int, ...],
    padding: int,
) -> np.ndarray:
    """Move table corners outward to keep context during perspective warp."""
    ordered = _order_corners(corners)
    pad = max(0, int(padding))
    if pad == 0:
        return ordered

    height, width = image_shape[:2]
    center = np.mean(ordered, axis=0)
    expanded = ordered.copy()
    for index, point in enumerate(ordered):
        direction = point - center
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            expanded[index] = point + direction / norm * pad

    expanded[:, 0] = np.clip(expanded[:, 0], 0, max(0, width - 1))
    expanded[:, 1] = np.clip(expanded[:, 1], 0, max(0, height - 1))
    return expanded.astype(np.float32)


def _target_size(corners: np.ndarray) -> tuple[int, int]:
    top_left, top_right, bottom_right, bottom_left = corners
    top_width = float(np.linalg.norm(top_right - top_left))
    bottom_width = float(np.linalg.norm(bottom_right - bottom_left))
    left_height = float(np.linalg.norm(bottom_left - top_left))
    right_height = float(np.linalg.norm(bottom_right - top_right))

    width = max(1, int(round(max(top_width, bottom_width))))
    height = max(1, int(round(max(left_height, right_height))))
    return width, height


def _combined_table_mask(
    horizontal_mask: np.ndarray,
    vertical_mask: np.ndarray,
) -> np.ndarray:
    combined = cv2.bitwise_or(horizontal_mask, vertical_mask)
    if combined.dtype != np.uint8:
        combined = np.clip(combined, 0, 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)


def find_table_corners(
    horizontal_mask: np.ndarray,
    vertical_mask: np.ndarray,
    *,
    min_area_ratio: float = 0.05,
) -> np.ndarray | None:
    """Find a quadrilateral outer table boundary from detected line masks."""
    if horizontal_mask.ndim != 2 or vertical_mask.ndim != 2:
        raise ValueError("find_table_corners expects 2D masks")

    mask = _combined_table_mask(horizontal_mask, vertical_mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = float(mask.shape[0] * mask.shape[1]) * float(min_area_ratio)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if float(cv2.contourArea(contour)) < min_area:
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter <= 0:
            continue

        for epsilon_ratio in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
            approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, closed=True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return _order_corners(approx.reshape(4, 2))

    return None


def _warp_interpolation(image: np.ndarray) -> int:
    if image.ndim == 2 and len(np.unique(image)) <= 2:
        return cv2.INTER_NEAREST
    return cv2.INTER_LINEAR


def warp_image_to_corners(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    padding: int = 0,
    output_size: tuple[int, int] | None = None,
) -> PerspectiveCorrectionResult:
    ordered = _order_corners(corners)
    padded = expand_corners(ordered, image.shape, padding)
    width, height = output_size or _target_size(padded)
    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(padded, destination)
    border_value = 255 if image.ndim == 2 else (255, 255, 255)
    warped = cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=_warp_interpolation(image),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return PerspectiveCorrectionResult(
        image=warped,
        corrected=True,
        corners=ordered,
        padded_corners=padded,
        transform=transform,
        output_size=(width, height),
    )


def correct_table_perspective(
    image: np.ndarray,
    horizontal_mask: np.ndarray,
    vertical_mask: np.ndarray,
    *,
    padding: int = 24,
) -> PerspectiveCorrectionResult:
    corners = find_table_corners(horizontal_mask, vertical_mask)
    if corners is None:
        return PerspectiveCorrectionResult(image=image, corrected=False)
    return warp_image_to_corners(image, corners, padding=padding)


def render_perspective_corners(
    image: np.ndarray,
    corners: np.ndarray | None,
    *,
    padded_corners: np.ndarray | None = None,
) -> np.ndarray:
    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    if padded_corners is not None:
        padded = np.asarray(padded_corners, dtype=np.int32).reshape(4, 1, 2)
        cv2.polylines(overlay, [padded], isClosed=True, color=(255, 0, 0), thickness=2)

    if corners is None:
        return overlay

    pts = np.asarray(corners, dtype=np.int32).reshape(4, 1, 2)
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    for index, point in enumerate(pts.reshape(4, 2), start=1):
        cv2.circle(overlay, tuple(point), radius=6, color=(0, 0, 255), thickness=-1)
        cv2.putText(
            overlay,
            str(index),
            tuple(point + np.array([8, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay
