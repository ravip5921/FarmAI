from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2


def connected_components(binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run connected component analysis on a binary image.

    Args:
        binary: Binary image (uint8 with 0/255 or boolean).

    Returns:
        num_labels, labels, stats, centroids (following OpenCV order)
    """
    if binary.dtype != np.uint8:
        bin_img = (binary > 0).astype(np.uint8) * 255
    else:
        bin_img = binary.copy()

    # Ensure background is 0, foreground is 255
    _, thresh = cv2.threshold(bin_img, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    return num_labels, labels, stats, centroids


def filter_components_by_area(binary: np.ndarray, min_area: int = 10) -> np.ndarray:
    """Return a binary mask keeping components with area >= min_area.

    Args:
        binary: Binary image (0/255 or boolean).
        min_area: Minimum area in pixels to keep.

    Returns:
        mask: uint8 binary mask with kept components set to 255.
    """
    num_labels, labels, stats, _ = connected_components(binary)
    mask = np.zeros_like(labels, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            mask[labels == label] = 255
    return mask


def estimate_average_character_height(binary: np.ndarray, min_area: int = 10) -> int:
    """Estimate average character height from connected components.

    Uses the median of component heights (bounding box heights) after filtering
    small components by area.

    Args:
        binary: Binary image (0/255 or boolean).
        min_area: Minimum component area to consider.

    Returns:
        estimated height in pixels (int). Returns 0 if no components found.
    """
    num_labels, labels, stats, _ = connected_components(binary)
    heights = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if h > 0:
            heights.append(h)

    if not heights:
        return 0

    # Use median as robust estimator
    median_h = int(np.median(np.array(heights)))
    return median_h
