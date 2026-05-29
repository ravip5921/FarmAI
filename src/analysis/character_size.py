from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .connected_components import connected_components, estimate_average_character_height


@dataclass
class CharacterSizeReport:
    average_height: int
    median_height: int
    component_count: int


def _extract_component_heights(binary: np.ndarray, min_area: int = 10) -> list[int]:
    if binary.dtype != np.uint8:
        binary = (binary > 0).astype(np.uint8) * 255

    num_labels, _, stats, _ = connected_components(binary)
    heights: list[int] = []
    for label in range(1, num_labels):

        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if height > 0:
            heights.append(height)

    return heights

def estimate_character_size(binary: np.ndarray, min_area: int = 10) -> CharacterSizeReport:
    """Estimate basic character-size statistics from a binary image."""

    heights = _extract_component_heights(binary, min_area=min_area)
    if not heights:
        return CharacterSizeReport(average_height=0, median_height=0, component_count=0)

    height_array = np.asarray(heights, dtype=np.int32)
    average_height = estimate_average_character_height(binary, min_area=min_area)
    median_height = int(np.median(height_array))

    return CharacterSizeReport(
        average_height=average_height,
        median_height=median_height,
        component_count=len(heights),
    )
