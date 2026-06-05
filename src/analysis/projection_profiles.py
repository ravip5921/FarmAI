from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ProjectionAxis = Literal["horizontal", "vertical"]


@dataclass(frozen=True)
class ProjectionPeak:
    position: int
    score: float
    start: int
    end: int


def projection_profile(mask: np.ndarray, axis: ProjectionAxis) -> np.ndarray:
    """Count foreground pixels along rows or columns of a binary mask."""
    foreground = np.asarray(mask) > 0
    if foreground.ndim != 2:
        raise ValueError("projection_profile expects a 2D mask")

    if axis == "horizontal":
        return np.count_nonzero(foreground, axis=1).astype(np.int32)
    if axis == "vertical":
        return np.count_nonzero(foreground, axis=0).astype(np.int32)
    raise ValueError(f"Unsupported projection axis: {axis}")


def smooth_profile(profile: np.ndarray, window_size: int = 1) -> np.ndarray:
    """Return a moving-average-smoothed copy of a 1D projection profile."""
    profile_array = np.asarray(profile, dtype=np.float32)
    if profile_array.ndim != 1:
        raise ValueError("smooth_profile expects a 1D profile")
    if window_size <= 1 or profile_array.size == 0:
        return profile_array.copy()

    kernel = np.ones(int(window_size), dtype=np.float32) / float(window_size)
    return np.convolve(profile_array, kernel, mode="same")


def find_projection_peaks(
    profile: np.ndarray,
    *,
    min_peak_ratio: float = 0.1,
    min_peak_value: float = 1.0,
    max_gap: int = 2,
) -> list[ProjectionPeak]:
    """Cluster high-support bins in a projection profile into peak centers."""
    profile_array = np.asarray(profile, dtype=np.float32)
    if profile_array.ndim != 1:
        raise ValueError("find_projection_peaks expects a 1D profile")
    if profile_array.size == 0:
        return []

    max_value = float(np.max(profile_array))
    if max_value <= 0:
        return []

    threshold = max(float(min_peak_value), max_value * float(min_peak_ratio))
    indices = np.flatnonzero(profile_array >= threshold)
    if indices.size == 0:
        return []

    clusters: list[list[int]] = [[int(indices[0])]]
    for index in indices[1:]:
        value = int(index)
        if value - clusters[-1][-1] <= max_gap:
            clusters[-1].append(value)
        else:
            clusters.append([value])

    peaks: list[ProjectionPeak] = []
    for cluster in clusters:
        weights = profile_array[cluster]
        if float(np.sum(weights)) > 0:
            position = int(round(float(np.average(cluster, weights=weights))))
        else:
            position = int(round(float(np.mean(cluster))))
        peaks.append(
            ProjectionPeak(
                position=position,
                score=float(np.max(weights)),
                start=int(cluster[0]),
                end=int(cluster[-1]),
            )
        )

    return peaks


def peak_positions(peaks: list[ProjectionPeak]) -> list[int]:
    return [peak.position for peak in peaks]
