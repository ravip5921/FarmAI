from __future__ import annotations

import math

import cv2
import numpy as np

from src.core.image import DocumentImage
from src.core.stage import PipelineStage


class SkewCorrectionStage(PipelineStage):
    def __init__(
        self, canny_low: int = 50, canny_high: int = 150, hough_threshold: int = 80
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold

    def _estimate_angle(self, image: np.ndarray) -> float:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=max(gray.shape[1] // 4, 40),
            maxLineGap=20,
        )

        if lines is None:
            print("[SkewCorrectionStage] No lines detected by Hough transform.")
            return 0.0

        angles: list[float] = []
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) <= 45:
                angles.append(angle)

        if not angles:
            print("[SkewCorrectionStage] No valid angles extracted.")
            return 0.0

        median_angle = float(np.median(angles))
        print(f"[SkewCorrectionStage] Detected angle: {median_angle:.2f}°")
        return median_angle

    def process(self, doc: DocumentImage) -> DocumentImage:
        image = doc.image
        angle = self._estimate_angle(image)

        if abs(angle) < 0.1:
            metadata = dict(doc.metadata)
            metadata["deskewed"] = False
            metadata["skew_angle"] = 0.0
            return DocumentImage(image.copy(), metadata)

        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        is_binary = bool(doc.metadata.get("binary")) or (
            image.ndim == 2 and len(np.unique(image)) <= 2
        )
        interpolation = cv2.INTER_NEAREST if is_binary else cv2.INTER_LINEAR
        rotated = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

        metadata = dict(doc.metadata)
        metadata["deskewed"] = True
        metadata["skew_angle"] = angle
        metadata["skew_interpolation"] = "nearest" if is_binary else "linear"

        return DocumentImage(rotated, metadata)
