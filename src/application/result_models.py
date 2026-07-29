from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProcessingSettings:
    template_id: str | None = "boar_room"
    ocr_engine: str = "llm-vision"
    extra_filtered_columns: tuple[str, ...] = ()
    ocr_padding: int = 2
    ocr_context_padding: int = 0


@dataclass(frozen=True)
class ProcessingProgress:
    stage: str
    completed: int = 0
    total: int = 0
    page_number: int = 1
    page_count: int = 1
    message: str = ""


@dataclass(frozen=True)
class UiColumn:
    index: int
    source_index: int
    key: str
    name: str
    value_type: str = "text"
    format: str | None = None
    range_min: float | None = None
    range_max: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "source_index": self.source_index,
            "key": self.key,
            "name": self.name,
            "value_type": self.value_type,
            "format": self.format,
            "range_min": self.range_min,
            "range_max": self.range_max,
        }


@dataclass(frozen=True)
class UiCell:
    row: int
    column_index: int
    source_column_index: int
    column_key: str
    column_name: str
    bbox: tuple[int, int, int, int]
    ocr_text: str
    confidence: float | None = None
    raw_text: str | None = None
    validation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "column_index": self.column_index,
            "source_column_index": self.source_column_index,
            "column_key": self.column_key,
            "column_name": self.column_name,
            "bbox": list(self.bbox),
            "ocr_text": self.ocr_text,
            "reviewed_text": self.ocr_text,
            "was_edited": False,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
            "validation_error": self.validation_error,
            "ground_truth_text": None,
            "ground_truth_match": None,
            "state": "validation_warning" if self.validation_error else "unscored",
        }


@dataclass
class PageProcessingResult:
    page_number: int
    source_image: np.ndarray = field(repr=False)
    overlay_image: np.ndarray = field(repr=False)
    image_width: int = 0
    image_height: int = 0
    skew_angle: float = 0.0
    columns: list[UiColumn] = field(default_factory=list)
    cells: list[UiCell] = field(default_factory=list)
    data_row_count: int = 0

    @property
    def warning_count(self) -> int:
        return sum(1 for cell in self.cells if cell.validation_error)

    def to_dict(
        self,
        *,
        source_url: str,
        overlay_url: str,
    ) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "source_url": source_url,
            "overlay_url": overlay_url,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "skew_angle": self.skew_angle,
            "columns": [column.to_dict() for column in self.columns],
            "cells": [cell.to_dict() for cell in self.cells],
            "data_row_count": self.data_row_count,
            "warning_count": self.warning_count,
            "metrics": None,
        }


@dataclass
class DocumentProcessingResult:
    filename: str
    template_id: str | None
    template_name: str | None
    ocr_engine: str
    pages: list[PageProcessingResult] = field(default_factory=list)

    @property
    def warning_count(self) -> int:
        return sum(page.warning_count for page in self.pages)
