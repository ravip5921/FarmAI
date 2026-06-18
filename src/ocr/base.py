from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class OcrText:
    text: str
    confidence: float | None = None


class CellOcrEngine(Protocol):
    def recognize(self, image: np.ndarray) -> OcrText: ...
