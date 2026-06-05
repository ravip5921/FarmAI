from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentImage:
    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "DocumentImage":
        return DocumentImage(image=self.image, metadata=dict(self.metadata))
