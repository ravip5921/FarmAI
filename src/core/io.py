from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:  # pragma: no cover - import availability depends on local environment
    import pypdfium2 as pdfium
except Exception:  # pragma: no cover - handled at runtime
    pdfium = None

from .image import DocumentImage


@dataclass(frozen=True)
class LoadedDocument:
    pages: list[DocumentImage]
    source_path: Path

    @property
    def is_multipage(self) -> bool:
        return len(self.pages) > 1


def _load_image_file(path: Path) -> DocumentImage:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return DocumentImage(image, metadata={"source_path": str(path)})


def _load_pdf_file(path: Path) -> LoadedDocument:
    pages: list[DocumentImage] = []
    if pdfium is not None:
        try:
            pdf = pdfium.PdfDocument(str(path))
        except Exception as exc:  # pragma: no cover - depends on local backend
            raise RuntimeError(f"Could not open PDF {path} with pypdfium2.") from exc

        for page_index in range(len(pdf)):
            page = pdf[page_index]
            bitmap = page.render(scale=2.0)
            rgb = bitmap.to_numpy()
            if rgb.ndim == 2:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            else:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pages.append(
                DocumentImage(
                    bgr,
                    metadata={
                        "source_path": str(path),
                        "page_index": page_index + 1,
                    },
                )
            )
    else:
        raise RuntimeError(
            f"Could not open PDF {path}. Install `pypdfium2` or convert the PDF "
            "to images first."
        )

    if not pages:
        raise ValueError(f"PDF has no pages: {path}")
    return LoadedDocument(pages=pages, source_path=path)


def load_document(path: str | Path) -> DocumentImage | LoadedDocument:
    """Load either a raster image or a PDF document from disk."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf_file(source)
    return _load_image_file(source)
