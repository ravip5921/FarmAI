from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from src.core.image import DocumentImage
from src.core.io import LoadedDocument, _load_pdf_file, load_document


class _FakeBitmap:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def to_numpy(self) -> np.ndarray:
        return self.array


class _FakePage:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.closed = False

    def render(self, *, scale: float):
        self.scale = scale
        return _FakeBitmap(self.array)

    def close(self) -> None:
        self.closed = True


class _FakePdf:
    def __init__(self, arrays: list[np.ndarray]) -> None:
        self.pages = [_FakePage(array) for array in arrays]
        self.closed = False

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> _FakePage:
        return self.pages[index]

    def close(self) -> None:
        self.closed = True


class _FakePdfium:
    def __init__(self, pdf: _FakePdf) -> None:
        self.pdf = pdf

    def PdfDocument(self, path: str) -> _FakePdf:
        self.path = path
        return self.pdf


class TestCoreIo(unittest.TestCase):
    def test_loaded_document_reports_multipage(self) -> None:
        one = LoadedDocument([DocumentImage(np.zeros((1, 1)))], Path("one.pdf"))
        two = LoadedDocument(
            [DocumentImage(np.zeros((1, 1))), DocumentImage(np.zeros((1, 1)))],
            Path("two.pdf"),
        )

        self.assertFalse(one.is_multipage)
        self.assertTrue(two.is_multipage)

    def test_load_document_reads_image_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.png"
            cv2.imwrite(str(path), np.full((2, 3, 3), 255, dtype=np.uint8))

            loaded = load_document(path)

        self.assertIsInstance(loaded, DocumentImage)
        self.assertEqual(loaded.image.shape, (2, 3, 3))
        self.assertEqual(loaded.metadata["source_path"], str(path))

    def test_load_document_raises_for_unreadable_image(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_document("missing.jpg")

    def test_load_document_routes_pdfs(self) -> None:
        with patch("src.core.io._load_pdf_file", return_value="pdf") as load_pdf:
            loaded = load_document("record.PDF")

        self.assertEqual(loaded, "pdf")
        load_pdf.assert_called_once_with(Path("record.PDF"))

    def test_load_pdf_file_converts_pages_and_metadata(self) -> None:
        gray = np.array([[0, 255]], dtype=np.uint8)
        rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
        fake_pdf = _FakePdf([gray, rgb])
        fake_pdfium = _FakePdfium(fake_pdf)

        with patch("src.core.io.pdfium", fake_pdfium):
            loaded = _load_pdf_file(Path("record.pdf"))

        self.assertEqual(len(loaded.pages), 2)
        self.assertEqual(loaded.pages[0].image.shape, (1, 2, 3))
        self.assertEqual(loaded.pages[1].image.shape, (1, 1, 3))
        self.assertEqual(loaded.pages[0].metadata["page_index"], 1)
        self.assertEqual(loaded.pages[1].metadata["page_index"], 2)
        self.assertTrue(all(page.closed for page in fake_pdf.pages))
        self.assertTrue(fake_pdf.closed)

    def test_load_pdf_file_requires_pdf_backend(self) -> None:
        with patch("src.core.io.pdfium", None):
            with self.assertRaisesRegex(RuntimeError, "Install `pypdfium2`"):
                _load_pdf_file(Path("record.pdf"))

    def test_load_pdf_file_rejects_empty_pdf(self) -> None:
        fake_pdf = _FakePdf([])

        with patch("src.core.io.pdfium", _FakePdfium(fake_pdf)):
            with self.assertRaisesRegex(ValueError, "PDF has no pages"):
                _load_pdf_file(Path("empty.pdf"))

        self.assertTrue(fake_pdf.closed)


if __name__ == "__main__":
    unittest.main()
