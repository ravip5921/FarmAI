from __future__ import annotations

import unittest

import numpy as np

from src.core.image import DocumentImage


class TestDocumentImage(unittest.TestCase):
    def test_copy_copies_metadata_and_image_reference(self) -> None:
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        doc = DocumentImage(image=arr, metadata={"a": 1})

        copy = doc.copy()

        # copy should be a different DocumentImage instance
        self.assertIsNot(doc, copy)
        # but image reference is preserved (shallow copy semantics)
        self.assertIs(copy.image, doc.image)
        # metadata should be a shallow copy (different dict)
        self.assertIsNot(doc.metadata, copy.metadata)
        self.assertEqual(copy.metadata, doc.metadata)

        # modifying the copy's metadata should not affect the original
        copy.metadata["b"] = 2
        self.assertNotIn("b", doc.metadata)


if __name__ == "__main__":
    unittest.main()
