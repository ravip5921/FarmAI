from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

from src.ocr.trocr_engine import TrOcrConfig, TrOcrHandwrittenEngine


class _FakeTensor:
    def __init__(self) -> None:
        self.device = None

    def to(self, device: str):
        self.device = device
        return self


class _FakeProcessor:
    model_name = None

    @classmethod
    def from_pretrained(cls, model_name: str):
        cls.model_name = model_name
        return cls()

    def __call__(self, *, images, return_tensors: str):
        self.images = images
        self.return_tensors = return_tensors
        return types.SimpleNamespace(pixel_values=_FakeTensor())

    def batch_decode(self, generated_ids, *, skip_special_tokens: bool):
        self.generated_ids = generated_ids
        self.skip_special_tokens = skip_special_tokens
        return ["  decoded text  "]


class _FakeModel:
    model_name = None

    def __init__(self) -> None:
        self.device = None
        self.eval_called = False

    @classmethod
    def from_pretrained(cls, model_name: str):
        cls.model_name = model_name
        return cls()

    def to(self, device: str) -> None:
        self.device = device

    def eval(self) -> None:
        self.eval_called = True

    def generate(self, pixel_values, *, max_new_tokens: int):
        self.pixel_values = pixel_values
        self.max_new_tokens = max_new_tokens
        return ["ids"]


class _FakeImage:
    @staticmethod
    def fromarray(array: np.ndarray):
        return ("image", array.copy())


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


@contextmanager
def _no_grad():
    yield


def _fake_modules():
    torch = types.SimpleNamespace(cuda=_FakeCuda(), no_grad=_no_grad)
    pil = types.ModuleType("PIL")
    pil.Image = _FakeImage
    transformers = types.ModuleType("transformers")
    transformers.TrOCRProcessor = _FakeProcessor
    transformers.VisionEncoderDecoderModel = _FakeModel
    return {
        "torch": torch,
        "PIL": pil,
        "transformers": transformers,
    }


class TestTrOcrHandwrittenEngine(unittest.TestCase):
    def test_init_loads_model_and_recognize_decodes_text(self) -> None:
        with patch.dict(sys.modules, _fake_modules()):
            engine = TrOcrHandwrittenEngine(
                TrOcrConfig(
                    model_name="custom/model",
                    device="cpu",
                    max_new_tokens=12,
                )
            )

        result = engine.recognize(np.array([[0, 255]], dtype=np.uint8))

        self.assertEqual(_FakeProcessor.model_name, "custom/model")
        self.assertEqual(_FakeModel.model_name, "custom/model")
        self.assertEqual(engine.device, "cpu")
        self.assertTrue(engine.model.eval_called)
        self.assertEqual(engine.model.max_new_tokens, 12)
        self.assertEqual(result.text, "decoded text")
        self.assertIsNone(result.confidence)

    def test_prepare_image_accepts_grayscale_bgr_bgra_and_clips_values(self) -> None:
        with patch.dict(sys.modules, _fake_modules()):
            engine = TrOcrHandwrittenEngine(TrOcrConfig(device="cpu"))

        gray = engine._prepare_image(np.array([[300.0, -1.0]], dtype=np.float32))
        bgr = engine._prepare_image(np.zeros((1, 1, 3), dtype=np.uint8))
        bgra = engine._prepare_image(np.zeros((1, 1, 4), dtype=np.uint8))

        self.assertEqual(gray[1].dtype, np.uint8)
        self.assertEqual(gray[1].shape, (1, 2, 3))
        self.assertEqual(bgr[1].shape, (1, 1, 3))
        self.assertEqual(bgra[1].shape, (1, 1, 3))

    def test_prepare_image_rejects_unsupported_shapes(self) -> None:
        with patch.dict(sys.modules, _fake_modules()):
            engine = TrOcrHandwrittenEngine(TrOcrConfig(device="cpu"))

        with self.assertRaisesRegex(ValueError, "2D grayscale or 3/4-channel"):
            engine._prepare_image(np.zeros((1, 1, 2), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
