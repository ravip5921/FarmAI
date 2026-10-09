from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

from src.ocr import llm_client
from src.ocr.base import OcrText
from src.ocr.column_rules import ColumnOcrRule
from src.ocr.llm_client import (
    LlmOcrVerifier,
    LlmVisionConfig,
    LlmVisionOcrEngine,
    build_llm_ocr_prompt,
    extract_chat_content,
    parse_llm_text_response,
)


class _Response:
    def __init__(self, body: dict) -> None:
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class TestLlmClient(unittest.TestCase):
    def test_env_file_ignores_comments_and_existing_values(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".env"
            llm_client._load_env_file(path)
            path.write_text(
                "# comment\ninvalid\nFARMAI_TEST_ENV='from file'\n", encoding="utf-8"
            )
            with patch.dict(os.environ, {"FARMAI_TEST_ENV": "existing"}):
                llm_client._load_env_file(path)
                self.assertEqual(os.environ["FARMAI_TEST_ENV"], "existing")
            with patch.dict(os.environ, {}, clear=True):
                llm_client._load_env_file(path)
                self.assertEqual(os.environ["FARMAI_TEST_ENV"], "from file")

    def test_unconfigured_and_blank_ocr_responses(self) -> None:
        image = np.zeros((3, 3), dtype=np.uint8)
        result = LlmVisionOcrEngine(LlmVisionConfig(api_url="", model="")).recognize(
            image
        )
        self.assertIn("not configured", result.validation_error or "")
        engine = LlmVisionOcrEngine()
        with patch.object(
            engine, "_chat_with_image", return_value='{"text":"","status":"blank"}'
        ):
            self.assertEqual(engine.recognize(image).text, "")
        with patch.object(
            engine, "_chat_with_image", return_value='{"text":"","status":"ok"}'
        ):
            self.assertIn("status ok", engine.recognize(image).validation_error or "")

    def test_verifier_modes_and_validation(self) -> None:
        text = OcrText(text="12", raw_text="raw")
        self.assertFalse(LlmOcrVerifier("off").should_verify(text, None))
        self.assertTrue(LlmOcrVerifier("always").should_verify(text, None))
        self.assertFalse(LlmOcrVerifier("invalid").should_verify(text, None))
        self.assertFalse(LlmOcrVerifier("invalid").should_verify(text, object()))
        with self.assertRaises(ValueError):
            LlmOcrVerifier("wrong").should_verify(text, None)
        rule = Mock()
        rule.validate.return_value = (False, "", "bad value")
        engine = Mock()
        engine.recognize_with_rule.return_value = OcrText(text="bad")
        verifier = LlmOcrVerifier("always", engine=engine)
        rejected = verifier.verify(
            image=np.zeros((2, 2), dtype=np.uint8), rule=rule, ocr_text=text
        )
        self.assertEqual(rejected.validation_error, "bad value")
        rule.validate.return_value = (True, "normalized", None)
        accepted = verifier.verify(
            image=np.zeros((2, 2), dtype=np.uint8), rule=rule, ocr_text=text
        )
        self.assertEqual(accepted.text, "normalized")
        self.assertEqual(
            llm_client._ocr_guess_for_prompt(OcrText(text="", raw_text="raw")), "raw"
        )

    def test_response_formats_and_invalid_payloads(self) -> None:
        self.assertEqual(extract_chat_content({"response": "reply"}), "reply")
        self.assertEqual(
            extract_chat_content({"choices": [{"message": {"content": "reply"}}]}),
            "reply",
        )
        self.assertEqual(
            extract_chat_content({"choices": [{"text": "reply"}]}), "reply"
        )
        with self.assertRaises(RuntimeError):
            extract_chat_content({"choices": [{}]})
        self.assertEqual(
            parse_llm_text_response('prefix {"text":"yes"} suffix'), ("yes", "ok")
        )
        self.assertEqual(parse_llm_text_response('"plain"'), ("plain", "ok"))
        self.assertEqual(parse_llm_text_response("plain"), ("plain", "ok"))

    def test_image_encoding_and_log_options(self) -> None:
        image = np.array([[300.0, -1.0]], dtype=np.float32)
        encoded = llm_client.encode_image_base64(
            image, config=LlmVisionConfig(image_extension=".png")
        )
        self.assertTrue(encoded)
        with patch.object(llm_client.cv2, "imencode", return_value=(False, None)):
            with self.assertRaises(RuntimeError):
                llm_client.encode_image_base64(image, config=LlmVisionConfig())
        payload = {"messages": [None, {"images": "bad"}, {"images": ["abc"]}]}
        with patch.dict(os.environ, {"FARMAI_LLM_LOG_FULL_IMAGES": "1"}):
            self.assertEqual(llm_client._request_payload_for_log(payload), payload)
        with patch.dict(os.environ, {"FARMAI_LLM_LOG_FULL_IMAGES": ""}):
            redacted = llm_client._request_payload_for_log(payload)
        self.assertTrue(redacted["messages"][2]["images"][0]["redacted"])
        with patch.dict(os.environ, {"FARMAI_LLM_LOG_DIR": "relative-logs"}):
            self.assertEqual(
                llm_client._llm_log_dir(), llm_client._PROJECT_ROOT / "relative-logs"
            )
        rule = ColumnOcrRule(
            index=0,
            key="temperature",
            value_type="temperature",
            format="NN.N",
            pattern=r"^\d+$",
            range_min=1,
            range_max=100,
        )
        prompt = build_llm_ocr_prompt(rule=rule)
        for expected in (
            "Expected format",
            "Validation regex",
            "Expected range",
            "Allowed characters",
        ):
            self.assertIn(expected, prompt)

    def test_parse_llm_text_response_reads_json_content(self) -> None:
        self.assertEqual(
            parse_llm_text_response('{"text": "All good", "status": "ok"}'),
            ("All good", "ok"),
        )
        self.assertEqual(
            parse_llm_text_response('```json\n{"text":"88.1","status":"ok"}\n```'),
            ("88.1", "ok"),
        )

    def test_extract_chat_content_reads_ollama_style_response(self) -> None:
        content = extract_chat_content(
            {"message": {"role": "assistant", "content": "answer"}}
        )

        self.assertEqual(content, "answer")

    def test_prompt_includes_template_context_and_existing_ocr(self) -> None:
        rule = ColumnOcrRule(
            index=6,
            key="comments",
            name="Comments",
            value_type="english_text",
            common_values=("All good",),
        )

        prompt = build_llm_ocr_prompt(
            rule=rule,
            ocr_text=OcrText(text="All gaad", validation_error="bad"),
        )

        self.assertIn("Column key: comments", prompt)
        self.assertIn("Column name: Comments", prompt)
        self.assertIn("Common values", prompt)
        self.assertIn("All gaad", prompt)

    def test_llm_engine_posts_image_and_parses_response(self) -> None:
        engine = LlmVisionOcrEngine(
            LlmVisionConfig(api_url="http://example.test/api/chat", model="model")
        )

        with patch(
            "src.ocr.llm_client.urllib.request.urlopen",
            return_value=_Response(
                {"message": {"content": '{"text":"88.1","status":"ok"}'}}
            ),
        ) as urlopen:
            result = engine.recognize(np.zeros((5, 5), dtype=np.uint8))

        self.assertEqual(result.text, "88.1")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://example.test/api/chat")
        self.assertEqual(request.headers["User-agent"], "curl/8.0.0")
        self.assertEqual(request.headers["Accept"], "application/json")
        self.assertEqual(len(request.headers["Idempotency-key"]), 64)
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "model")
        self.assertTrue(payload["messages"][0]["images"][0])

    def test_llm_engine_can_log_redacted_request_payload(self) -> None:
        engine = LlmVisionOcrEngine(
            LlmVisionConfig(api_url="http://example.test/api/chat", model="model")
        )

        with TemporaryDirectory() as tmpdir:
            with (
                patch.dict(
                    "os.environ",
                    {
                        "FARMAI_LLM_LOG_REQUESTS": "1",
                        "FARMAI_LLM_LOG_DIR": tmpdir,
                        "FARMAI_LLM_LOG_FULL_IMAGES": "",
                    },
                ),
                patch(
                    "src.ocr.llm_client.urllib.request.urlopen",
                    return_value=_Response(
                        {"message": {"content": '{"text":"88.1","status":"ok"}'}}
                    ),
                ),
            ):
                engine.recognize(np.zeros((5, 5), dtype=np.uint8))

            files = list(Path(tmpdir).glob("llm_request_*.json"))
            self.assertEqual(len(files), 1)
            record = json.loads(files[0].read_text(encoding="utf-8"))

        image_record = record["request"]["messages"][0]["images"][0]
        self.assertEqual(record["headers"]["User-Agent"], "curl/8.0.0")
        self.assertTrue(image_record["redacted"])
        self.assertIn("base64_chars", image_record)
        self.assertIn("sha256", image_record)
        self.assertIn("88.1", record["response_raw"])

    def test_llm_engine_returns_validation_error_on_timeout(self) -> None:
        engine = LlmVisionOcrEngine(
            LlmVisionConfig(api_url="http://example.test/api/chat", model="model")
        )

        with patch(
            "src.ocr.llm_client.urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            result = engine.recognize(np.zeros((5, 5), dtype=np.uint8))

        self.assertEqual(result.text, "")
        self.assertIn("timed out", result.validation_error or "")

    def test_llm_verifier_checks_invalid_mode(self) -> None:
        verifier = LlmOcrVerifier(mode="invalid")
        rule = ColumnOcrRule(
            index=1,
            key="hi",
            value_type="temperature",
            pattern=r"^\d+$",
        )

        self.assertTrue(
            verifier.should_verify(
                OcrText(text="", validation_error="expected temperature pattern"),
                rule,
            )
        )
        self.assertFalse(verifier.should_verify(OcrText(text="88"), rule))

    def test_llm_verifier_falls_back_to_ocr_text_on_timeout(self) -> None:
        engine = LlmVisionOcrEngine(
            LlmVisionConfig(api_url="http://example.test/api/chat", model="model")
        )
        verifier = LlmOcrVerifier(mode="always", engine=engine)

        with patch(
            "src.ocr.llm_client.urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            result = verifier.verify(
                image=np.zeros((5, 5), dtype=np.uint8),
                rule=None,
                ocr_text=OcrText(text="baseline", confidence=42.0),
            )

        self.assertEqual(result.text, "baseline")
        self.assertEqual(result.confidence, 42.0)
        self.assertIn("timed out", result.raw_text or "")


if __name__ == "__main__":
    unittest.main()
