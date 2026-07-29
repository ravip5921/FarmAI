from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import OcrText


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_load_env_file(_PROJECT_ROOT / ".env")

DEFAULT_LLM_API_URL = os.getenv("FARMAI_LLM_API_URL", "")
DEFAULT_LLM_MODEL = os.getenv("FARMAI_LLM_MODEL", "")
DEFAULT_LLM_TIMEOUT_SECONDS = float(os.getenv("FARMAI_LLM_TIMEOUT_SECONDS", "120"))


@dataclass(frozen=True)
class LlmVisionConfig:
    api_url: str = DEFAULT_LLM_API_URL
    model: str = DEFAULT_LLM_MODEL
    timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    image_extension: str = ".jpg"
    jpeg_quality: int = 92


class LlmVisionOcrEngine:
    """Vision LLM OCR backend using an Ollama-style chat/image API."""

    def __init__(self, config: LlmVisionConfig | None = None):
        self.config = config or LlmVisionConfig()

    def recognize(self, image: np.ndarray) -> OcrText:
        return self.recognize_with_rule(image, rule=None)

    def recognize_with_rule(
        self,
        image: np.ndarray,
        *,
        rule: Any | None = None,
        ocr_text: OcrText | None = None,
    ) -> OcrText:
        prompt = build_llm_ocr_prompt(rule=rule, ocr_text=ocr_text)
        try:
            content = self._chat_with_image(prompt, image)
        except RuntimeError as exc:
            return OcrText(text="", confidence=None, validation_error=str(exc))
        text, status = parse_llm_text_response(content)
        if status in {"blank", "illegible", "uncertain"} and not text:
            return OcrText(text="", confidence=None, raw_text=content)
        return OcrText(
            text=text,
            confidence=None,
            raw_text=content if text != content.strip() else None,
            validation_error=None if text else f"llm returned status {status}",
        )

    def _chat_with_image(self, prompt: str, image: np.ndarray) -> str:
        if not self.config.api_url or not self.config.model:
            raise RuntimeError(
                "LLM OCR is not configured. Set FARMAI_LLM_API_URL and "
                "FARMAI_LLM_MODEL in .env or the process environment."
            )
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encode_image_base64(image, config=self.config)],
                }
            ],
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "curl/8.0.0",
        }
        request = urllib.request.Request(
            self.config.api_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                raw = response.read().decode("utf-8")
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            _write_llm_debug_record(
                api_url=self.config.api_url,
                payload=payload,
                headers=headers,
                error=str(exc),
            )
            raise RuntimeError(f"LLM OCR request failed: {exc}") from exc
        _write_llm_debug_record(
            api_url=self.config.api_url,
            payload=payload,
            headers=headers,
            response_raw=raw,
        )
        return extract_chat_content(json.loads(raw))


@dataclass(frozen=True)
class LlmOcrVerifier:
    mode: str
    engine: LlmVisionOcrEngine = LlmVisionOcrEngine()

    def should_verify(self, result: OcrText, rule: Any | None) -> bool:
        mode = self.mode.strip().lower()
        if mode == "off":
            return False
        if mode == "always":
            return True
        if mode != "invalid":
            raise ValueError(f"Unknown LLM verifier mode: {self.mode}")
        if not result.text.strip() or result.validation_error:
            return True
        if rule is None:
            return False
        validate = getattr(rule, "validate", None)
        if validate is None:
            return False
        valid, _text, _error = validate(result.text)
        return not valid

    def verify(
        self,
        *,
        image: np.ndarray,
        rule: Any | None,
        ocr_text: OcrText,
    ) -> OcrText:
        llm_result = self.engine.recognize_with_rule(
            image,
            rule=rule,
            ocr_text=ocr_text,
        )
        if llm_result.validation_error and not llm_result.text.strip():
            return OcrText(
                text=ocr_text.text,
                confidence=ocr_text.confidence,
                raw_text=(
                    f"ocr={_ocr_guess_for_prompt(ocr_text)!r}; "
                    f"llm_error={llm_result.validation_error}"
                ),
                validation_error=ocr_text.validation_error,
            )
        validate = getattr(rule, "validate", None)
        if validate is not None and llm_result.text.strip():
            valid, text, error = validate(llm_result.text)
            if not valid:
                return OcrText(
                    text="",
                    confidence=ocr_text.confidence,
                    raw_text=(
                        f"ocr={ocr_text.text!r}; llm={llm_result.text!r}; "
                        f"llm_raw={llm_result.raw_text or ''}"
                    ),
                    validation_error=error,
                )
            llm_result = OcrText(
                text=text,
                confidence=llm_result.confidence,
                raw_text=(
                    f"ocr={_ocr_guess_for_prompt(ocr_text)!r}; "
                    f"llm_raw={llm_result.raw_text or ''}"
                ),
            )
        return llm_result


def build_llm_ocr_prompt(
    *,
    rule: Any | None,
    ocr_text: OcrText | None = None,
) -> str:
    lines = [
        "You are reading one cropped cell from a farm record table.",
        'Return JSON only: {"text":"...","status":"ok|blank|illegible|uncertain"}.',
        "Use an empty text value for blank or illegible cells.",
        "Do not explain your answer.",
    ]
    if rule is not None:
        lines.extend(_rule_prompt_lines(rule))
    if ocr_text is not None:
        lines.append(f"Existing OCR guess: {_ocr_guess_for_prompt(ocr_text)!r}")
        if ocr_text.validation_error:
            lines.append(f"Existing OCR validation error: {ocr_text.validation_error}")
    return "\n".join(lines)


def _ocr_guess_for_prompt(ocr_text: OcrText) -> str:
    if ocr_text.text.strip():
        return ocr_text.text
    return ocr_text.raw_text or ocr_text.text


def _rule_prompt_lines(rule: Any) -> list[str]:
    lines = [
        f"Column key: {getattr(rule, 'key', '')}",
        f"Column name: {getattr(rule, 'name', '')}",
        f"Expected value type: {getattr(rule, 'value_type', '')}",
    ]
    value_format = getattr(rule, "format", None)
    if value_format:
        lines.append(f"Expected format: {value_format}")
    pattern = getattr(rule, "pattern", None)
    if pattern:
        lines.append(f"Validation regex: {pattern}")
    range_min = getattr(rule, "range_min", None)
    range_max = getattr(rule, "range_max", None)
    if range_min is not None or range_max is not None:
        lines.append(f"Expected range: {range_min} to {range_max}")
    common_values = tuple(getattr(rule, "common_values", ()) or ())
    if common_values:
        lines.append(f"Common values: {json.dumps(common_values)}")
    if getattr(rule, "value_type", "") == "temperature":
        lines.append("Allowed characters for this column: digits and decimal point.")
    return lines


def encode_image_base64(image: np.ndarray, *, config: LlmVisionConfig) -> str:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    params: list[int] = []
    if config.image_extension.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(config.jpeg_quality)]
    ok, encoded = cv2.imencode(config.image_extension, array, params)
    if not ok:
        raise RuntimeError("Could not encode cell image for LLM OCR")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def extract_chat_content(response: dict[str, Any]) -> str:
    message = response.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(response.get("response"), str):
        return response["response"]
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            choice_message = first.get("message")
            if isinstance(choice_message, dict) and isinstance(
                choice_message.get("content"), str
            ):
                return choice_message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    raise RuntimeError(f"LLM OCR response did not contain message content: {response}")


def parse_llm_text_response(content: str) -> tuple[str, str]:
    stripped = _strip_code_fence(content.strip())
    data = _try_load_json(stripped)
    if data is None:
        data = _try_load_json_from_substring(stripped)
    if isinstance(data, dict):
        text = str(data.get("text", "")).strip()
        status = str(data.get("status", "ok")).strip().lower() or "ok"
        return text, status
    return stripped.strip().strip('"'), "ok"


def _strip_code_fence(value: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL)
    return match.group(1) if match else value


def _try_load_json(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _try_load_json_from_substring(value: str):
    match = re.search(r"\{.*\}", value, flags=re.DOTALL)
    if match is None:
        return None
    return _try_load_json(match.group(0))


def _write_llm_debug_record(
    *,
    api_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    response_raw: str | None = None,
    error: str | None = None,
) -> None:
    if not _env_bool("FARMAI_LLM_LOG_REQUESTS"):
        return
    output_dir = _llm_log_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "api_url": api_url,
        "headers": headers,
        "request": _request_payload_for_log(payload),
        "response_raw": response_raw,
        "error": error,
    }
    path = output_dir / f"llm_request_{time.time_ns()}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _request_payload_for_log(payload: dict[str, Any]) -> dict[str, Any]:
    if _env_bool("FARMAI_LLM_LOG_FULL_IMAGES"):
        return copy.deepcopy(payload)

    redacted = copy.deepcopy(payload)
    for message in redacted.get("messages", []):
        if not isinstance(message, dict):
            continue
        images = message.get("images")
        if not isinstance(images, list):
            continue
        message["images"] = [
            {
                "redacted": True,
                "base64_chars": len(str(image)),
                "sha256": hashlib.sha256(str(image).encode("ascii")).hexdigest(),
            }
            for image in images
        ]
    return redacted


def _llm_log_dir() -> Path:
    value = os.getenv("FARMAI_LLM_LOG_DIR", "debug_outputs/llm_requests")
    path = Path(value)
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
