"""Packy/OpenAI-compatible LLM client helpers.

中文注释：
第一阶段只封装“可稳定调用”的底层客户端，不把它直接绑到 planner/critic。
这样后续接 advisor 时，Prompt 组织、业务回退和模型调用细节可以解耦。
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_PACKY_BASE_URL = "https://www.packyapi.com/v1"
DEFAULT_PACKY_MODEL = "gpt-5.2"


class PackyLLMError(RuntimeError):
    """Raised when the Packy/OpenAI-compatible backend returns an unusable response."""


class PackyLLMConfig(BaseModel):
    """Configuration for the Packy/OpenAI-compatible client."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    api_key: str = Field(min_length=1)
    base_url: str = Field(default=DEFAULT_PACKY_BASE_URL, min_length=1)
    model: str = Field(default=DEFAULT_PACKY_MODEL, min_length=1)
    timeout_sec: float = Field(default=30.0, gt=0.0, le=300.0)

    @classmethod
    def from_env(cls) -> "PackyLLMConfig":
        """Build config from project env vars first, then OpenAI-compatible fallbacks."""

        api_key = os.getenv("AEGRA_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("missing AEGRA_LLM_API_KEY or OPENAI_API_KEY")
        base_url = (
            os.getenv("AEGRA_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or DEFAULT_PACKY_BASE_URL
        )
        model = os.getenv("AEGRA_LLM_MODEL") or DEFAULT_PACKY_MODEL
        timeout_value = os.getenv("AEGRA_LLM_TIMEOUT_SEC")
        timeout_sec = float(timeout_value) if timeout_value else 30.0
        return cls(api_key=api_key, base_url=base_url, model=model, timeout_sec=timeout_sec)


class PackyLLMResponse(BaseModel):
    """Normalized text response returned by the low-level Packy client."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str = Field(min_length=1)
    text: str = Field(default="")
    raw_payload: dict[str, Any] | None = None
    raw_text: str | None = None
    finish_reason: str | None = None


def _extract_text_from_content_blocks(content: Any) -> str:
    """Extract text from OpenAI-style content blocks."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("text"), str):
            parts.append(item["text"])
            continue
        if item.get("type") == "text" and isinstance(item.get("content"), str):
            parts.append(item["content"])
    return "".join(parts)


def _extract_text_from_completion_payload(payload: dict[str, Any]) -> tuple[str, str | None]:
    """Extract assistant text from a standard chat-completions JSON payload."""

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return "", None

    parts: list[str] = []
    finish_reason: str | None = None
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        if finish_reason is None and isinstance(choice.get("finish_reason"), str):
            finish_reason = choice["finish_reason"]
        message = choice.get("message")
        if isinstance(message, dict):
            block_text = _extract_text_from_content_blocks(message.get("content"))
            if block_text:
                parts.append(block_text)
        delta = choice.get("delta")
        if isinstance(delta, dict):
            chunk_text = _extract_text_from_content_blocks(delta.get("content"))
            if chunk_text:
                parts.append(chunk_text)
    return "".join(parts), finish_reason


def _extract_text_from_sse_blob(raw_text: str) -> tuple[str, str | None]:
    """Parse an SSE-like chunk stream returned by non-standard gateways.

    中文注释：
    PackyAPI 当前可能在非 stream 请求下也返回 `data: {...}` 的 chunk 文本。
    这里把所有 delta.content 拼接起来，给上层一个稳定文本结果。
    """

    parts: list[str] = []
    finish_reason: str | None = None
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("data: "):
            continue
        payload_text = line[len("data: ") :]
        if payload_text == "[DONE]":
            break
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        chunk_text, chunk_finish_reason = _extract_text_from_completion_payload(payload)
        if chunk_text:
            parts.append(chunk_text)
        if finish_reason is None and chunk_finish_reason:
            finish_reason = chunk_finish_reason
    return "".join(parts), finish_reason


class PackyLLMClient:
    """Minimal OpenAI-compatible client tuned for PackyAPI quirks."""

    def __init__(self, config: PackyLLMConfig, *, http_client: httpx.Client | None = None) -> None:
        self._config = config
        self._owns_http_client = http_client is None
        self._http = http_client or httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout_sec,
        )

    @property
    def config(self) -> PackyLLMConfig:
        return self._config

    def close(self) -> None:
        if self._owns_http_client:
            self._http.close()

    def __enter__(self) -> "PackyLLMClient":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> None:
        self.close()

    def list_models(self) -> list[str]:
        """Return model IDs exposed by the configured gateway."""

        response = self._http.get("/models")
        self._raise_for_status(response)
        payload = self._parse_json_response(response)
        models = payload.get("data")
        if not isinstance(models, list):
            raise PackyLLMError("gateway /models response does not contain a data list")
        model_ids: list[str] = []
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                model_ids.append(item["id"])
        return model_ids

    def complete_chat(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> PackyLLMResponse:
        """Call `/chat/completions` and normalize text output.

        中文注释：
        第一阶段只支持 chat completions，因为我们已经验证过 PackyAPI 的
        `responses` 接口不稳定，而 `chat/completions` 至少能稳定返回内容。
        """

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload: dict[str, Any] = {
            "model": model or self._config.model,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        response = self._http.post("/chat/completions", json=payload)
        self._raise_for_status(response)

        raw_text = response.text
        json_payload = self._try_parse_json(raw_text)
        text = ""
        finish_reason: str | None = None
        if json_payload is not None:
            text, finish_reason = _extract_text_from_completion_payload(json_payload)
        if not text:
            text, finish_reason = _extract_text_from_sse_blob(raw_text)
        if not text:
            raise PackyLLMError("gateway returned no assistant text in JSON or SSE-compatible form")

        return PackyLLMResponse(
            model=str(payload["model"]),
            text=text,
            raw_payload=json_payload,
            raw_text=raw_text,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _try_parse_json(raw_text: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _parse_json_response(self, response: httpx.Response) -> dict[str, Any]:
        payload = self._try_parse_json(response.text)
        if payload is None:
            raise PackyLLMError("gateway response is not valid JSON")
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        detail = response.text.strip()
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message") or error.get("code") or detail
                raise PackyLLMError(f"gateway request failed: {message}")
        raise PackyLLMError(f"gateway request failed with status {response.status_code}: {detail}")


__all__ = [
    "DEFAULT_PACKY_BASE_URL",
    "DEFAULT_PACKY_MODEL",
    "PackyLLMClient",
    "PackyLLMConfig",
    "PackyLLMError",
    "PackyLLMResponse",
    "_extract_text_from_completion_payload",
    "_extract_text_from_sse_blob",
]
