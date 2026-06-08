"""Packy/OpenAI-compatible LLM client helpers.

中文注释：
第一阶段只封装“可稳定调用”的底层客户端，不把它直接绑到 planner/critic。
这样后续接 advisor 时，Prompt 组织、业务回退和模型调用细节可以解耦。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    input_cost_per_1m_tokens: float | None = Field(default=None, ge=0.0)
    output_cost_per_1m_tokens: float | None = Field(default=None, ge=0.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_backoff_sec: float = Field(default=1.0, ge=0.0, le=30.0)

    @field_validator("base_url", mode="before")
    @classmethod
    def _normalize_base_url(cls, value: str | None) -> str:
        """Accept Packy root URLs while keeping chat-completions calls on /v1."""

        base_url = str(value or DEFAULT_PACKY_BASE_URL).strip().rstrip("/")
        if base_url in {"https://www.packyapi.com", "http://www.packyapi.com"}:
            return f"{base_url}/v1"
        return base_url

    @classmethod
    def from_env(cls) -> "PackyLLMConfig":
        """Build config from project env vars first, then OpenAI-compatible fallbacks."""

        if not os.getenv("AEGRA_LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            load_llm_env_file()
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
        retries_value = os.getenv("AEGRA_LLM_MAX_RETRIES")
        max_retries = int(retries_value) if retries_value else 2
        backoff_value = os.getenv("AEGRA_LLM_RETRY_BACKOFF_SEC")
        retry_backoff_sec = float(backoff_value) if backoff_value else 1.0
        input_cost = _env_float("AEGRA_LLM_INPUT_COST_PER_1M_TOKENS")
        output_cost = _env_float("AEGRA_LLM_OUTPUT_COST_PER_1M_TOKENS")
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
            input_cost_per_1m_tokens=input_cost,
            output_cost_per_1m_tokens=output_cost,
        )


def _env_float(key: str) -> float | None:
    value = os.getenv(key)
    if value in {None, ""}:
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def load_llm_env_file(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE entries from a local .env file without overwriting env vars."""

    env_path = Path(path)
    if not env_path.exists():
        return
    entries: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            entries[key] = value

    aegra_key_is_blank = "AEGRA_LLM_API_KEY" in entries and not entries["AEGRA_LLM_API_KEY"]
    for key, value in entries.items():
        if aegra_key_is_blank and key.startswith("AEGRA_LLM_"):
            continue
        if not value or key in os.environ:
            continue
        os.environ[key] = value


class PackyLLMResponse(BaseModel):
    """Normalized text response returned by the low-level Packy client."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str = Field(min_length=1)
    text: str = Field(default="")
    usage: dict[str, int] | None = None
    cost_usd: float | None = None
    raw_payload: dict[str, Any] | None = None
    raw_text: str | None = None
    finish_reason: str | None = None


_LLM_USAGE_LEDGER: list[dict[str, Any]] = []
_LLM_USAGE_LEDGER_LOCK = Lock()


def reset_llm_usage_ledger() -> None:
    """Clear process-local LLM usage accounting."""

    with _LLM_USAGE_LEDGER_LOCK:
        _LLM_USAGE_LEDGER.clear()


def get_llm_usage_ledger() -> list[dict[str, Any]]:
    """Return a copy of process-local LLM usage records."""

    with _LLM_USAGE_LEDGER_LOCK:
        return [dict(item) for item in _LLM_USAGE_LEDGER]


def summarize_llm_usage_ledger() -> dict[str, Any]:
    """Aggregate process-local token usage and estimated cost."""

    records = get_llm_usage_ledger()
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cost_usd = 0.0
    for record in records:
        usage = record.get("usage")
        if isinstance(usage, dict):
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)
        cost_usd += float(record.get("cost_usd") or 0.0)
    return {
        "call_count": len(records),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "records": records,
    }


def _append_llm_usage_record(record: dict[str, Any]) -> None:
    with _LLM_USAGE_LEDGER_LOCK:
        _LLM_USAGE_LEDGER.append(record)


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

        response = self._post_with_retry("/chat/completions", payload)
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
        usage = _extract_usage_payload(json_payload)
        cost_usd = self._estimate_cost_usd(usage)
        if usage is not None or cost_usd is not None:
            _append_llm_usage_record(
                {
                    "model": str(payload["model"]),
                    "usage": usage,
                    "cost_usd": cost_usd or 0.0,
                    "input_cost_per_1m_tokens": self._config.input_cost_per_1m_tokens,
                    "output_cost_per_1m_tokens": self._config.output_cost_per_1m_tokens,
                }
            )

        return PackyLLMResponse(
            model=str(payload["model"]),
            text=text,
            usage=usage,
            cost_usd=cost_usd,
            raw_payload=json_payload,
            raw_text=raw_text,
            finish_reason=finish_reason,
        )

    def _post_with_retry(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        attempts = int(self._config.max_retries) + 1
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self._http.post(path, json=payload)
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt >= attempts:
                    raise PackyLLMError(
                        f"llm_transport_error after {attempt} attempt(s): {exc}"
                    ) from exc
                self._sleep_before_retry(attempt)
                continue
            if response.status_code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < attempts:
                last_exc = None
                self._sleep_before_retry(attempt)
                continue
            return response
        if last_exc is not None:
            raise PackyLLMError(f"llm_transport_error: {last_exc}") from last_exc
        raise PackyLLMError("llm_transport_error: exhausted retries without response")

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = float(self._config.retry_backoff_sec) * max(0, attempt)
        if delay > 0:
            time.sleep(delay)

    def _estimate_cost_usd(self, usage: dict[str, int] | None) -> float | None:
        if usage is None:
            return None
        input_rate = self._config.input_cost_per_1m_tokens
        output_rate = self._config.output_cost_per_1m_tokens
        if input_rate is None and output_rate is None:
            return None
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        input_cost = (prompt_tokens / 1_000_000.0) * float(input_rate or 0.0)
        output_cost = (completion_tokens / 1_000_000.0) * float(output_rate or 0.0)
        return input_cost + output_cost

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


def _extract_usage_payload(payload: dict[str, Any] | None) -> dict[str, int] | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = _coerce_token_count(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("prompt")
    )
    completion_tokens = _coerce_token_count(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("completion")
    )
    total_tokens = _coerce_token_count(usage.get("total_tokens") or usage.get("total"))
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
    normalized = {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }
    return normalized if any(normalized.values()) else None


def _coerce_token_count(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


__all__ = [
    "DEFAULT_PACKY_BASE_URL",
    "DEFAULT_PACKY_MODEL",
    "PackyLLMClient",
    "PackyLLMConfig",
    "PackyLLMError",
    "PackyLLMResponse",
    "get_llm_usage_ledger",
    "load_llm_env_file",
    "reset_llm_usage_ledger",
    "summarize_llm_usage_ledger",
    "_extract_usage_payload",
    "_extract_text_from_completion_payload",
    "_extract_text_from_sse_blob",
]
