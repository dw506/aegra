from __future__ import annotations

import httpx
import pytest

from src.core.agents.packy_llm import (
    PackyLLMClient,
    PackyLLMConfig,
    _extract_text_from_completion_payload,
    _extract_text_from_sse_blob,
)


def test_extract_text_from_completion_payload_reads_standard_message_content() -> None:
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "hello from packy",
                },
            }
        ]
    }

    text, finish_reason = _extract_text_from_completion_payload(payload)

    assert text == "hello from packy"
    assert finish_reason == "stop"


def test_extract_text_from_sse_blob_concatenates_chunk_content() -> None:
    raw_text = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null,"index":0}]}',
            'data: {"choices":[{"delta":{"content":" there"},"finish_reason":null,"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
            "data: [DONE]",
        ]
    )

    text, finish_reason = _extract_text_from_sse_blob(raw_text)

    assert text == "Hi there"
    assert finish_reason == "stop"


def test_packy_client_falls_back_to_sse_text_when_gateway_returns_stream_blob() -> None:
    raw_sse = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"Packy"},"finish_reason":null,"index":0}]}',
            'data: {"choices":[{"delta":{"content":" works"},"finish_reason":null,"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
            "data: [DONE]",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            status_code=200,
            text=raw_sse,
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url="https://www.packyapi.com/v1", transport=transport) as http_client:
        client = PackyLLMClient(
            PackyLLMConfig(api_key="test-key", base_url="https://www.packyapi.com/v1", model="gpt-5.2"),
            http_client=http_client,
        )
        response = client.complete_chat(user_prompt="hello")

    assert response.text == "Packy works"
    assert response.finish_reason == "stop"
    assert response.model == "gpt-5.2"


def test_packy_llm_config_from_env_prefers_aegra_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGRA_LLM_API_KEY", "aegra-key")
    monkeypatch.setenv("AEGRA_LLM_BASE_URL", "https://aegra.example/v1")
    monkeypatch.setenv("AEGRA_LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("AEGRA_LLM_TIMEOUT_SEC", "45")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example/v1")

    config = PackyLLMConfig.from_env()

    assert config.api_key == "aegra-key"
    assert config.base_url == "https://aegra.example/v1"
    assert config.model == "gpt-5.4"
    assert config.timeout_sec == 45.0


def test_packy_llm_config_from_env_falls_back_to_openai_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AEGRA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("AEGRA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("AEGRA_LLM_MODEL", raising=False)
    monkeypatch.delenv("AEGRA_LLM_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example/v1")

    config = PackyLLMConfig.from_env()

    assert config.api_key == "openai-key"
    assert config.base_url == "https://openai.example/v1"
    assert config.model == "gpt-5.2"
