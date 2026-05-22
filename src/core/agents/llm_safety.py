"""Small safety helpers for LLM prompt and response handling."""

from __future__ import annotations

from typing import Any


SENSITIVE_LLM_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
REDACTED_VALUE = "[REDACTED]"


def sanitize_llm_payload(value: Any) -> Any:
    """Return a JSON-compatible copy with sensitive fields redacted."""

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(str(key)):
                sanitized[key] = REDACTED_VALUE
            else:
                sanitized[key] = sanitize_llm_payload(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_llm_payload(item) for item in value]
    return value


def json_depth(value: Any) -> int:
    """Return the maximum container nesting depth for a parsed JSON value."""

    if isinstance(value, dict):
        if not value:
            return 1
        return 1 + max(json_depth(item) for item in value.values())
    if isinstance(value, list):
        if not value:
            return 1
        return 1 + max(json_depth(item) for item in value)
    return 0


def response_within_limits(value: Any, *, raw_text: str, max_chars: int, max_depth: int) -> bool:
    """Validate coarse LLM response size before model-specific schema parsing."""

    return len(raw_text) <= max_chars and json_depth(value) <= max_depth


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in SENSITIVE_LLM_KEY_MARKERS)


__all__ = [
    "REDACTED_VALUE",
    "SENSITIVE_LLM_KEY_MARKERS",
    "json_depth",
    "response_within_limits",
    "sanitize_llm_payload",
]
