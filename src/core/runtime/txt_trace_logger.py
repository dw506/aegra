"""Append-only categorized text trace for operation runs."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any


SENSITIVE_KEYS = {
    "password",
    "token",
    "api_key",
    "secret",
    "cookie",
    "authorization",
    "private_key",
}


class TxtTraceLogger:
    """Write compact, human-readable operation trace blocks."""

    def __init__(self, operation_id: str, log_dir: str | Path = "runs") -> None:
        self.operation_id = operation_id
        self.path = Path(log_dir) / f"{operation_id}.run.txt"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, category: str, message: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{self._now()} 【{category}】 {message}\n")

    def write_block(self, category: str, title: str, data: dict[str, Any] | None = None) -> None:
        payload = self._redact(data or {})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
            handle.write(f"{self._now()} 【{category}】 {title}\n")
            for key, value in payload.items():
                value = self._json_safe(value)
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                handle.write(f"  {key}: {value}\n")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _redact(cls, value: Any) -> Any:
        if isinstance(value, dict):
            redacted: dict[Any, Any] = {}
            for key, item in value.items():
                if str(key).lower() in SENSITIVE_KEYS:
                    redacted[key] = "[REDACTED]"
                else:
                    redacted[key] = cls._redact(item)
            return redacted
        if isinstance(value, list):
            return [cls._redact(item) for item in value]
        return value

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return cls._json_safe(value.model_dump(mode="json"))
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, (datetime, date, Path)):
            return str(value)
        return value


__all__ = ["TxtTraceLogger"]
