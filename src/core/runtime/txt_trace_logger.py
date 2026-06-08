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
    "apikey",
    "secret",
    "cookie",
    "authorization",
    "auth",
    "private_key",
    "privatekey",
}


class TxtTraceLogger:
    """Write compact, human-readable operation trace blocks."""

    def __init__(
        self,
        operation_id: str,
        log_dir: str | Path = "runs",
        *,
        filename: str | None = None,
        operation_subdir: bool = False,
    ) -> None:
        self.operation_id = operation_id
        base = Path(log_dir) / operation_id if operation_subdir else Path(log_dir)
        self.path = base / (filename or f"{operation_id}.run.txt")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def operation_trace(cls, operation_id: str, runtime_root: str | Path = "var/runtime") -> "TxtTraceLogger":
        """Return the canonical human-readable operation trace logger."""

        return cls(
            operation_id,
            log_dir=runtime_root,
            filename="operation-trace.txt",
            operation_subdir=True,
        )

    def write_header(self, title: str, data: dict[str, Any] | None = None) -> None:
        payload = self._redact(data or {})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
            handle.write("=" * 60 + "\n")
            handle.write(f"{title}\n")
            for key, value in payload.items():
                value = self._json_safe(value)
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                handle.write(f"{key}: {value}\n")
            handle.write("=" * 60 + "\n")

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
                if cls._is_sensitive_key(key):
                    redacted[key] = "[REDACTED]"
                else:
                    redacted[key] = cls._redact(item)
            return redacted
        if isinstance(value, list):
            return [cls._redact(item) for item in value]
        return value

    @staticmethod
    def _is_sensitive_key(key: Any) -> bool:
        normalized = str(key).lower().replace("-", "_")
        return any(part in normalized for part in SENSITIVE_KEYS)

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
