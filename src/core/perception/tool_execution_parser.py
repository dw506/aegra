"""Parser for adapter-neutral tool execution results."""

from __future__ import annotations

from typing import Any

from src.core.agents.agent_models import OutcomeRecord
from src.core.perception.parser_protocol import ParsedWorkerResult


class ToolExecutionParser:
    """Parse ToolExecutionResult payloads into perception records."""

    name = "tool_execution_parser"
    _EXCERPT_LIMIT = 500

    def supports(self, raw_result: dict[str, Any], outcome: OutcomeRecord) -> bool:
        return self._tool_execution(raw_result, outcome) is not None

    def parse(self, raw_result: dict[str, Any], outcome: OutcomeRecord) -> ParsedWorkerResult:
        tool_execution = self._tool_execution(raw_result, outcome) or {}
        adapter = self._string(tool_execution.get("adapter"), "unknown_adapter")
        tool = self._string(tool_execution.get("tool"), "unknown_tool")
        success = bool(tool_execution.get("success", False))
        exit_code = tool_execution.get("exit_code")
        command_id = self._optional_string(tool_execution.get("command_id"))
        payload_ref = self._optional_string(tool_execution.get("payload_ref"))
        stdout = self._string(tool_execution.get("stdout"), "")
        stderr = self._string(tool_execution.get("stderr"), "")
        summary = f"Tool {tool} executed via {adapter}"
        parsed_payload = self._parsed_payload(raw_result, outcome)
        observation_payload = {
            "category": "tool_execution",
            "adapter": adapter,
            "tool": tool,
            "success": success,
            "exit_code": exit_code,
            "command_id": command_id,
            "stdout_excerpt": self._excerpt(stdout),
            "stderr_excerpt": self._excerpt(stderr),
        }
        if parsed_payload:
            observation_payload["parsed"] = parsed_payload
            observation_payload["entities"] = self._list(parsed_payload.get("entities"))
            observation_payload["relations"] = self._list(parsed_payload.get("relations"))
            observation_payload["findings"] = self._list(parsed_payload.get("findings"))
            observation_payload["runtime_hints"] = self._dict(parsed_payload.get("runtime_hints"))
            observation_payload["writeback_hints"] = self._dict(parsed_payload.get("writeback_hints"))

        return ParsedWorkerResult(
            observations=[
                {
                    "summary": summary,
                    "confidence": outcome.confidence,
                    "payload": observation_payload,
                }
            ],
            evidence=[
                {
                    "summary": f"Tool execution evidence for {tool}",
                    "confidence": outcome.confidence,
                    "payload_ref": payload_ref,
                    "payload": {
                        "kind": "tool_execution_evidence",
                        "adapter": adapter,
                        "tool": tool,
                        "command_id": command_id,
                        "success": success,
                        "exit_code": exit_code,
                    },
                }
            ],
            findings=self._list(parsed_payload.get("findings")) if parsed_payload else [],
            metadata={"parser": self.name, "parsed": parsed_payload} if parsed_payload else {"parser": self.name},
        )

    def _tool_execution(self, raw_result: dict[str, Any], outcome: OutcomeRecord) -> dict[str, Any] | None:
        candidates = [
            raw_result.get("tool_execution"),
            outcome.payload.get("tool_execution"),
        ]
        extra = raw_result.get("extra")
        if isinstance(extra, dict):
            candidates.append(extra.get("tool_execution"))
        for candidate in candidates:
            if isinstance(candidate, dict):
                return dict(candidate)
        return None

    def _parsed_payload(self, raw_result: dict[str, Any], outcome: OutcomeRecord) -> dict[str, Any]:
        candidates = [
            raw_result.get("parsed"),
            outcome.payload.get("parsed"),
        ]
        extra = raw_result.get("extra")
        if isinstance(extra, dict):
            candidates.append(extra.get("parsed"))
            mcp_payload = extra.get("mcp_payload")
            if isinstance(mcp_payload, dict):
                candidates.append(mcp_payload.get("parsed"))
        mcp_payload = raw_result.get("mcp_payload")
        if isinstance(mcp_payload, dict):
            candidates.append(mcp_payload.get("parsed"))
        for candidate in candidates:
            if isinstance(candidate, dict):
                return dict(candidate)
        return {}

    def _excerpt(self, value: str) -> str:
        return value[: self._EXCERPT_LIMIT]

    @staticmethod
    def _string(value: Any, default: str) -> str:
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _list(value: Any) -> list[Any]:
        return list(value) if isinstance(value, list) else []

    @staticmethod
    def _dict(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}


__all__ = ["ToolExecutionParser"]
