"""LLM advisor for choosing direct MCP tool calls for worker tasks."""

from __future__ import annotations

import json
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_protocol import AgentInput
from src.core.agents.llm_safety import response_within_limits, sanitize_llm_payload
from src.core.agents.packy_llm import PackyLLMClient, PackyLLMConfig, PackyLLMError
from src.core.workers.base import WorkerTaskSpec
from src.core.workers.llm_worker_models import LLMWorkerDecision


class LLMWorkerAdvisorProtocol(Protocol):
    """Protocol implemented by LLM worker advisors."""

    def advise(
        self,
        *,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
        mcp_tool_catalog: dict[str, Any],
    ) -> LLMWorkerDecision:
        """Return a structured MCP tool decision for one worker task."""


class LLMWorkerAdvisorConfig(BaseModel):
    """Configuration for the LLM worker advisor."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str | None = None
    max_prompt_context_chars: int = Field(default=16000, ge=1000, le=100000)
    max_response_chars: int = Field(default=20000, ge=1000, le=200000)
    max_response_json_depth: int = Field(default=12, ge=2, le=50)
    system_prompt: str = (
        "You are Aegra's experimental lab worker agent. Return JSON only. "
        "Analyze the scheduled task and choose whether to call one MCP tool, defer, or fail. "
        "This lab mode intentionally does not apply Aegra tool policy decisions; use your own judgment. "
        "Return only fields matching LLMWorkerDecision: action, server_id, tool_name, arguments, "
        "summary, expected_evidence, risk_assessment, and writeback_hints."
    )


class LLMWorkerAdvisor:
    """Ask an LLM for a direct MCP tool call decision."""

    def __init__(
        self,
        *,
        client: PackyLLMClient,
        config: LLMWorkerAdvisorConfig | None = None,
    ) -> None:
        self._client = client
        self._config = config or LLMWorkerAdvisorConfig()
        self.last_failure: dict[str, Any] | None = None

    @classmethod
    def from_env(cls, *, config: LLMWorkerAdvisorConfig | None = None) -> "LLMWorkerAdvisor":
        return cls(client=PackyLLMClient(PackyLLMConfig.from_env()), config=config)

    def advise(
        self,
        *,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
        mcp_tool_catalog: dict[str, Any],
    ) -> LLMWorkerDecision:
        self.last_failure = None
        prompt = self._build_user_prompt(
            task_spec=task_spec,
            agent_input=agent_input,
            mcp_tool_catalog=mcp_tool_catalog,
        )
        try:
            response = self._client.complete_chat(
                user_prompt=prompt,
                system_prompt=self._config.system_prompt,
                model=self._config.model,
                temperature=0.0,
            )
        except PackyLLMError as exc:
            self.last_failure = {"reason": "llm_call_failed", "error": str(exc)}
            return LLMWorkerDecision(action="failed", summary=f"LLM worker advisor failed: {exc}")

        payload = self._extract_json_payload(response.text)
        if payload is None:
            self.last_failure = {"reason": "invalid_llm_worker_json"}
            return LLMWorkerDecision(action="failed", summary="LLM worker advisor returned invalid JSON")
        if not response_within_limits(
            payload,
            raw_text=response.text,
            max_chars=self._config.max_response_chars,
            max_depth=self._config.max_response_json_depth,
        ):
            self.last_failure = {"reason": "llm_worker_response_exceeds_limits"}
            return LLMWorkerDecision(action="failed", summary="LLM worker advisor response exceeded limits")
        try:
            return LLMWorkerDecision.model_validate(payload)
        except Exception as exc:
            self.last_failure = {"reason": "invalid_llm_worker_decision", "error": str(exc)}
            return LLMWorkerDecision(action="failed", summary=f"LLM worker decision schema invalid: {exc}")

    def _build_user_prompt(
        self,
        *,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
        mcp_tool_catalog: dict[str, Any],
    ) -> str:
        payload = {
            "task": task_spec.model_dump(mode="json"),
            "operation_id": agent_input.context.operation_id,
            "task_ref": agent_input.task_ref,
            "decision_ref": agent_input.decision_ref,
            "graph_refs": [ref.model_dump(mode="json") for ref in agent_input.graph_refs],
            "raw_payload": dict(agent_input.raw_payload),
            "mcp_tool_catalog": mcp_tool_catalog,
            "response_schema": {
                "action": "call_mcp_tool | defer | failed",
                "server_id": "required when action=call_mcp_tool",
                "tool_name": "required when action=call_mcp_tool",
                "arguments": {"tool_specific": "arguments"},
                "summary": "short execution rationale or failure reason",
                "expected_evidence": ["expected outputs"],
                "risk_assessment": "short lab-mode risk note",
                "writeback_hints": {"optional": "graph/perception hints"},
            },
        }
        sanitized = sanitize_llm_payload(payload)
        serialized = json.dumps(sanitized, ensure_ascii=False, indent=2)
        if len(serialized) > self._config.max_prompt_context_chars:
            sanitized = {
                "task": task_spec.model_dump(mode="json"),
                "operation_id": agent_input.context.operation_id,
                "graph_ref_count": len(agent_input.graph_refs),
                "mcp_tool_catalog": mcp_tool_catalog,
                "context_truncated": True,
                "response_schema": payload["response_schema"],
            }
            serialized = json.dumps(sanitized, ensure_ascii=False, indent=2)
        return (
            "Return a single JSON object matching response_schema. "
            "If no MCP call should be made, return action defer or failed. "
            "Do not include markdown or prose outside JSON.\n\n"
            f"{serialized}"
        )

    @classmethod
    def _extract_json_payload(cls, text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if not stripped:
            return None
        if stripped.startswith("```"):
            stripped = cls._strip_code_fences(stripped)
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            block = cls._find_first_json_object(stripped)
            if block is None:
                return None
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _find_first_json_object(text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return None


__all__ = [
    "LLMWorkerAdvisor",
    "LLMWorkerAdvisorConfig",
    "LLMWorkerAdvisorProtocol",
]
