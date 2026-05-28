"""LLM advisor for bounded Stage Agent decisions."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.agents.packy_llm import PackyLLMClient, PackyLLMError
from src.core.stage.base_stage_agent import StageAgentDecision
from src.core.stage.models import StageTask, StageType


class LLMStageAdvisorConfig(BaseModel):
    """Prompt limits and model options for stage execution advice."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str | None = None
    max_context_chars: int = Field(default=28000, ge=4000)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class LLMStageAdvisor:
    """Let Stage Agents reason over graph/runtime context and available tools."""

    def __init__(
        self,
        *,
        client: PackyLLMClient,
        config: LLMStageAdvisorConfig | None = None,
    ) -> None:
        self._client = client
        self._config = config or LLMStageAdvisorConfig(model=getattr(client.config, "model", None))

    def decide(
        self,
        *,
        agent_name: str,
        stage_type: StageType,
        task: StageTask,
        graph_context: dict[str, Any],
        runtime_context: dict[str, Any],
        memory: list[dict[str, Any]],
        available_tools: dict[str, Any],
    ) -> StageAgentDecision | dict[str, Any]:
        prompt = self._build_prompt(
            agent_name=agent_name,
            stage_type=stage_type,
            task=task,
            graph_context=graph_context,
            runtime_context=runtime_context,
            memory=memory,
            available_tools=available_tools,
        )
        try:
            response = self._client.complete_chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                model=self._config.model,
                temperature=self._config.temperature,
            )
        except PackyLLMError as exc:
            return StageAgentDecision(
                action="need_replan",
                rationale=f"llm stage advisor unavailable: {exc}",
            )
        payload = _extract_json_object(response.text)
        if payload is None:
            return StageAgentDecision(
                action="need_replan",
                rationale="llm stage advisor returned non-json decision",
            )
        try:
            return StageAgentDecision.model_validate(payload)
        except ValidationError as exc:
            return StageAgentDecision(
                action="need_replan",
                rationale=f"llm stage advisor returned invalid decision schema: {exc}",
            )

    def _build_prompt(
        self,
        *,
        agent_name: str,
        stage_type: StageType,
        task: StageTask,
        graph_context: dict[str, Any],
        runtime_context: dict[str, Any],
        memory: list[dict[str, Any]],
        available_tools: dict[str, Any],
    ) -> str:
        payload = {
            "agent_name": agent_name,
            "stage_type": stage_type.value,
            "task": task.model_dump(mode="json"),
            "graph_context": graph_context,
            "runtime_context": runtime_context,
            "memory": memory[-8:],
            "available_tools": available_tools,
            "decision_schema": {
                "action": "call_tool | finish | need_replan",
                "rationale": "short reason",
                "tool_call": {
                    "server_id": "required only for call_tool",
                    "tool_name": "required only for call_tool",
                    "arguments": {},
                    "timeout_seconds": 60,
                },
                "finish": {
                    "status": "succeeded | partial | failed | needs_replan",
                    "summary": "stage outcome",
                    "observations": [],
                    "evidence": [],
                    "findings": [],
                    "discovered_entities": [],
                    "discovered_relations": [],
                    "capabilities_gained": [],
                    "credentials": [],
                    "sessions": [],
                    "pivot_routes": [],
                    "next_stage_candidates": [],
                    "runtime_hints": {},
                    "writeback_hints": {},
                },
            },
        }
        return (
            "Return only JSON matching StageAgentDecision. "
            "Select one safe next step using the task, graph memory, runtime state and tool catalog. "
            "When evidence is sufficient, finish with structured observations and writeback data.\n\n"
            f"{_truncate_json(payload, self._config.max_context_chars)}"
        )


SYSTEM_PROMPT = (
    "You are an Aegra Stage Agent advisor. You may reason about authorized "
    "penetration-testing tasks, but you must only choose registered tools from "
    "the supplied catalog. Do not invent shell commands, exploit payload source, "
    "or direct KG/AG/TG/runtime mutations. StageResult writeback happens through "
    "the structured finish fields."
)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(1))
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _truncate_json(payload: dict[str, Any], max_chars: int) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80] + "...<truncated>"


__all__ = ["LLMStageAdvisor", "LLMStageAdvisorConfig"]
