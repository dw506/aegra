"""LLM advisor for bounded Stage Agent decisions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.agents.packy_llm import PackyLLMClient, PackyLLMError
from src.core.stage.base_stage_agent import StageAgentDecision
from src.core.stage.models import StageExecutionRequest, StageTask, StageType


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_BY_AGENT: dict[str, str] = {
    "recon_agent": "recon_agent.md",
    "vuln_analysis_agent": "vuln_analysis_agent.md",
    "exploit_agent": "exploit_validation_agent.md",
    "exploit_validation_agent": "exploit_validation_agent.md",
    "access_pivot_agent": "access_pivot_agent.md",
    "goal_agent": "goal_agent.md",
}
PROMPT_BY_STAGE: dict[StageType, str] = {
    StageType.RECON: "recon_agent.md",
    StageType.RECON_STAGE: "recon_agent.md",
    StageType.VULN_ANALYSIS: "vuln_analysis_agent.md",
    StageType.VULN_ANALYSIS_STAGE: "vuln_analysis_agent.md",
    StageType.EXPLOIT: "exploit_validation_agent.md",
    StageType.EXPLOIT_STAGE: "exploit_validation_agent.md",
    StageType.ACCESS_PIVOT: "access_pivot_agent.md",
    StageType.ACCESS_PIVOT_STAGE: "access_pivot_agent.md",
    StageType.GOAL: "goal_agent.md",
    StageType.GOAL_STAGE: "goal_agent.md",
}


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
        request: StageExecutionRequest | None = None,
        task: StageTask | None = None,
        graph_context: dict[str, Any],
        runtime_context: dict[str, Any],
        memory: list[dict[str, Any]],
        available_tools: dict[str, Any],
        policy_context: dict[str, Any] | None = None,
    ) -> StageAgentDecision | dict[str, Any]:
        prompt = self._build_prompt(
            agent_name=agent_name,
            stage_type=stage_type,
            request=request,
            task=task,
            graph_context=graph_context,
            runtime_context=runtime_context,
            policy_context=dict(policy_context or {}),
            memory=memory,
            available_tools=available_tools,
        )
        system_prompt = self._build_system_prompt(agent_name=agent_name, stage_type=stage_type)
        try:
            response = self._client.complete_chat(
                system_prompt=system_prompt,
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
            repaired = self._repair_decision(
                original_text=response.text,
                prompt=prompt,
            )
            if repaired is not None:
                return repaired
            return StageAgentDecision(action="need_replan", rationale="llm stage advisor returned non-json decision")
        try:
            return StageAgentDecision.model_validate(payload)
        except ValidationError as exc:
            repaired = self._repair_decision(
                original_text=response.text,
                prompt=prompt,
                validation_error=str(exc),
            )
            if repaired is not None:
                return repaired
            return StageAgentDecision(
                action="need_replan",
                rationale=f"llm stage advisor returned invalid decision schema: {exc}",
            )

    def _build_prompt(
        self,
        *,
        agent_name: str,
        stage_type: StageType,
        request: StageExecutionRequest | None = None,
        task: StageTask | None = None,
        graph_context: dict[str, Any],
        runtime_context: dict[str, Any],
        policy_context: dict[str, Any],
        memory: list[dict[str, Any]],
        available_tools: dict[str, Any],
    ) -> str:
        request_payload = request.model_dump(mode="json") if request is not None else None
        task_payload = task.model_dump(mode="json") if task is not None else None
        payload = {
            "agent_name": agent_name,
            "stage_type": stage_type.value,
            "stage_execution_request": request_payload,
            "legacy_task": task_payload,
            "graph_state_snapshot": graph_context,
            "runtime_context": runtime_context,
            "policy_context": policy_context,
            "memory": memory[-8:],
            "mcp_tool_catalog": available_tools,
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
                    "graph_update_intents": [],
                    "evidence_refs": [],
                    "confidence": 0.0,
                    "risk_level": "low | medium | high | critical",
                    "policy_notes": [],
                    "retry_recommendation": "optional",
                    "replan_recommendation": "optional",
                    "next_stage_suggestion": {},
                    "handoff_suggestion": {
                        "suggested_agent": "optional next agent",
                        "suggested_stage": "optional next stage",
                        "reason": "why handoff is appropriate",
                        "confidence": 0.0,
                        "required_context_refs": [],
                    },
                },
            },
        }
        return (
            "Return only JSON matching StageAgentDecision. "
            "You are the complete LLM decision maker for this stage. Decide whether the stage can run, "
            "which MCP tool to call if any, why, and how prior tool results affect completion. "
            "Use only supplied graph state, policy, stage execution request, tool catalog and memory. "
            "Do not invent scan results, vulnerabilities, credentials, sessions or access. "
            "Do not write KG/AG/Runtime directly; propose intents and handoff_suggestion in the finish payload. "
            "If evidence is insufficient, finish with need_more_info/partial or request need_replan.\n\n"
            f"{_truncate_json(payload, self._config.max_context_chars)}"
        )

    def _build_system_prompt(self, *, agent_name: str, stage_type: StageType) -> str:
        return f"{COMMON_SYSTEM_RULES}\n\n{_load_agent_prompt(agent_name=agent_name, stage_type=stage_type)}"

    def _repair_decision(
        self,
        *,
        original_text: str,
        prompt: str,
        validation_error: str | None = None,
    ) -> StageAgentDecision | None:
        repair_prompt = (
            "Repair the previous response into strict StageAgentDecision JSON only. "
            "Do not add new facts. Preserve the intended action when possible.\n\n"
            f"Validation error: {validation_error or 'not valid JSON'}\n\n"
            f"Original response:\n{original_text[:4000]}\n\n"
            f"Original prompt excerpt:\n{prompt[:4000]}"
        )
        try:
            response = self._client.complete_chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                model=self._config.model,
                temperature=0.0,
            )
        except PackyLLMError:
            return None
        payload = _extract_json_object(response.text)
        if payload is None:
            return None
        try:
            return StageAgentDecision.model_validate(payload)
        except ValidationError:
            return None


SYSTEM_PROMPT = (
    "Deprecated compatibility alias. Use COMMON_SYSTEM_RULES plus the agent-specific prompt."
)


COMMON_SYSTEM_RULES = (
    "You are an Aegra LLM Stage Agent, not a tool wrapper. "
    "Return only StageAgentDecision JSON. "
    "Choose only one action: call_tool, finish, or need_replan. "
    "Call only tools present in mcp_tool_catalog. "
    "Do not output shell commands. Do not invent shell commands. "
    "Do not invent environment facts, vulnerabilities, credentials, or sessions. "
    "Do not directly write KG or AG. "
    "KG fact intents and AG process intents are only suggestions inside the StageResult finish payload; "
    "ResultApplier writes them."
)


def _load_agent_prompt(*, agent_name: str, stage_type: StageType) -> str:
    filename = PROMPT_BY_AGENT.get(agent_name) or PROMPT_BY_STAGE.get(stage_type)
    if filename is None:
        filename = PROMPT_BY_STAGE.get(stage_type.canonical, "recon_agent.md")
    return (PROMPT_DIR / filename).read_text(encoding="utf-8")


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
