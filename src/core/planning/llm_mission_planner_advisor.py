"""LLM advisor for graph-driven planner decisions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.agents.packy_llm import PackyLLMClient, PackyLLMError
from src.core.planning.models import PlannerOutcome


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PLANNER_GLOBAL_CONTROL_PROMPT = PROMPT_DIR / "planner_global_control.md"


class LLMMissionPlannerAdvisorConfig(BaseModel):
    """Prompt limits and model options for mission planning advice."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str | None = None
    max_context_chars: int = Field(default=24000, ge=4000)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class LLMMissionPlannerAdvisor:
    """Ask an LLM to choose the next PlannerOutcome from graph-tool context."""

    def __init__(
        self,
        *,
        client: PackyLLMClient,
        config: LLMMissionPlannerAdvisorConfig | None = None,
    ) -> None:
        self._client = client
        self._config = config or LLMMissionPlannerAdvisorConfig(
            model=getattr(client.config, "model", None)
        )

    def propose_next_decision(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any],
        recent_execution_results: list[dict[str, Any]] | None = None,
    ) -> PlannerOutcome:
        """Return the next planner outcome from the LLM."""

        operation_id = str(graph_context.get("operation_id") or "operation")
        cycle_index = int(graph_context.get("cycle_index") or graph_context.get("runtime", {}).get("cycle_index") or 0)
        prompt = self._build_decision_prompt(
            goal=goal,
            graph_context=graph_context,
            policy_context=policy_context,
            recent_execution_results=recent_execution_results,
        )
        try:
            response = self._client.complete_chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                model=self._config.model,
                temperature=self._config.temperature,
            )
        except PackyLLMError as exc:
            return _fallback_outcome(
                operation_id=operation_id,
                cycle_index=cycle_index,
                reason="llm mission planner unavailable",
                stop_condition="planner_llm_unavailable",
                metadata={"planner": "llm_mission_planner", "accepted": False, "error": str(exc)},
            )

        payload = _extract_json_object(response.text)
        if payload is None:
            return _fallback_outcome(
                operation_id=operation_id,
                cycle_index=cycle_index,
                reason="llm mission planner returned non-json",
                stop_condition="invalid_planner_json",
                metadata={"planner": "llm_mission_planner", "accepted": False, "raw_text": response.text[:1000]},
            )

        payload.setdefault("operation_id", operation_id)
        payload.setdefault("cycle_index", cycle_index)
        payload.setdefault("metadata", {})
        payload["metadata"] = {
            **dict(payload["metadata"]),
            "planner": "llm_mission_planner",
            "accepted": True,
            "model": response.model,
            "usage": response.usage,
        }
        try:
            return PlannerOutcome.model_validate(payload)
        except ValidationError as exc:
            return _fallback_outcome(
                operation_id=operation_id,
                cycle_index=cycle_index,
                reason="llm mission planner returned invalid PlannerOutcome schema",
                stop_condition="invalid_planner_schema",
                metadata={
                    "planner": "llm_mission_planner",
                    "accepted": False,
                    "error": str(exc),
                    "raw_text": response.text[:2000],
                    "normalized_payload": payload,
                },
            )

    def _build_decision_prompt(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any],
        recent_execution_results: list[dict[str, Any]] | None = None,
    ) -> str:
        outcome_contract = {
            "operation_id": "operation id string",
            "cycle_index": 0,
            "action": "execute | replan | pause_for_review | stop_success | stop_failed",
            "directive": {
                "operation_id": "operation id string",
                "cycle_index": 0,
                "capability": "recon | analysis | exploit | pivot | lateral | goal | evidence",
                "objective": "bounded one-round objective",
                "target_refs": [{"graph": "kg", "ref_id": "node id copied from min_summary", "ref_type": "Host"}],
                "allowed_tools": [],
                "tool_hints": [],
                "max_tools": 8,
                "success_hint": "what is enough for this round",
                "required_context": {},
                "risk_level": "low | medium | high | critical",
            },
            "reason": "short summary, no chain of thought",
            "stop_condition": "contract_satisfied | failure reason | null",
            "confidence": 0.8,
            "metadata": {},
        }
        context = _truncate_json(
            {
                "mission_goal": goal,
                "operation_id": graph_context.get("operation_id"),
                "cycle_index": graph_context.get("cycle_index"),
                "policy": policy_context,
                "min_summary": graph_context.get("min_summary") or {},
                "success_condition_progress": graph_context.get("success_condition_progress") or {},
                "graph_tools": graph_context.get("graph_tools") or {},
                "agent_capabilities": graph_context.get("agent_capabilities") or [],
                "mcp_tool_catalog": _slim_tool_catalog(graph_context.get("mcp_tool_catalog") or {}),
                "recent_results": list(recent_execution_results or []),
                "planner_outcome_contract": outcome_contract,
            },
            self._config.max_context_chars,
        )
        return (
            "Return strict JSON only matching PlannerOutcome. "
            "The provided graph context (min_summary, success_condition_progress, recent results) is the "
            "complete, precomputed context for this turn; decide from what is given. You cannot fetch more "
            "graph detail mid-decision. If the provided context is insufficient to act safely, choose replan. "
            "AG is a result timeline: one ATTACK_STEP per execution round plus terminal outcomes. "
            "Do not output shell commands. Do not output MCP tool arguments. "
            "Do not micro-control tool calls; output one RoundDirective with a capability and bounded objective. "
            "ExecutionAgent may autonomously choose allowed tools, including run_command, inside authorized scope. "
            "PlannerAgent is the only global controller that may output stop_success or stop_failed. "
            "Do not use a fixed stage sequence and do not require every capability to run. "
            "Select the next capability from success_condition_progress.missing, min_summary, Policy, "
            "recent results, and ToolCatalog. "
            "Treat recent ExecutionResult control hints as hard constraints: if a recent result contains "
            "next_step_guidance or capability guidance, "
            "follow it unless it conflicts with Policy or newer evidence. If a recent result contains "
            "supported_bounded_validation_candidate=false, do not choose capability=exploit for that target. "
            "target_refs MUST be a list of objects {\"graph\":\"kg\"|\"ag\", \"ref_id\":\"<exact node id from min_summary>\"}, "
            "never bare id strings; use [] when no specific node applies. Plain ids may go in required_context. "
            "You may emit planner_tool_calls ONLY for write-level judgment records "
            "(record_finding/record_attack_step/link_evidence); these are advisory and do not change this "
            "turn's decision. Machine facts from tools are written deterministically after execution. "
            "If evidence is insufficient, choose action=execute with an appropriate capability or choose replan. "
            "If policy does not allow the next action, choose pause_for_review. "
            "Never select a target whose host appears in policy.blocked_hosts; treat those hosts as strictly out of scope "
            "(they are control-plane infrastructure, not assessment targets). "
            "A blocked_host / target_out_of_scope rejection means ONLY that one host is off-limits, NOT that the chain is "
            "dead: retarget the in-scope assessment hosts (e.g. the discovered entry-zone services) and keep executing. "
            "Do NOT pause_for_review merely because a call against a blocked host was rejected — pause only when no in-scope "
            "target or action remains. "
            "If success_condition_progress.eligible_for_stop=true, choose action=stop_success with stop_condition=contract_satisfied. "
            "For action=execute, directive must be non-null. For stop/replan/pause actions, directive must be null. "
            "Use reason for a concise justification without chain-of-thought.\n\n"
            f"{context}"
        )

def _load_planner_global_control_prompt() -> str:
    return PLANNER_GLOBAL_CONTROL_PROMPT.read_text(encoding="utf-8")


SYSTEM_PROMPT = _load_planner_global_control_prompt()


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


def _slim_tool_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    """Project the MCP catalog to tool names + short descriptions for the planner.

    The planner only selects a capability and is told never to emit tool
    arguments, so the per-tool inputSchemas (the bulk of the catalog, ~3.6k
    tokens) are dead weight in its prompt. Keep name/description/availability and
    drop the schemas; the executor still receives the full catalog.
    """

    if not isinstance(catalog, dict):
        return {}
    slim: dict[str, Any] = {}
    for server_id, server in catalog.items():
        if not isinstance(server, dict):
            continue
        slim_tools: list[dict[str, Any]] = []
        for tool in server.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            entry: dict[str, Any] = {"name": tool.get("name")}
            if tool.get("description"):
                entry["description"] = tool.get("description")
            if tool.get("available") is False:
                entry["available"] = False
            slim_tools.append(entry)
        slim[server_id] = {"tools": slim_tools}
    return slim


def _truncate_json(payload: dict[str, Any], max_chars: int) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80] + "...<truncated>"


def _fallback_outcome(
    *,
    operation_id: str,
    cycle_index: int,
    reason: str,
    stop_condition: str | None,
    metadata: dict[str, Any],
) -> PlannerOutcome:
    return PlannerOutcome(
        operation_id=operation_id,
        cycle_index=cycle_index,
        action="replan",
        directive=None,
        reason=reason,
        stop_condition=stop_condition,
        confidence=0.0,
        metadata=metadata,
    )


__all__ = ["LLMMissionPlannerAdvisor", "LLMMissionPlannerAdvisorConfig"]
