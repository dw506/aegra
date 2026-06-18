"""LLM advisor for graph-driven planner decisions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.agents.packy_llm import PackyLLMClient, PackyLLMError
from src.core.planning.models import PlannerDecision, PlannerOutcome
from src.core.stage.models import normalize_stage_name


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
        recent_stage_results: list[dict[str, Any]] | None = None,
    ) -> PlannerOutcome | PlannerDecision | dict[str, Any]:
        """Return the next planner outcome from the LLM."""

        operation_id = str(graph_context.get("operation_id") or "operation")
        cycle_index = int(graph_context.get("cycle_index") or graph_context.get("runtime", {}).get("cycle_index") or 0)
        prompt = self._build_decision_prompt(
            goal=goal,
            graph_context=graph_context,
            policy_context=policy_context,
            recent_stage_results=recent_stage_results,
        )
        try:
            response = self._client.complete_chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                model=self._config.model,
                temperature=self._config.temperature,
            )
        except PackyLLMError as exc:
            return _fallback_decision(
                operation_id=operation_id,
                cycle_index=cycle_index,
                goal=goal,
                decision="replan",
                reason="llm mission planner unavailable",
                stop_condition="planner_llm_unavailable",
                metadata={"planner": "llm_mission_planner", "accepted": False, "error": str(exc)},
            )

        payload = _extract_json_object(response.text)
        if payload is None:
            return _fallback_decision(
                operation_id=operation_id,
                cycle_index=cycle_index,
                goal=goal,
                decision="replan",
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
        if payload.get("action") is not None:
            try:
                return PlannerOutcome.model_validate(payload)
            except ValidationError as exc:
                return _fallback_decision(
                    operation_id=operation_id,
                    cycle_index=cycle_index,
                    goal=goal,
                    decision="replan",
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
        payload = _normalize_decision_payload(payload, goal=goal)
        try:
            return PlannerDecision.model_validate(payload)
        except ValidationError as exc:
            return _fallback_decision(
                operation_id=operation_id,
                cycle_index=cycle_index,
                goal=goal,
                decision="replan",
                reason="llm mission planner returned invalid PlannerDecision schema",
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
        recent_stage_results: list[dict[str, Any]] | None = None,
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
                "target_refs": [],
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
                "kg": graph_context.get("kg") or graph_context.get("kg_summary") or {},
                "ag_process": graph_context.get("ag_process_summary") or graph_context.get("ag") or graph_context.get("ag_summary") or {},
                "runtime": graph_context.get("runtime") or graph_context.get("runtime_summary") or {},
                "policy": policy_context,
                "recent_evidence": graph_context.get("recent_evidence") or [],
                "known_assets": graph_context.get("known_assets") or [],
                "known_services": graph_context.get("known_services") or [],
                "zone_objectives": graph_context.get("zone_objectives") or [],
                "pivot_candidates": graph_context.get("pivot_candidates") or [],
                "self_infrastructure_addresses": graph_context.get("self_infrastructure_addresses") or [],
                "active_sessions": graph_context.get("active_sessions") or [],
                "recent_attack_process_nodes": graph_context.get("recent_attack_process_nodes") or [],
                "recent_handoff_suggestions": graph_context.get("recent_handoff_suggestions") or [],
                "recent_failures": graph_context.get("recent_failures") or [],
                "validation_dead_ends": graph_context.get("validation_dead_ends") or [],
                "current_goal": graph_context.get("current_goal") or goal,
                "recent_results": list(recent_stage_results or []),
                "planner_outcome_contract": outcome_contract,
                "min_summary": graph_context.get("min_summary") or {},
                "graph_tools": graph_context.get("graph_tools") or {},
                "agent_capabilities": graph_context.get("agent_capabilities") or [],
                "mcp_tool_capabilities": graph_context.get("mcp_tool_capabilities") or graph_context.get("mcp_tool_catalog") or {},
            },
            self._config.max_context_chars,
        )
        return (
            "Return strict JSON only matching PlannerOutcome. "
            "Use the supplied graph tool summaries and available graph_tools before choosing. "
            "AG is a result timeline: one ATTACK_STEP per execution round plus terminal outcomes. "
            "Do not output shell commands. Do not output MCP tool arguments. "
            "Do not micro-control tool calls; output one RoundDirective with a capability and bounded objective. "
            "ExecutionAgent may autonomously choose allowed tools, including run_command, inside authorized scope. "
            "PlannerAgent is the only global controller that may output stop_success or stop_failed. "
            "Do not use a fixed stage sequence and do not require every capability to run. "
            "Select the next capability from success_condition_progress.missing, KG, AG timeline, Runtime, Policy and ToolCatalog. "
            "Treat recent RoundResult/StageResult control hints as hard constraints: if a recent result contains "
            "next_step_guidance or capability guidance, "
            "follow it unless it conflicts with Policy or newer evidence. If a recent result contains "
            "supported_bounded_validation_candidate=false, do not choose capability=exploit for that target. "
            "If Runtime validation_dead_ends contains validation_dead_end(target=T, reason=no_supported_profile), "
            "do not choose capability=exploit for target T unless required_context includes a new_profile, "
            "profile_id, matched_profile_id, validation_profile, or new_evidence. "
            "When validation is blocked by no_supported_profile, prefer capability=recon to fingerprint a service that "
            "has not yet been fingerprinted; do NOT repeatedly re-fingerprint a service that already has product/"
            "version evidence — once fingerprinted, move toward validation or pivot. "
            "Read zone_objectives: for any zone with directly_reachable=false and reached=false, the mission is not "
            "complete and recon alone will never reach it. To make progress on such a zone you must obtain a "
            "capability/session and an authorized pivot route. When a zone like that is unreached, prioritize "
            "obtaining a session on a host in pivot_candidates (these expose transport-capable services and are "
            "likely bridges toward the unreached zone), then choose capability=pivot to enumerate routes from that "
            "session, register a pivot route, and discover restricted-zone services — rather than looping recon. "
            "Treat any address in self_infrastructure_addresses as the operation's own control plane / tooling: it "
            "is NOT an attack surface — never select it as a target, fingerprint it, or analyze it for vulnerabilities. "
            "Use graph_tools.write only for planner judgment-level records such as record_finding/link_evidence; "
            "machine facts from tools are written deterministically after execution. "
            "If evidence is insufficient, choose action=execute with an appropriate capability or choose replan. "
            "If policy does not allow the next action, choose pause_for_review. "
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


def _truncate_json(payload: dict[str, Any], max_chars: int) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80] + "...<truncated>"


def _fallback_decision(
    *,
    operation_id: str,
    cycle_index: int,
    goal: str,
    decision: str,
    reason: str,
    stop_condition: str | None,
    metadata: dict[str, Any],
) -> PlannerDecision:
    return PlannerDecision(
        operation_id=operation_id,
        cycle_index=cycle_index,
        decision=decision,  # type: ignore[arg-type]
        selected_agent=None,
        selected_stage=None,
        objective=goal,
        target_refs=[],
        required_context={},
        success_criteria=[],
        risk_level="medium",
        max_steps=1,
        reasoning_summary=reason,
        handoff_acceptance=None,
        stop_condition=stop_condition,
        confidence=0.0,
        metadata=metadata,
    )


def _normalize_decision_payload(payload: dict[str, Any], *, goal: str) -> dict[str, Any]:
    allowed_top_level = {
        "operation_id",
        "cycle_index",
        "decision",
        "selected_agent",
        "selected_stage",
        "objective",
        "target_refs",
        "required_context",
        "success_criteria",
        "risk_level",
        "max_steps",
        "task_brief",
        "autonomy_level",
        "allowed_tool_names",
        "target_selection",
        "handoff_policy",
        "reasoning_summary",
        "handoff_acceptance",
        "stop_condition",
        "confidence",
        "metadata",
    }
    normalized = {key: value for key, value in payload.items() if key in allowed_top_level}

    selected_task = payload.get("selected_next_task")
    if "decision" not in normalized:
        normalized.update(_legacy_planner_result_to_decision(payload, selected_task=selected_task))
    elif normalized.get("decision") == "dispatch_agent" and selected_task:
        normalized.update(_legacy_selected_task_fields(selected_task))

    normalized.setdefault("objective", goal)
    normalized.setdefault("target_refs", [])
    normalized.setdefault("required_context", {})
    normalized.setdefault("success_criteria", [])
    normalized.setdefault("risk_level", "medium")
    normalized.setdefault("max_steps", 3)
    normalized.setdefault("task_brief", None)
    normalized.setdefault("autonomy_level", None)
    normalized.setdefault("allowed_tool_names", None)
    normalized.setdefault("target_selection", None)
    normalized.setdefault("handoff_policy", None)
    normalized.setdefault("reasoning_summary", payload.get("summary") or "")
    normalized.setdefault("handoff_acceptance", None)
    normalized.setdefault("stop_condition", None)
    normalized.setdefault("confidence", 0.5)
    normalized.setdefault("metadata", {})
    if not normalized.get("objective"):
        normalized["objective"] = goal
    normalized["target_refs"] = [ref for ref in normalized.get("target_refs", []) if isinstance(ref, dict)]
    normalized["required_context"] = dict(normalized.get("required_context") or {})
    normalized["success_criteria"] = [
        str(value) for value in normalized.get("success_criteria", []) if value is not None
    ]
    normalized["max_steps"] = int(normalized.get("max_steps") or 3)
    normalized["confidence"] = float(normalized.get("confidence") if normalized.get("confidence") is not None else 0.5)
    normalized["metadata"] = dict(normalized.get("metadata") or {})
    return normalized


def _legacy_planner_result_to_decision(
    payload: dict[str, Any],
    *,
    selected_task: Any,
) -> dict[str, Any]:
    if payload.get("stop_condition"):
        return {
            "decision": "stop_failed" if payload.get("replan_needed") else "stop_success",
            "selected_agent": None,
            "selected_stage": None,
        }
    if payload.get("replan_needed"):
        return {
            "decision": "replan",
            "selected_agent": None,
            "selected_stage": None,
        }
    if selected_task:
        return {
            "decision": "dispatch_agent",
            **_legacy_selected_task_fields(selected_task),
        }
    return {
        "decision": "replan",
        "selected_agent": None,
        "selected_stage": None,
    }


def _legacy_selected_task_fields(selected_task: Any) -> dict[str, Any]:
    if isinstance(selected_task, str):
        return {"selected_agent": None, "selected_stage": None}
    if not isinstance(selected_task, dict):
        return {"selected_agent": None, "selected_stage": None}
    selected_stage = selected_task.get("selected_stage")
    raw_stage = selected_task.get("stage_type") or selected_task.get("task_type") or selected_stage
    try:
        selected_stage = normalize_stage_name(raw_stage)
    except ValueError:
        selected_stage = selected_task.get("selected_stage")
    return {
        "selected_agent": selected_task.get("selected_agent"),
        "selected_stage": selected_stage,
        "objective": selected_task.get("objective") or selected_task.get("label"),
        "target_refs": selected_task.get("target_refs") or selected_task.get("input_refs") or [],
        "required_context": selected_task.get("required_context") or {},
        "success_criteria": selected_task.get("success_criteria") or [],
        "risk_level": selected_task.get("risk_level") or "medium",
        "max_steps": selected_task.get("max_steps") or 3,
        "task_brief": selected_task.get("task_brief"),
        "autonomy_level": selected_task.get("autonomy_level"),
        "allowed_tool_names": selected_task.get("allowed_tool_names"),
        "target_selection": selected_task.get("target_selection"),
        "handoff_policy": selected_task.get("handoff_policy"),
    }


__all__ = ["LLMMissionPlannerAdvisor", "LLMMissionPlannerAdvisorConfig"]
