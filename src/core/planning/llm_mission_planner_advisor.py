"""LLM advisor for graph-driven planner decisions."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.agents.packy_llm import PackyLLMClient, PackyLLMError
from src.core.planning.mission_planner_agent import MissionPlannerResult
from src.core.planning.models import PlannerDecision
from src.core.stage.models import GraphUpdateIntent, StageType


class LLMMissionPlannerAdvisorConfig(BaseModel):
    """Prompt limits and model options for mission planning advice."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str | None = None
    max_context_chars: int = Field(default=24000, ge=4000)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class LLMMissionPlannerAdvisor:
    """Ask an LLM to choose the next PlannerDecision from KG/AG/runtime context.

    The primary path returns PlannerDecision. Legacy StageTask proposals remain
    available only through propose_stage_tasks for old callers.
    """

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
    ) -> PlannerDecision | dict[str, Any]:
        """Return the next PlannerDecision from the LLM."""

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
        payload = _normalize_decision_payload(payload, goal=goal)
        payload.setdefault("metadata", {})
        payload["metadata"] = {
            **dict(payload["metadata"]),
            "planner": "llm_mission_planner",
            "accepted": True,
            "model": response.model,
            "usage": response.usage,
        }
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

    def propose_stage_tasks(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any],
        recent_stage_results: list[dict[str, Any]] | None = None,
    ) -> MissionPlannerResult | dict[str, Any]:
        operation_id = str(graph_context.get("operation_id") or "operation")
        prompt = self._build_legacy_prompt(
            goal=goal,
            graph_context=graph_context,
            policy_context=policy_context,
            recent_stage_results=recent_stage_results,
        )
        try:
            response = self._client.complete_chat(
                system_prompt=LEGACY_SYSTEM_PROMPT,
                user_prompt=prompt,
                model=self._config.model,
                temperature=self._config.temperature,
            )
        except PackyLLMError as exc:
            return MissionPlannerResult(
                operation_id=operation_id,
                reasoning_summary="llm mission planner unavailable",
                replan_needed=True,
                stop_condition="planner_llm_unavailable",
                summary="llm mission planner unavailable",
                metadata={"planner": "llm_mission_planner", "accepted": False, "error": str(exc)},
            )

        payload = _extract_json_object(response.text)
        if payload is None:
            return MissionPlannerResult(
                operation_id=operation_id,
                reasoning_summary="llm mission planner returned non-json",
                replan_needed=True,
                stop_condition="invalid_planner_json",
                summary="llm mission planner returned non-json",
                metadata={"planner": "llm_mission_planner", "accepted": False, "raw_text": response.text[:1000]},
            )
        payload.setdefault("operation_id", operation_id)
        if "stage_tasks" in payload and "new_stage_tasks" not in payload:
            payload["new_stage_tasks"] = payload["stage_tasks"]
        payload.pop("stage_tasks", None)
        if "summary" in payload and "reasoning_summary" not in payload:
            payload["reasoning_summary"] = payload["summary"]
        payload = _normalize_planner_payload(payload)
        payload.setdefault("metadata", {})
        payload["metadata"] = {
            **dict(payload["metadata"]),
            "planner": "llm_mission_planner",
            "accepted": True,
            "model": response.model,
            "usage": response.usage,
        }
        try:
            return MissionPlannerResult.model_validate(payload)
        except ValidationError as exc:
            return MissionPlannerResult(
                operation_id=operation_id,
                reasoning_summary="llm mission planner returned invalid PlannerResult schema",
                replan_needed=True,
                stop_condition="invalid_planner_schema",
                summary="llm mission planner returned invalid PlannerResult schema",
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
        decision_contract = {
            "operation_id": "operation id string",
            "cycle_index": 0,
            "decision": "dispatch_agent | replan | pause_for_review | stop_success | stop_failed",
            "selected_agent": "recon_agent | vuln_analysis_agent | exploit_validation_agent | access_pivot_agent | goal_agent | null",
            "selected_stage": "RECON_STAGE | VULN_ANALYSIS_STAGE | EXPLOIT_STAGE | ACCESS_PIVOT_STAGE | GOAL_STAGE | null",
            "objective": "bounded objective for the selected agent or stop/replan reason",
            "target_refs": [],
            "required_context": {},
            "success_criteria": [],
            "risk_level": "low | medium | high | critical",
            "max_steps": 3,
            "reasoning_summary": "short summary, no chain of thought",
            "handoff_acceptance": None,
            "stop_condition": None,
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
                "active_sessions": graph_context.get("active_sessions") or [],
                "recent_attack_process_nodes": graph_context.get("recent_attack_process_nodes") or [],
                "recent_handoff_suggestions": graph_context.get("recent_handoff_suggestions") or [],
                "recent_failures": graph_context.get("recent_failures") or [],
                "current_goal": graph_context.get("current_goal") or goal,
                "recent_results": list(recent_stage_results or []),
                "planner_decision_contract": decision_contract,
                "allowed_agents": [
                    "recon_agent",
                    "vuln_analysis_agent",
                    "exploit_validation_agent",
                    "access_pivot_agent",
                    "goal_agent",
                ],
                "allowed_stages": [
                    "RECON_STAGE",
                    "VULN_ANALYSIS_STAGE",
                    "EXPLOIT_STAGE",
                    "ACCESS_PIVOT_STAGE",
                    "GOAL_STAGE",
                ],
            },
            self._config.max_context_chars,
        )
        return (
            "Return strict JSON only matching PlannerDecision. "
            "Read KG, AG, Runtime and Policy before choosing. "
            "AG is the attack process graph: it records each Planner decision, Agent execution, Tool call and Result. "
            "Do not output shell commands. Do not output MCP tool arguments. "
            "Do not use a fixed stage sequence. Select the next agent from graph evidence. "
            "If evidence is insufficient, choose recon_agent or replan. "
            "If policy does not allow the next action, choose pause_for_review. "
            "If the goal is already satisfied, choose stop_success. "
            "For dispatch_agent, selected_agent and selected_stage must be non-null. "
            "For non-dispatch decisions, selected_agent may be null. "
            "Use reasoning_summary for a concise justification without chain-of-thought.\n\n"
            f"{context}"
        )

    def _build_legacy_prompt(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any],
        recent_stage_results: list[dict[str, Any]] | None = None,
    ) -> str:
        stage_task_contract = {
            "task_id": "unique stage task id chosen from graph context",
            "stage_type": "one of RECON_STAGE, VULN_ANALYSIS_STAGE, EXPLOIT_STAGE, ACCESS_PIVOT_STAGE, GOAL_STAGE",
            "objective": "bounded objective for one stage agent",
            "target_scope": {},
            "prerequisites": [],
            "input_refs": [],
            "expected_outputs": [],
            "constraints": [],
            "status": "ready",
            "target_refs": [],
            "required_context": {},
            "success_criteria": [],
            "max_steps": 3,
            "risk_level": "low | medium | high | critical",
            "priority": 50,
            "metadata": {},
        }
        context = _truncate_json(
            {
                "mission_goal": goal,
                "graph_context": graph_context,
                "policy_context": policy_context,
                "recent_stage_results": list(recent_stage_results or []),
                "planner_result_contract": {
                    "operation_id": "operation id string",
                    "reasoning_summary": "short summary, no chain of thought",
                    "new_stage_tasks": [stage_task_contract],
                    "selected_next_task": stage_task_contract | {"nullable": True},
                    "task_updates": [],
                    "replan_needed": False,
                    "stop_condition": None,
                    "graph_update_intents": [],
                    "confidence": 0.8,
                    "metadata": {},
                },
                "allowed_stage_types": [
                    "RECON_STAGE",
                    "VULN_ANALYSIS_STAGE",
                    "EXPLOIT_STAGE",
                    "ACCESS_PIVOT_STAGE",
                    "GOAL_STAGE",
                ],
            },
            self._config.max_context_chars,
        )
        return (
            "Return strict JSON only matching PlannerResult/MissionPlannerResult. "
            "Use the supplied planner_result_contract as a field contract, not as values to copy. "
            "Derive task ids, stage order, target_refs, required_context, priorities and selected_next_task from the current "
            "KG/AG/TG/runtime graph and recent_stage_results. Do not use a fixed stage sequence. "
            "If recent stage output already created TG candidates or graph facts, read them before proposing more work. "
            "Allowed stage_type values are RECON_STAGE, VULN_ANALYSIS_STAGE, EXPLOIT_STAGE, ACCESS_PIVOT_STAGE and GOAL_STAGE. "
            "Do not output shell commands or facts not present in the graph context. "
            "Do not use fields named current_phase, phase_reasoning, mission_status, goal_summary, next_task, "
            "stage_task_updates, candidate_tasks, deferred_tasks, risks, graph, action, fields, or task_id inside graph_update_intents. "
            "If no graph_update_intents are needed, return an empty list.\n\n"
            f"{context}"
        )


SYSTEM_PROMPT = (
    "You are Aegra's PlannerAgent. You own global goal understanding, graph-state analysis, "
    "policy-aware next-agent selection, and stop/replan decisions. Code only assembles context "
    "and validates your JSON; do not rely on hard-coded stage ordering. "
    "Policy must constrain every decision. Do not output shell commands or MCP tool arguments. "
    "Output strict JSON only, with no chain-of-thought."
)


LEGACY_SYSTEM_PROMPT = (
    "You are Aegra's legacy PlannerAgent. You produce PlannerResult StageTask proposals for older callers. "
    "Do not output shell commands. Output strict JSON only, with no chain-of-thought."
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
    stage_type = _normalize_stage_type(selected_task.get("stage_type") or selected_task.get("task_type"))
    selected_stage = stage_type.legacy.value if stage_type is not None else selected_task.get("selected_stage")
    stage_agent = {
        "RECON_STAGE": "recon_agent",
        "VULN_ANALYSIS_STAGE": "vuln_analysis_agent",
        "EXPLOIT_STAGE": "exploit_validation_agent",
        "ACCESS_PIVOT_STAGE": "access_pivot_agent",
        "GOAL_STAGE": "goal_agent",
    }.get(str(selected_stage))
    return {
        "selected_agent": selected_task.get("selected_agent") or stage_agent,
        "selected_stage": selected_stage,
        "objective": selected_task.get("objective") or selected_task.get("label"),
        "target_refs": selected_task.get("target_refs") or selected_task.get("input_refs") or [],
        "required_context": selected_task.get("required_context") or {},
        "success_criteria": selected_task.get("success_criteria") or [],
        "risk_level": selected_task.get("risk_level") or "medium",
        "max_steps": selected_task.get("max_steps") or 3,
    }


def _normalize_planner_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_top_level = {
        "operation_id",
        "reasoning_summary",
        "new_stage_tasks",
        "selected_next_task",
        "task_updates",
        "replan_needed",
        "stop_condition",
        "graph_update_intents",
        "confidence",
        "metadata",
        "dependencies",
        "summary",
    }
    normalized = {key: value for key, value in payload.items() if key in allowed_top_level}
    tasks = [
        task
        for item in normalized.get("new_stage_tasks", []) if isinstance(item, dict)
        if (task := _normalize_stage_task(item)) is not None
    ]
    normalized["new_stage_tasks"] = tasks
    selected = normalized.get("selected_next_task")
    if isinstance(selected, str):
        normalized["selected_next_task"] = next((task for task in tasks if task.get("task_id") == selected), None)
    elif isinstance(selected, dict):
        normalized["selected_next_task"] = _normalize_stage_task(selected)
    else:
        normalized["selected_next_task"] = None
    normalized["graph_update_intents"] = [
        intent
        for item in normalized.get("graph_update_intents", []) if isinstance(item, dict)
        if (intent := _normalize_graph_update_intent(item)) is not None
    ]
    normalized["task_updates"] = [dict(item) for item in normalized.get("task_updates", []) if isinstance(item, dict)]
    normalized["dependencies"] = [dict(item) for item in normalized.get("dependencies", []) if isinstance(item, dict)]
    return normalized


def _normalize_stage_task(item: dict[str, Any]) -> dict[str, Any] | None:
    raw_stage_type = item.get("stage_type") or item.get("task_type")
    stage_type = _normalize_stage_type(raw_stage_type)
    objective = item.get("objective") or item.get("label") or item.get("summary")
    task_id = item.get("task_id") or item.get("id")
    if stage_type is None or objective is None:
        return None
    if not task_id:
        task_id = f"stage-{stage_type.canonical.value.replace('_', '-')}-{abs(hash(str(objective))) % 100000}"
    allowed = {
        "task_id",
        "stage_type",
        "objective",
        "target_scope",
        "prerequisites",
        "input_refs",
        "expected_outputs",
        "constraints",
        "status",
        "target_refs",
        "required_context",
        "success_criteria",
        "max_steps",
        "risk_level",
        "priority",
        "metadata",
    }
    normalized = {key: value for key, value in item.items() if key in allowed}
    normalized.update(
        {
            "task_id": str(task_id),
            "stage_type": stage_type.legacy.value,
            "objective": str(objective),
            "target_scope": dict(item.get("target_scope") or {}),
            "prerequisites": [str(value) for value in item.get("prerequisites", []) if value is not None],
            "input_refs": [ref for ref in item.get("input_refs", []) if isinstance(ref, dict)],
            "expected_outputs": [str(value) for value in item.get("expected_outputs", []) if value is not None],
            "constraints": [str(value) for value in item.get("constraints", []) if value is not None],
            "status": str(item.get("status") or "ready"),
            "target_refs": [ref for ref in item.get("target_refs", []) if isinstance(ref, dict)],
            "required_context": dict(item.get("required_context") or {}),
            "success_criteria": [str(value) for value in item.get("success_criteria", []) if value is not None],
            "max_steps": int(item.get("max_steps") or 3),
            "risk_level": str(item.get("risk_level") or "low"),
            "priority": int(item.get("priority") or 50),
            "metadata": dict(item.get("metadata") or {}),
        }
    )
    return normalized


def _normalize_stage_type(value: Any) -> StageType | None:
    if value is None:
        return None
    raw = str(value).strip()
    aliases = {
        "reconnaissance": "RECON_STAGE",
        "recon": "RECON_STAGE",
        "scan": "RECON_STAGE",
        "enumeration": "RECON_STAGE",
        "vulnerability_analysis": "VULN_ANALYSIS_STAGE",
        "vuln_analysis": "VULN_ANALYSIS_STAGE",
        "vulnerability": "VULN_ANALYSIS_STAGE",
        "exploit": "EXPLOIT_STAGE",
        "exploitation": "EXPLOIT_STAGE",
        "access": "ACCESS_PIVOT_STAGE",
        "pivot": "ACCESS_PIVOT_STAGE",
        "access_pivot": "ACCESS_PIVOT_STAGE",
        "goal": "GOAL_STAGE",
        "validation": "GOAL_STAGE",
    }
    candidate = aliases.get(raw.lower(), raw)
    try:
        return StageType(candidate)
    except ValueError:
        return None


def _normalize_graph_update_intent(item: dict[str, Any]) -> dict[str, Any] | None:
    allowed = {
        "target_graph",
        "operation",
        "entity_type",
        "entity_ref",
        "payload",
        "evidence_refs",
        "confidence",
        "source",
    }
    candidate = {key: value for key, value in item.items() if key in allowed}
    try:
        return GraphUpdateIntent.model_validate(candidate).model_dump(mode="json")
    except ValidationError:
        return None


__all__ = ["LLMMissionPlannerAdvisor", "LLMMissionPlannerAdvisorConfig"]
