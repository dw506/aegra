"""LLM advisor for graph-driven mission planning."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.agents.packy_llm import PackyLLMClient, PackyLLMError
from src.core.planning.mission_planner_agent import MissionPlannerResult


class LLMMissionPlannerAdvisorConfig(BaseModel):
    """Prompt limits and model options for mission planning advice."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str | None = None
    max_context_chars: int = Field(default=24000, ge=4000)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class LLMMissionPlannerAdvisor:
    """Ask an LLM to produce stage tasks from KG/AG/TG/runtime context.

    The advisor is deliberately bounded: it may only return StageTask-like
    proposals and dependency edges. Tool commands and direct graph mutations are
    handled later by Stage Agents and ResultApplier.
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

    def propose_stage_tasks(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any],
        recent_stage_results: list[dict[str, Any]] | None = None,
    ) -> MissionPlannerResult | dict[str, Any]:
        operation_id = str(graph_context.get("operation_id") or "operation")
        prompt = self._build_prompt(
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
                metadata={"planner": "llm_mission_planner", "accepted": False, "error": str(exc)},
            )

    def _build_prompt(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any],
        recent_stage_results: list[dict[str, Any]] | None = None,
    ) -> str:
        context = _truncate_json(
            {
                "mission_goal": goal,
                "graph_context": graph_context,
                "policy_context": policy_context,
                "recent_stage_results": list(recent_stage_results or []),
                "allowed_stage_types": [
                    "recon",
                    "vuln_analysis",
                    "exploit",
                    "access_pivot",
                    "goal",
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
            "Return only JSON matching PlannerResult/MissionPlannerResult. "
            "You are the complete Planner Agent. Understand the user goal, analyze KG/AG/TG/Runtime/Policy, "
            "identify the current phase, decompose or update stage tasks, select the next best StageTask, "
            "and emit graph_update_intents for ResultApplier. Do not use a fixed stage order. "
            "Use target_refs already present in KG/AG/TG/query context; do not invent shell commands or facts.\n\n"
            f"{context}"
        )


SYSTEM_PROMPT = (
    "You are Aegra's LLM Planner Agent. You own global goal understanding, graph-state analysis, "
    "stage task decomposition, TG update intent generation, and next-task selection. Code only "
    "assembles context and validates your JSON; do not rely on hard-coded stage ordering. "
    "Policy must constrain every task you propose. Do not output shell commands. "
    "Output strict JSON only, with no chain-of-thought."
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


__all__ = ["LLMMissionPlannerAdvisor", "LLMMissionPlannerAdvisorConfig"]
