"""GoalAgent context builder."""

from __future__ import annotations

from typing import Any

from src.core.stage.models import StageExecutionRequest


def build_goal_context(
    request: StageExecutionRequest,
    graph_context: dict[str, Any],
    runtime_context: dict[str, Any],
    policy_context: dict[str, Any],
    memory: list[dict[str, Any]],
    available_tools: dict[str, Any],
) -> dict[str, Any]:
    # Extract success_condition_progress from runtime context (never contains private secrets)
    runtime_meta = runtime_context.get("metadata") or runtime_context.get("execution", {}).get("metadata") or {}
    success_progress = runtime_meta.get("success_condition_progress") or {}

    return {
        **graph_context,
        "stage_context_builder": "goal_context_builder",
        "goal_focus": {
            "objective": request.objective,
            "success_criteria": list(request.success_criteria),
            "goal_requirements": request.required_context.get("goal_requirements")
            or request.required_context.get("success_conditions")
            or runtime_context.get("goal_requirements")
            or runtime_context.get("success_conditions")
            or {},
            "success_condition_progress": {
                "satisfied": success_progress.get("satisfied") or [],
                "missing": success_progress.get("missing") or [],
                "eligible_for_stop": success_progress.get("eligible_for_stop") or False,
                # Never include private_success_rubric, raw flags, or tokens here
            },
            "recent_evidence": graph_context.get("recent_evidence") or [],
            "ag_process_history": request.ag_process_history,
            "policy_scope": policy_context,
            "memory_tail": memory[-4:],
            "tool_catalog": available_tools,
        },
        # Explicit reminder to GoalAgent: do not end operation, only submit proof
        "goal_agent_constraints": {
            "may_emit_stop_success": False,
            "may_expose_private_rubric": False,
            "may_guess_flags_or_tokens": False,
            "must_use_existing_evidence_refs": True,
        },
    }
