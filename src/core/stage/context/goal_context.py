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
    return {
        **graph_context,
        "stage_context_builder": "goal_context_builder",
        "allowed_tool_names": ["goal_check", "chain_goal_check", "internal_service_discover", "http_probe", "artifact_store"],
        "goal_focus": {
            "objective": request.objective,
            "success_criteria": list(request.success_criteria),
            "recent_evidence": graph_context.get("recent_evidence") or [],
            "ag_process_history": request.ag_process_history,
            "runtime_sessions": runtime_context.get("active_sessions") or runtime_context.get("sessions") or [],
            "runtime_pivot_routes": runtime_context.get("pivot_routes") or [],
            "policy_scope": policy_context,
            "memory_tail": memory[-4:],
            "tool_catalog": available_tools,
        },
    }
