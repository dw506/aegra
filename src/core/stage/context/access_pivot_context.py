"""AccessPivotAgent context builder."""

from __future__ import annotations

from typing import Any

from src.core.stage.models import StageExecutionRequest


def build_access_pivot_context(
    request: StageExecutionRequest,
    graph_context: dict[str, Any],
    runtime_context: dict[str, Any],
    policy_context: dict[str, Any],
    memory: list[dict[str, Any]],
    available_tools: dict[str, Any],
) -> dict[str, Any]:
    return {
        **graph_context,
        "stage_context_builder": "access_pivot_context_builder",
        "allowed_tool_names": [
            "credential_check",
            "session_probe",
            "session_open_lab",
            "identity_context_probe",
            "privilege_context_probe",
            "pivot_route_probe",
            "internal_service_discover",
            "tcp_connect_probe",
            "http_probe",
        ],
        "access_pivot_focus": {
            "objective": request.objective,
            "credentials": request.required_context.get("credentials") or runtime_context.get("credentials") or [],
            "active_sessions": runtime_context.get("active_sessions") or graph_context.get("active_sessions") or [],
            "pivot_candidates": request.required_context.get("pivot_candidates") or [],
            "pivot_routes": runtime_context.get("pivot_routes") or [],
            "policy_scope": policy_context,
            "memory_tail": memory[-4:],
            "tool_catalog": available_tools,
            "validation_mode": "authorized_session_credential_pivot_reachability_only",
        },
    }
