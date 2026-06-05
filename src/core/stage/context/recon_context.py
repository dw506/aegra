"""ReconAgent context builder."""

from __future__ import annotations

from typing import Any

from src.core.stage.models import StageExecutionRequest


def build_recon_context(
    request: StageExecutionRequest,
    graph_context: dict[str, Any],
    runtime_context: dict[str, Any],
    policy_context: dict[str, Any],
    memory: list[dict[str, Any]],
    available_tools: dict[str, Any],
) -> dict[str, Any]:
    return {
        **graph_context,
        "stage_context_builder": "recon_context_builder",
        "recon_focus": {
            "objective": request.objective,
            "target_refs": [ref.model_dump(mode="json") for ref in request.target_refs],
            "known_assets": graph_context.get("known_assets") or request.kg_snapshot.get("known_assets") or [],
            "known_services": graph_context.get("known_services") or request.kg_snapshot.get("known_services") or [],
            "policy_scope": policy_context,
            "tool_names": _tool_names(available_tools),
            "tool_catalog": available_tools,
            "memory_tail": memory[-4:],
            "runtime_reachability": runtime_context.get("reachability") or runtime_context.get("pivot_routes") or {},
        },
    }


def _tool_names(available_tools: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for server in available_tools.values():
        if not isinstance(server, dict):
            continue
        for tool in server.get("tools", []) if isinstance(server.get("tools"), list) else []:
            if isinstance(tool, dict) and (tool.get("name") or tool.get("tool_name")):
                names.append(str(tool.get("name") or tool.get("tool_name")))
    return sorted(set(names))
