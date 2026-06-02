"""VulnAnalysisAgent context builder."""

from __future__ import annotations

from typing import Any

from src.core.stage.models import StageExecutionRequest


def build_vuln_analysis_context(
    request: StageExecutionRequest,
    graph_context: dict[str, Any],
    runtime_context: dict[str, Any],
    policy_context: dict[str, Any],
    memory: list[dict[str, Any]],
    available_tools: dict[str, Any],
) -> dict[str, Any]:
    return {
        **graph_context,
        "stage_context_builder": "vuln_analysis_context_builder",
        "allowed_tool_names": ["vuln_profile_match", "validation_precheck", "whatweb_fingerprint", "nuclei_scan", "http_probe"],
        "vulnerability_analysis_focus": {
            "objective": request.objective,
            "services": graph_context.get("known_services") or request.required_context.get("services") or [],
            "fingerprints": request.required_context.get("fingerprints") or [],
            "recent_evidence": graph_context.get("recent_evidence") or [],
            "recent_failures": graph_context.get("recent_failures") or request.ag_process_history.get("recent_failures") or [],
            "policy_scope": policy_context,
            "memory_tail": memory[-4:],
            "tool_catalog": available_tools,
            "runtime_context": runtime_context,
        },
    }
