"""Read-only adapters for the automation run visualization console."""

from __future__ import annotations

from typing import Any

from src.core.models.runtime import RuntimeState

KG_TYPES = {
    "Host",
    "Service",
    "Network",
    "Technology",
    "Identity",
    "Credential",
    "Session",
    "Vulnerability",
    "Finding",
    "Evidence",
    "Observation",
    "Goal",
    "Unknown",
}
KG_STATUS = {"unknown", "observed", "suspected", "verified", "rejected", "active", "inactive", "blocked", "failed"}
AG_STATUS = {"pending", "running", "success", "failed", "blocked", "skipped"}
AG_EDGE_TYPES = {"enables", "supports", "contradicts", "derives_from", "triggers_replan", "verifies", "fails", "blocks", "unknown"}
AGENTS = ["ReconAgent", "VulnAnalysisAgent", "ExploitValidationAgent", "AccessPivotAgent", "GoalAgent"]


def build_unified_visualization(
    *,
    operation_id: str,
    kg_payload: dict[str, Any] | None = None,
    ag_payload: dict[str, Any] | None = None,
    runtime_state: RuntimeState | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the frontend visualization read model without mutating runtime or graphs."""

    runtime = _runtime_payload(runtime_state)
    kg = _mapping(kg_payload)
    ag = _mapping(ag_payload)
    kg_nodes = [_kg_node(node) for node in _list(kg.get("nodes"))]
    kg_edges = [_kg_edge(edge) for edge in _list(kg.get("edges"))]
    ag_nodes = [_ag_node(node) for node in _list(ag.get("nodes"))]
    ag_edges = [_ag_edge(edge) for edge in _list(ag.get("edges"))]
    timeline = _timeline_events(runtime=runtime, ag_nodes=ag_nodes)
    tool_trace = _tool_trace(runtime=runtime, ag_nodes=ag_nodes, timeline=timeline)
    evidence = _evidence_items(kg_nodes=kg_nodes, ag_nodes=ag_nodes, timeline=timeline)
    findings = _runtime_findings(runtime)
    agent_trace = _agent_trace(ag_nodes=ag_nodes, runtime=runtime, tool_trace=tool_trace)
    attack_path = _attack_path(ag_nodes=ag_nodes, ag_edges=ag_edges)
    operation = _operation(operation_id=operation_id, runtime=runtime)
    overview = _overview(
        operation=operation,
        kg_nodes=kg_nodes,
        ag_nodes=ag_nodes,
        evidence=evidence,
        findings=findings,
        agent_trace=agent_trace,
        attack_path=attack_path,
    )
    return {
        "operation": operation,
        "overview": overview,
        "kg": {"nodes": kg_nodes, "edges": kg_edges},
        "ag": {"nodes": ag_nodes, "edges": ag_edges},
        "timeline": timeline,
        "tool_trace": tool_trace,
        "evidence": evidence,
        "findings": findings,
        "agent_trace": agent_trace,
        "attack_path": attack_path,
    }


def _operation(*, operation_id: str, runtime: dict[str, Any]) -> dict[str, Any]:
    execution = _mapping(runtime.get("execution"))
    metadata = _mapping(execution.get("metadata"))
    return {
        "id": operation_id,
        "target_scope": _target_scope(runtime),
        "status": _string(runtime.get("operation_status") or execution.get("status")) or "unknown",
        "current_round": _current_round(runtime),
        "goal_status": _goal_status(runtime),
        "created_at": _string(execution.get("created_at")),
        "updated_at": _string(runtime.get("last_updated") or execution.get("finished_at") or execution.get("started_at")),
        "metadata": metadata,
    }


def _overview(
    *,
    operation: dict[str, Any],
    kg_nodes: list[dict[str, Any]],
    ag_nodes: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    agent_trace: list[dict[str, Any]],
    attack_path: dict[str, Any],
) -> dict[str, Any]:
    latest_trace = agent_trace[-1] if agent_trace else None
    latest_decision = _mapping(latest_trace.get("planner_decision")) if latest_trace else None
    current_agent = None
    if latest_decision:
        selected = _list(latest_decision.get("selected_agents"))
        current_agent = _string(selected[0]) if selected else None
    return {
        "asset_count": sum(1 for node in kg_nodes if node["type"] in {"Host", "Network", "Identity"}),
        "service_count": sum(1 for node in kg_nodes if node["type"] == "Service"),
        "finding_count": len(findings) or sum(1 for node in kg_nodes if node["type"] in {"Finding", "Vulnerability"}),
        "verified_finding_count": sum(
            1 for node in kg_nodes if node["type"] in {"Finding", "Vulnerability"} and node["status"] == "verified"
        ),
        "evidence_count": len(evidence),
        "access_count": sum(1 for node in kg_nodes if node["type"] in {"Session", "Credential"} and node["status"] == "active"),
        "current_agent": current_agent,
        "latest_decision": latest_decision,
        "main_path_summary": [node["display_name"] for node in attack_path["nodes"][:8]],
        "latest_evidence": evidence[-1] if evidence else None,
        "operation_status": operation["status"],
        "current_round": operation["current_round"],
        "goal_status": operation["goal_status"],
    }


def _kg_node(item: dict[str, Any]) -> dict[str, Any]:
    props = _merged_properties(item)
    node_type = _kg_type(_string(item.get("type") or item.get("node_type") or props.get("type")))
    status = _kg_status(_string(item.get("status") or item.get("truth_status") or item.get("activation_status") or props.get("status")))
    node_id = _string(item.get("id")) or "unknown"
    return {
        "id": node_id,
        "type": node_type,
        "display_name": _kg_display_name(node_type, node_id, item, props),
        "summary": _summary(item, props),
        "status": status,
        "confidence": _confidence(item.get("confidence") or props.get("confidence")),
        "target": _target(item, props),
        "created_by": _string(item.get("source_task_id") or props.get("created_by") or props.get("source_agent")),
        "round": _int(props.get("round") or props.get("cycle_index")),
        "evidence_ids": _evidence_ids(item, props),
        "metadata": props,
        "created_at": _string(item.get("first_seen") or item.get("created_at") or props.get("created_at")),
        "updated_at": _string(item.get("last_seen") or item.get("updated_at") or props.get("updated_at")),
        "linked_ag_node_ids": [],
    }


def _kg_edge(item: dict[str, Any]) -> dict[str, Any]:
    props = _merged_properties(item)
    edge_type = _string(item.get("type") or item.get("edge_type") or props.get("type")) or "related_to"
    return {
        "id": _string(item.get("id")) or f"{item.get('source')}::{edge_type}::{item.get('target')}",
        "source": _string(item.get("source")) or "",
        "target": _string(item.get("target")) or "",
        "type": edge_type,
        "display_name": _humanize(edge_type),
        "summary": _summary(item, props),
        "evidence_ids": _evidence_ids(item, props),
        "confidence": _confidence(item.get("confidence") or props.get("confidence")),
        "metadata": props,
    }


def _ag_node(item: dict[str, Any]) -> dict[str, Any]:
    props = _merged_properties(item)
    node_id = _string(item.get("id")) or "unknown"
    raw_type = _string(item.get("node_type") or item.get("type") or props.get("node_role") or item.get("kind"))
    agent = _agent_name(item, props)
    round_no = _int(item.get("cycle_index") or props.get("cycle_index") or props.get("round")) or 0
    result_summary = _string(props.get("result_summary") or props.get("visual_summary") or item.get("summary")) or "No summary available"
    action_summary = _string(props.get("action_summary") or props.get("objective") or props.get("visual_title") or item.get("label")) or "No summary available"
    ag_type = _ag_type(raw_type, agent, props)
    status = _ag_status(_string(item.get("status") or props.get("status") or props.get("visual_outcome")))
    target = _target(item, props)
    return {
        "id": node_id,
        "type": ag_type,
        "round": round_no,
        "agent": agent,
        "display_name": _ag_display_name(ag_type, target, action_summary, result_summary, props),
        "action_summary": action_summary,
        "result_summary": result_summary,
        "target": target,
        "status": status,
        "confidence": _confidence(item.get("confidence") or props.get("confidence")),
        "evidence_ids": _evidence_ids(item, props),
        "kg_node_ids": _kg_ref_ids(item, props),
        "is_main_path": _bool(props.get("is_main_path") or props.get("main_path")),
        "started_at": _string(props.get("started_at") or item.get("created_at")),
        "ended_at": _string(props.get("ended_at")),
        "metadata": props,
        "raw_type": raw_type or "unknown",
    }


def _ag_edge(item: dict[str, Any]) -> dict[str, Any]:
    props = _merged_properties(item)
    raw_type = _string(item.get("type") or item.get("edge_type") or props.get("type")) or "unknown"
    edge_type = _ag_edge_type(raw_type)
    return {
        "id": _string(item.get("id")) or f"{item.get('source')}::{edge_type}::{item.get('target')}",
        "source": _string(item.get("source")) or "",
        "target": _string(item.get("target")) or "",
        "type": edge_type,
        "display_name": _string(item.get("display_name") or item.get("label") or props.get("display_name")) or _humanize(edge_type),
        "summary": _summary(item, props),
        "evidence_ids": _evidence_ids(item, props),
        "confidence": _confidence(item.get("confidence") or props.get("confidence")),
        "metadata": props,
    }


def _timeline_events(*, runtime: dict[str, Any], ag_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for node in sorted(ag_nodes, key=lambda item: (item["round"], _phase_order(_phase_for_ag(item)), item["id"])):
        phase = _phase_for_ag(node)
        events.append(
            {
                "id": f"timeline::{node['id']}",
                "round": node["round"],
                "phase": phase,
                "agent": node["agent"] if node["agent"] != "Other" else None,
                "target": node["target"],
                "display_name": node["display_name"],
                "summary": node["result_summary"] or node["action_summary"] or "No summary available",
                "status": node["status"],
                "tool_name": _string(node["metadata"].get("tool_name")),
                "tool_trace_refs": _tool_trace_refs_for_node(node),
                "evidence_ids": list(node["evidence_ids"]),
                "kg_updates": _list(node["metadata"].get("kg_updates") or node["metadata"].get("kg_node_ids")),
                "ag_updates": [node["id"]],
                "created_at": node["started_at"] or node["ended_at"],
                "metadata": node["metadata"],
            }
        )
    for event in _list(runtime.get("pending_events")):
        event_id = _string(event.get("event_id")) or f"runtime-event::{len(events)}"
        events.append(
            {
                "id": f"timeline::{event_id}",
                "round": _current_round(runtime),
                "phase": "error" if _runtime_event_is_error(event) else "result_apply",
                "agent": None,
                "target": _string(_mapping(event.get("metadata")).get("target")),
                "display_name": _string(event.get("summary") or event.get("event_type")) or "Runtime event",
                "summary": _string(event.get("summary")) or "No summary available",
                "status": "failed" if _runtime_event_is_error(event) else "success",
                "tool_name": None,
                "tool_trace_refs": [],
                "evidence_ids": _event_refs(event),
                "kg_updates": [],
                "ag_updates": [],
                "created_at": _string(event.get("created_at")),
                "metadata": event,
            }
        )
    return sorted(events, key=lambda item: (item["round"], _phase_order(item["phase"]), item["id"]))


def _tool_trace(
    *,
    runtime: dict[str, Any],
    ag_nodes: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    stage_by_round = _stage_context_by_round(ag_nodes)
    for node in ag_nodes:
        if node["raw_type"] != "TOOL_CALL" and not _string(node["metadata"].get("tool_name")):
            continue
        trace = _tool_trace_from_ag_node(node, stage_by_round.get(node["round"], {}))
        traces[trace["id"]] = trace
    execution = _mapping(runtime.get("execution"))
    metadata = _mapping(execution.get("metadata"))
    for item in _list(metadata.get("audit_log")):
        event = _mapping(item)
        if event.get("event_type") != "stage_tool_trace":
            continue
        trace = _tool_trace_from_audit_event(event, stage_by_round.get(_current_round(runtime), {}), len(traces))
        traces.setdefault(trace["id"], trace)
    for event in timeline:
        for trace_id in event.get("tool_trace_refs", []):
            if trace_id in traces:
                traces[trace_id]["timeline_event_ids"].append(event["id"])
    for trace in traces.values():
        trace["timeline_event_ids"] = sorted(set(trace["timeline_event_ids"]))
    return sorted(traces.values(), key=lambda item: (_int(item.get("round")) or -1, _int(item.get("step")) or 0, item["id"]))


def _evidence_items(
    *,
    kg_nodes: list[dict[str, Any]],
    ag_nodes: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for node in kg_nodes:
        if node["type"] == "Evidence":
            items[node["id"]] = _evidence_from_kg_node(node)
        for evidence_id in node["evidence_ids"]:
            items.setdefault(evidence_id, _evidence_stub(evidence_id))
            items[evidence_id]["linked_kg_node_ids"].append(node["id"])
    for node in ag_nodes:
        for evidence_id in node["evidence_ids"]:
            items.setdefault(evidence_id, _evidence_stub(evidence_id))
            items[evidence_id]["linked_ag_node_ids"].append(node["id"])
            items[evidence_id]["round"] = items[evidence_id]["round"] if items[evidence_id]["round"] is not None else node["round"]
            items[evidence_id]["created_by"] = items[evidence_id]["created_by"] or node["agent"]
    for event in timeline:
        for evidence_id in event["evidence_ids"]:
            items.setdefault(evidence_id, _evidence_stub(evidence_id))
            items[evidence_id]["round"] = items[evidence_id]["round"] if items[evidence_id]["round"] is not None else event["round"]
            items[evidence_id]["created_by"] = items[evidence_id]["created_by"] or event["agent"]
    for item in items.values():
        item["linked_kg_node_ids"] = sorted(set(item["linked_kg_node_ids"]))
        item["linked_ag_node_ids"] = sorted(set(item["linked_ag_node_ids"]))
    return sorted(items.values(), key=lambda item: (_int(item.get("round")) or -1, item["id"]))


def _runtime_findings(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    execution = _mapping(runtime.get("execution"))
    metadata = _mapping(execution.get("metadata"))
    result: list[dict[str, Any]] = []
    for index, item in enumerate(_list(metadata.get("findings"))):
        if not isinstance(item, dict):
            continue
        props = _merged_properties(item)
        finding_id = _string(item.get("finding_id") or item.get("id")) or f"finding::{index}"
        result.append(
            {
                "id": finding_id,
                "finding_id": finding_id,
                "title": _string(props.get("title") or props.get("summary")) or finding_id,
                "summary": _summary(item, props),
                "kind": _string(props.get("kind") or props.get("category") or props.get("type")) or "finding",
                "severity": _string(props.get("severity")) or "info",
                "status": _string(props.get("status") or props.get("validation_status")) or "observed",
                "confidence": props.get("confidence"),
                "target": _target(item, props),
                "evidence_ids": _evidence_ids(item, props),
                "source_agent": _string(props.get("source_agent")),
                "stage_task_id": _string(props.get("stage_task_id")),
                "recorded_at": _string(props.get("recorded_at")),
                "metadata": props,
            }
        )
    return result


def _agent_trace(*, ag_nodes: list[dict[str, Any]], runtime: dict[str, Any], tool_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rounds = sorted({node["round"] for node in ag_nodes} | ({_current_round(runtime)} if runtime else set()))
    result: list[dict[str, Any]] = []
    for round_no in rounds:
        nodes = [node for node in ag_nodes if node["round"] == round_no]
        planner = next((node for node in nodes if node["type"] == "ReplanStep" or node["raw_type"] == "PLANNER_DECISION"), None)
        stage_result = next((node for node in nodes if node["raw_type"] == "STAGE_RESULT"), None)
        selected_agents = sorted({node["agent"] for node in nodes if node["agent"] != "Other" and node["type"] != "ReplanStep"})
        round_traces = [trace for trace in tool_trace if trace["round"] == round_no]
        evidence_ids = sorted({evidence_id for node in nodes for evidence_id in node["evidence_ids"]} | {evidence_id for trace in round_traces for evidence_id in trace["evidence_ids"]})
        finding_count = sum(1 for node in nodes if node["type"] in {"AnalysisStep", "ValidationStep"} and node["status"] == "success")
        decision_summary = planner["result_summary"] if planner else "No planner decision recorded"
        selected_agent = selected_agents[0] if selected_agents else _string((planner or {}).get("metadata", {}).get("selected_agent"))
        selected_stage = _string((planner or {}).get("metadata", {}).get("selected_stage") or (stage_result or {}).get("metadata", {}).get("stage_type"))
        objective = _string((planner or {}).get("metadata", {}).get("objective") or (stage_result or {}).get("metadata", {}).get("objective"))
        task_brief = _string((planner or {}).get("metadata", {}).get("task_brief"))
        status = _agent_trace_status(nodes)
        summary = _string((stage_result or {}).get("result_summary") or decision_summary) or "No summary available"
        agent_states = []
        for agent in AGENTS:
            matching = [node for node in nodes if node["agent"] == agent]
            agent_states.append(
                {
                    "agent": agent,
                    "state": _agent_state(matching),
                    "summary": matching[-1]["result_summary"] if matching else "Idle",
                }
            )
        result.append(
            {
                "id": f"agent-trace::{round_no}",
                "cycle_index": round_no,
                "round": round_no,
                "selected_agent": selected_agent,
                "selected_stage": selected_stage,
                "objective": objective,
                "task_brief": task_brief,
                "status": status,
                "summary": summary,
                "handoff_suggestion": _handoff_suggestion(nodes),
                "replan_recommendation": _replan_recommendation(nodes),
                "stage_result": stage_result,
                "tool_traces": round_traces,
                "tool_count": len(round_traces),
                "evidence_count": len(evidence_ids),
                "finding_count": finding_count,
                "round": round_no,
                "planner_decision": {
                    "selected_agents": selected_agents,
                    "decision_summary": decision_summary,
                    "reason": _string((planner or {}).get("metadata", {}).get("reasoning_summary")) or decision_summary,
                    "expected_outcome": _string((planner or {}).get("metadata", {}).get("objective")) or "No expected outcome recorded",
                    "blocked_by": _list((planner or {}).get("metadata", {}).get("blocked_by")),
                    "priority": _priority((planner or {}).get("metadata", {}).get("risk_level")),
                },
                "agent_states": agent_states,
                "created_at": _string((planner or {}).get("started_at")),
            }
        )
    return result


def _tool_trace_from_ag_node(node: dict[str, Any], stage_context: dict[str, Any]) -> dict[str, Any]:
    metadata = _mapping(node.get("metadata"))
    trace_id = _string(metadata.get("trace_id") or node.get("id")) or "tool-trace"
    return {
        "id": trace_id,
        "round": node["round"],
        "agent": node["agent"],
        "stage": _string(metadata.get("stage_type") or stage_context.get("stage")),
        "step": _int(metadata.get("step") or metadata.get("step_index")),
        "server_id": _string(metadata.get("server_id")),
        "tool_name": _string(metadata.get("tool_name") or node.get("display_name")) or "tool",
        "arguments": _mapping(metadata.get("arguments")),
        "success": _bool(metadata.get("success")) if metadata.get("success") is not None else node["status"] == "success",
        "exit_code": metadata.get("exit_code"),
        "summary": _string(metadata.get("output_summary") or metadata.get("visual_summary") or node.get("result_summary")) or "No summary available",
        "stdout_excerpt": _string(metadata.get("stdout_excerpt") or metadata.get("stdout")) or "",
        "stderr_excerpt": _string(metadata.get("stderr_excerpt") or metadata.get("stderr")) or "",
        "raw_output_ref": _string(metadata.get("raw_output_ref")),
        "evidence_ids": list(node["evidence_ids"]),
        "created_at": _string(metadata.get("started_at") or node.get("started_at")),
        "timeline_event_ids": [],
        "metadata": metadata,
    }


def _tool_trace_from_audit_event(event: dict[str, Any], stage_context: dict[str, Any], index: int) -> dict[str, Any]:
    trace_id = _string(event.get("trace_id") or event.get("raw_output_ref")) or f"audit-tool-trace::{index}"
    return {
        "id": trace_id,
        "round": _int(event.get("cycle_index") or stage_context.get("round")) or 0,
        "agent": _string(event.get("agent_name") or stage_context.get("agent")),
        "stage": _string(event.get("stage_type") or stage_context.get("stage")),
        "step": _int(event.get("step") or event.get("step_index")),
        "server_id": _string(event.get("server_id")),
        "tool_name": _string(event.get("tool_name")) or "tool",
        "arguments": _mapping(event.get("arguments")),
        "success": _bool(event.get("success")) if event.get("success") is not None else None,
        "exit_code": event.get("exit_code"),
        "summary": _string(event.get("summary")) or "No summary available",
        "stdout_excerpt": _string(event.get("stdout_excerpt") or event.get("stdout")) or "",
        "stderr_excerpt": _string(event.get("stderr_excerpt") or event.get("stderr")) or "",
        "raw_output_ref": _string(event.get("raw_output_ref")),
        "evidence_ids": _event_refs(event),
        "created_at": _string(event.get("at") or event.get("created_at")),
        "timeline_event_ids": [],
        "metadata": event,
    }


def _stage_context_by_round(ag_nodes: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for node in ag_nodes:
        if node["raw_type"] != "STAGE_RESULT":
            continue
        result[node["round"]] = {
            "round": node["round"],
            "agent": node["agent"],
            "stage": _string(node["metadata"].get("stage_type")),
        }
    return result


def _tool_trace_refs_for_node(node: dict[str, Any]) -> list[str]:
    metadata = _mapping(node.get("metadata"))
    refs = []
    for key in ("trace_id", "tool_trace_id", "raw_output_ref"):
        value = _string(metadata.get(key))
        if value:
            refs.append(value)
    return sorted(set(refs))


def _agent_trace_status(nodes: list[dict[str, Any]]) -> str:
    statuses = {node["status"] for node in nodes}
    for status in ("running", "blocked", "failed", "success", "skipped", "pending"):
        if status in statuses:
            return status
    return "pending"


def _handoff_suggestion(nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    node = next((item for item in nodes if item["raw_type"] == "HANDOFF_SUGGESTION"), None)
    return node


def _replan_recommendation(nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    node = next((item for item in nodes if item["type"] == "ReplanStep" or item["raw_type"] in {"PLANNER_DECISION", "STOP_DECISION"}), None)
    return node


def _attack_path(*, ag_nodes: list[dict[str, Any]], ag_edges: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = [node for node in ag_nodes if node["is_main_path"]]
    if not nodes:
        nodes = [
            node
            for node in ag_nodes
            if node["status"] in {"success"} and node["type"] in {"ReconStep", "AnalysisStep", "ValidationStep", "AccessStep", "PivotStep", "GoalCheckStep"}
        ]
    nodes = sorted(nodes, key=lambda item: (item["round"], item["id"]))
    node_ids = {node["id"] for node in nodes}
    edges = [edge for edge in ag_edges if edge["source"] in node_ids and edge["target"] in node_ids]
    return {"nodes": nodes, "edges": edges}


def _kg_type(value: str | None) -> str:
    if not value:
        return "Unknown"
    aliases = {"NetworkZone": "Network", "DataAsset": "Technology", "PrivilegeState": "Identity"}
    normalized = aliases.get(value, value)
    return normalized if normalized in KG_TYPES else "Unknown"


def _kg_status(value: str | None) -> str:
    if not value:
        return "unknown"
    normalized = value.lower()
    aliases = {"candidate": "suspected", "validated": "verified", "succeeded": "verified", "valid": "verified", "revoked": "inactive"}
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in KG_STATUS else "unknown"


def _ag_status(value: str | None) -> str:
    if not value:
        return "pending"
    normalized = value.lower()
    aliases = {
        "succeeded": "success",
        "completed": "success",
        "verified": "success",
        "planned": "pending",
        "not_satisfied": "failed",
        "satisfied": "success",
        "needs_replan": "blocked",
        "need_more_info": "blocked",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in AG_STATUS else "running"


def _ag_type(raw_type: str | None, agent: str, props: dict[str, Any]) -> str:
    raw = raw_type or ""
    if raw == "PLANNER_DECISION" or raw == "HANDOFF_SUGGESTION" or _string(props.get("decision")):
        return "ReplanStep"
    if raw == "BLOCKED_REASON":
        return "FailureStep"
    if raw == "GOAL_CHECK" or agent == "GoalAgent":
        return "GoalCheckStep"
    if raw == "TOOL_CALL":
        return _stage_to_ag_type(_string(props.get("stage_type")))
    if agent == "ReconAgent":
        return "ReconStep"
    if agent == "VulnAnalysisAgent":
        return "AnalysisStep"
    if agent == "ExploitValidationAgent":
        return "ValidationStep"
    if agent == "AccessPivotAgent":
        stage = _string(props.get("stage_type")) or ""
        return "PivotStep" if "PIVOT" in stage else "AccessStep"
    return _stage_to_ag_type(_string(props.get("stage_type"))) or "UnknownStep"


def _stage_to_ag_type(stage: str | None) -> str:
    stage = stage or ""
    if "RECON" in stage:
        return "ReconStep"
    if "VULN" in stage or "ANALYSIS" in stage:
        return "AnalysisStep"
    if "EXPLOIT" in stage or "VALIDATION" in stage:
        return "ValidationStep"
    if "ACCESS" in stage:
        return "AccessStep"
    if "PIVOT" in stage:
        return "PivotStep"
    if "GOAL" in stage:
        return "GoalCheckStep"
    return "UnknownStep"


def _ag_edge_type(value: str) -> str:
    normalized = value.lower()
    aliases = {
        "planned": "derives_from",
        "dispatched_to": "enables",
        "called_tool": "enables",
        "produced_result": "derives_from",
        "supported_by_evidence": "supports",
        "updated_kg": "derives_from",
        "suggested_handoff": "triggers_replan",
        "blocked_by_policy": "blocks",
        "satisfied_goal": "verifies",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in AG_EDGE_TYPES else "unknown"


def _kg_display_name(node_type: str, node_id: str, item: dict[str, Any], props: dict[str, Any]) -> str:
    explicit = _string(item.get("display_name") or props.get("display_name"))
    if explicit:
        return _truncate(explicit, 80)
    target = _target(item, props)
    if node_type == "Host":
        return _string(item.get("hostname") or item.get("address") or props.get("hostname") or props.get("ip") or props.get("address")) or f"Host {node_id}"
    if node_type == "Service":
        port = _string(item.get("port") or props.get("port"))
        service = _string(item.get("service_name") or props.get("service_name") or props.get("service"))
        if target and port:
            return _truncate(f"{target}:{port} {service or ''}".strip(), 80)
        return f"Service {node_id}"
    if node_type == "Finding":
        finding = _string(item.get("finding_name") or props.get("finding_name") or item.get("label"))
        if finding and target:
            return _truncate(f"{finding} on {target}", 80)
        return f"Finding {node_id}"
    if node_type == "Evidence":
        source = _string(props.get("source_name") or props.get("tool") or props.get("source_tool") or item.get("evidence_kind"))
        if source and target:
            return _truncate(f"{source} evidence for {target}", 80)
        return f"Evidence {node_id}"
    if node_type == "Session":
        return _truncate(f"Session on {target}", 80) if target else f"Session {node_id}"
    if node_type == "Credential":
        identity = _string(props.get("identity_or_target") or props.get("principal") or item.get("principal") or target)
        return _truncate(f"Credential for {identity}", 80) if identity else f"Credential {node_id}"
    if node_type == "Goal":
        goal = _string(item.get("description") or props.get("goal_summary") or item.get("label"))
        return _truncate(f"Goal: {goal}", 80) if goal else f"Goal {node_id}"
    return _truncate(_string(item.get("label")) or f"{node_type} {node_id}", 80)


def _ag_display_name(ag_type: str, target: str | None, action_summary: str, result_summary: str, props: dict[str, Any]) -> str:
    explicit = _string(props.get("display_name"))
    if explicit and explicit not in {"Recon", "Exploit", "Finding", "Host", "Service"}:
        return _truncate(explicit, 80)
    result = _truncate(result_summary, 60)
    if ag_type == "ReconStep":
        return _truncate(f"Recon {target or 'target'}: {result}", 80)
    if ag_type == "AnalysisStep":
        return _truncate(f"Analyze {target or 'target'}: {result}", 80)
    if ag_type == "ValidationStep":
        return _truncate(f"Validate {target or 'target'}: {result}", 80)
    if ag_type == "AccessStep":
        return _truncate(f"Access {target or 'target'}: {result}", 80)
    if ag_type == "PivotStep":
        source = _string(props.get("source") or props.get("source_host")) or "source"
        return _truncate(f"Pivot from {source} to {target or 'target'}: {result}", 80)
    if ag_type == "GoalCheckStep":
        return _truncate(f"Goal check: {result}", 80)
    if ag_type == "ReplanStep":
        return _truncate(f"Replan: {action_summary or result}", 80)
    if ag_type == "FailureStep":
        return _truncate(f"Failed: {result}", 80)
    return _truncate(action_summary or result or "Unknown step", 80)


def _evidence_from_kg_node(node: dict[str, Any]) -> dict[str, Any]:
    metadata = _mapping(node.get("metadata"))
    return {
        "id": node["id"],
        "source": _evidence_source(metadata.get("source") or metadata.get("evidence_kind")),
        "source_name": _string(metadata.get("source_name") or metadata.get("tool") or metadata.get("source_tool") or metadata.get("evidence_kind")) or "Evidence",
        "target": node["target"],
        "summary": node["summary"],
        "structured_facts": _list(metadata.get("structured_facts") or metadata.get("parsed_output")),
        "raw_output": _string(metadata.get("raw_output")),
        "linked_kg_node_ids": [node["id"]],
        "linked_ag_node_ids": [],
        "created_by": node["created_by"],
        "round": node["round"],
        "created_at": node["created_at"],
        "metadata": metadata,
    }


def _evidence_stub(evidence_id: str) -> dict[str, Any]:
    return {
        "id": evidence_id,
        "source": "system",
        "source_name": "Evidence reference",
        "target": None,
        "summary": f"Evidence reference {evidence_id}",
        "structured_facts": [],
        "raw_output": None,
        "linked_kg_node_ids": [],
        "linked_ag_node_ids": [],
        "created_by": None,
        "round": None,
        "created_at": None,
        "metadata": {},
    }


def _phase_for_ag(node: dict[str, Any]) -> str:
    raw = node["raw_type"]
    if raw == "PLANNER_DECISION":
        return "planner_decision"
    if raw == "TOOL_CALL":
        return "tool_call"
    if raw == "STAGE_RESULT":
        return "result_apply"
    if raw == "GOAL_CHECK":
        return "goal_check"
    if raw == "BLOCKED_REASON":
        return "error"
    if node["type"] == "ReplanStep":
        return "planner_decision"
    return "agent_execution"


def _phase_order(phase: str) -> int:
    return {
        "planner_decision": 1,
        "agent_execution": 2,
        "tool_call": 3,
        "extraction": 4,
        "result_apply": 5,
        "goal_check": 6,
        "error": 7,
    }.get(phase, 99)


def _agent_state(nodes: list[dict[str, Any]]) -> str:
    if not nodes:
        return "idle"
    statuses = {node["status"] for node in nodes}
    if "running" in statuses:
        return "running"
    if "blocked" in statuses:
        return "blocked"
    if "failed" in statuses:
        return "failed"
    if "success" in statuses:
        return "success"
    if "skipped" in statuses:
        return "skipped"
    return "selected"


def _agent_name(item: dict[str, Any], props: dict[str, Any]) -> str:
    raw = _string(item.get("agent_name") or props.get("agent_name") or props.get("selected_agent") or props.get("agent"))
    aliases = {
        "recon_agent": "ReconAgent",
        "vuln_analysis_agent": "VulnAnalysisAgent",
        "exploit_validation_agent": "ExploitValidationAgent",
        "access_pivot_agent": "AccessPivotAgent",
        "goal_agent": "GoalAgent",
        "planner_agent": "Other",
    }
    return aliases.get((raw or "").lower(), raw or "Other")


def _target_scope(runtime: dict[str, Any]) -> list[Any]:
    execution = _mapping(runtime.get("execution"))
    metadata = _mapping(execution.get("metadata"))
    return _list(metadata.get("target_inventory") or runtime.get("targets") or runtime.get("scope_targets"))


def _current_round(runtime: dict[str, Any]) -> int:
    metadata = _mapping(_mapping(runtime.get("execution")).get("metadata"))
    candidates = [
        metadata.get("current_round"),
        metadata.get("cycle_index"),
        _mapping(metadata.get("last_control_cycle")).get("cycle_index"),
    ]
    for value in candidates:
        number = _int(value)
        if number is not None:
            return number
    return 0


def _goal_status(runtime: dict[str, Any]) -> str:
    metadata = _mapping(_mapping(runtime.get("execution")).get("metadata"))
    goal = runtime.get("goal_satisfied") or metadata.get("goal_satisfied")
    if goal is True or goal == "true":
        return "satisfied"
    if goal is False or goal == "false":
        return "not_satisfied"
    return _string(metadata.get("goal_status")) or "unknown"


def _runtime_payload(runtime_state: RuntimeState | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(runtime_state, RuntimeState):
        return runtime_state.model_dump(mode="json")
    return dict(runtime_state or {})


def _merged_properties(item: dict[str, Any]) -> dict[str, Any]:
    props = dict(item)
    nested = _mapping(item.get("properties"))
    props.update(nested)
    return props


def _summary(item: dict[str, Any], props: dict[str, Any]) -> str:
    return _string(props.get("summary") or item.get("summary") or item.get("label")) or "No summary available"


def _target(item: dict[str, Any], props: dict[str, Any]) -> str | None:
    for key in ("target", "visual_target", "host", "address", "url", "endpoint", "service_id", "asset_id"):
        value = item.get(key) if key in item else props.get(key)
        if value:
            return str(value)
    refs = _list(item.get("refs") or props.get("refs") or item.get("source_refs"))
    if refs:
        first = refs[0]
        if isinstance(first, dict):
            return _string(first.get("label") or first.get("ref_id") or first.get("entity_id"))
        return _string(first)
    return None


def _evidence_ids(item: dict[str, Any], props: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("evidence_ids", "evidence_refs"):
        for value in _list(item.get(key) or props.get(key)):
            if value:
                values.append(str(value))
    evidence_chain = _mapping(item.get("evidence_chain") or props.get("evidence_chain"))
    for value in _list(evidence_chain.get("evidence_ids")):
        if value:
            values.append(str(value))
    return sorted(set(values))


def _kg_ref_ids(item: dict[str, Any], props: dict[str, Any]) -> list[str]:
    refs = []
    for key in ("refs", "source_refs", "subject_refs", "created_from"):
        refs.extend(_list(item.get(key) or props.get(key)))
    result = []
    for ref in refs:
        if isinstance(ref, dict):
            graph = _string(ref.get("graph") or ref.get("entity_kind"))
            if graph and graph.lower() not in {"kg", "node"}:
                continue
            value = _string(ref.get("ref_id") or ref.get("entity_id") or ref.get("id"))
            if value:
                result.append(value)
        elif ref:
            result.append(str(ref))
    return sorted(set(result))


def _event_refs(event: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("evidence_ref", "evidence_refs", "artifact_ref", "artifact_refs", "payload_ref", "raw_output_ref"):
        refs.extend(str(item) for item in _list(event.get(key)) if item)
    return sorted(set(refs))


def _runtime_event_is_error(event: dict[str, Any]) -> bool:
    text = f"{event.get('event_type', '')} {event.get('summary', '')}".lower()
    return any(part in text for part in ("error", "failed", "blocked", "denied"))


def _evidence_source(value: Any) -> str:
    text = _string(value).lower() if _string(value) else ""
    for source in ("tool", "agent", "planner", "extractor", "user", "system"):
        if source in text:
            return source
    return "system"


def _priority(value: Any) -> str:
    text = (_string(value) or "").lower()
    return text if text in {"low", "medium", "high"} else "medium"


def _humanize(value: str) -> str:
    return value.replace("_", " ").strip().title() or "Unknown"


def _truncate(value: str, limit: int) -> str:
    text = value.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, set):
        return list(value)
    return [value]


def _string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _bool(value: Any) -> bool:
    return value is True or value in {"true", "True", "1", 1, "yes"}


__all__ = ["build_unified_visualization"]
