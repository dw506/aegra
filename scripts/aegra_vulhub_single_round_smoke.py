from __future__ import annotations

import json
import uuid
import urllib.request
import urllib.error
from urllib.parse import urljoin

from src.core.models.ag import GraphRef
from src.core.models.events import (
    AgentResultStatus,
    AgentRole,
    AgentTaskResult,
    EvidenceArtifact,
    FactWriteKind,
    FactWriteRequest,
    ObservationRecord,
    RuntimeBudgetDelta,
    RuntimeControlRequest,
    RuntimeControlType,
)
from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntime
from src.core.models.tg import TaskGraph, TaskNode, TaskStatus, TaskType
from src.core.runtime.result_applier import PhaseTwoResultApplier


OPERATION_ID = "op-vulhub-tomcat-12615-auto"
TASK_ID = "task-detect-tomcat-cve-2017-12615"
TARGET_URL = "http://host.docker.internal:8081/"
SERVICE_ID = "svc-vulhub-tomcat-8081"
VULN_ID = f"vuln::CVE-2017-12615::{SERVICE_ID}"


def safe_probe(target_url: str) -> dict:
    marker = "aegra-safe-marker-" + uuid.uuid4().hex[:8]
    put_url = urljoin(target_url, marker + ".jsp/")
    get_url = urljoin(target_url, marker + ".jsp")

    result = {
        "layer": "MCP Tool Layer",
        "tool_name": "vuln.tomcat_cve_2017_12615_probe",
        "target_url": target_url,
        "put_url": put_url,
        "get_url": get_url,
        "marker": marker,
        "validated": False,
        "confidence": 0.0,
        "steps": [],
    }

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        req = urllib.request.Request(target_url, method="OPTIONS")
        with opener.open(req, timeout=5) as r:
            result["steps"].append({
                "step": "OPTIONS",
                "status": r.status,
                "allow": r.headers.get("Allow", ""),
            })
    except Exception as e:
        result["steps"].append({"step": "OPTIONS", "error": str(e)})

    try:
        req = urllib.request.Request(put_url, data=marker.encode(), method="PUT")
        with opener.open(req, timeout=5) as r:
            result["steps"].append({"step": "PUT_STATIC_MARKER", "status": r.status})
    except urllib.error.HTTPError as e:
        result["steps"].append({"step": "PUT_STATIC_MARKER", "status": e.code, "error": str(e)})
    except Exception as e:
        result["steps"].append({"step": "PUT_STATIC_MARKER", "error": str(e)})

    try:
        with opener.open(get_url, timeout=5) as r:
            body = r.read(512).decode("utf-8", "ignore")
            matched = marker in body
            result["steps"].append({"step": "GET_MARKER", "status": r.status, "matched": matched})
            result["validated"] = matched
            result["confidence"] = 0.92 if matched else 0.35
            result["summary"] = (
                "CVE-2017-12615 validated by safe static marker retrieval"
                if matched
                else "marker retrieval did not validate CVE-2017-12615"
            )
    except urllib.error.HTTPError as e:
        result["steps"].append({"step": "GET_MARKER", "status": e.code, "error": str(e)})
        result["summary"] = "CVE-2017-12615 not validated by marker retrieval"
        result["confidence"] = 0.2
    except Exception as e:
        result["steps"].append({"step": "GET_MARKER", "error": str(e)})
        result["summary"] = "CVE-2017-12615 probe failed before completion"
        result["confidence"] = 0.0

    return result


def critic_agent(tool_result: dict) -> dict:
    if tool_result["validated"]:
        return {
            "layer": "CriticAgent",
            "decision": "accept",
            "quality": "high",
            "confidence": tool_result["confidence"],
            "retry": False,
            "replan": False,
            "reason": "safe PUT marker and GET verification succeeded",
        }

    return {
        "layer": "CriticAgent",
        "decision": "accept",
        "quality": "medium",
        "confidence": tool_result["confidence"],
        "retry": False,
        "replan": False,
        "reason": "target was tested, but marker verification did not confirm the vulnerability",
    }


def build_result(tool_result: dict, critic: dict) -> AgentTaskResult:
    service_ref = GraphRef(graph="kg", ref_id=SERVICE_ID, ref_type="Service", label="Vulhub Tomcat 8081")
    vuln_ref = GraphRef(graph="kg", ref_id=VULN_ID, ref_type="Vulnerability", label="Tomcat CVE-2017-12615")
    refs = [service_ref, vuln_ref] if tool_result["validated"] else [service_ref]

    evidence = EvidenceArtifact(
        kind="vulnerability_validation",
        summary=tool_result["summary"],
        payload_ref=f"runtime://worker-results/{TASK_ID}",
        refs=refs,
        metadata={
            "tool_result": tool_result,
            "critic": critic,
            "safe_payload_summary": "static marker only; no command execution payload",
        },
    )

    observation = ObservationRecord(
        category="vulnerability_validation",
        summary=tool_result["summary"],
        confidence=float(critic["confidence"]),
        refs=refs,
        payload={
            "service_id": SERVICE_ID,
            "vulnerability_id": VULN_ID,
            "cve": "CVE-2017-12615",
            "validated": tool_result["validated"],
            "critic": critic,
        },
    )

    fact_writes = []
    if tool_result["validated"]:
        fact_writes = [
            FactWriteRequest(
                kind=FactWriteKind.ENTITY_UPSERT,
                source_task_id=TASK_ID,
                subject_ref=vuln_ref,
                attributes={
                    "vulnerability_name": "Apache Tomcat CVE-2017-12615",
                    "validation_status": "validated",
                    "cve": "CVE-2017-12615",
                    "summary": tool_result["summary"],
                    "confidence": critic["confidence"],
                    "safe_payload_summary": "static marker only; no command execution payload",
                },
                confidence=float(critic["confidence"]),
                evidence_ids=[evidence.evidence_id],
                summary=f"Validated {VULN_ID}",
            ),
            FactWriteRequest(
                kind=FactWriteKind.RELATION_UPSERT,
                source_task_id=TASK_ID,
                subject_ref=service_ref,
                relation_type="HAS_VULNERABILITY",
                object_ref=vuln_ref,
                attributes={"validation_status": "validated", "cve": "CVE-2017-12615"},
                confidence=float(critic["confidence"]),
                evidence_ids=[evidence.evidence_id],
                summary=f"{SERVICE_ID} has CVE-2017-12615",
            ),
        ]

    return AgentTaskResult(
        request_id=f"request::{TASK_ID}",
        agent_role=AgentRole.VULNERABILITY_VALIDATION_WORKER,
        operation_id=OPERATION_ID,
        task_id=TASK_ID,
        tg_node_id=TASK_ID,
        status=AgentResultStatus.SUCCEEDED,
        summary=tool_result["summary"],
        observations=[observation],
        evidence=[evidence],
        fact_write_requests=fact_writes,
        runtime_requests=[
            RuntimeControlRequest(
                request_type=RuntimeControlType.CONSUME_BUDGET,
                source_task_id=TASK_ID,
                budget_delta=RuntimeBudgetDelta(operations=1, noise=0.1, risk=0.1),
                reason="single safe Vulhub vulnerability validation",
            )
        ],
        outcome_payload={
            "validated": tool_result["validated"],
            "status": "validated" if tool_result["validated"] else "not_detected",
            "confidence": critic["confidence"],
            "service_id": SERVICE_ID,
            "vulnerability_id": VULN_ID,
            "critic": critic,
            "tool_result": tool_result,
        },
    )


def main() -> None:
    trace = []

    state = RuntimeState(
        operation_id=OPERATION_ID,
        execution=OperationRuntime(operation_id=OPERATION_ID),
    )
    state.execution.metadata["runtime_policy"] = {
        "authorized_hosts": ["host.docker.internal", "127.0.0.1", "localhost"],
        "authorized_ports": [8081],
        "max_rounds": 1,
        "allow_safe_file_write_probe": True,
        "allow_command_execution_payload": False,
    }
    state.execution.metadata["control_plane"] = {"audit_redaction_enabled": True}
    state.register_task(TaskRuntime(task_id=TASK_ID, tg_node_id=TASK_ID))

    service_ref = GraphRef(graph="kg", ref_id=SERVICE_ID, ref_type="Service", label="Vulhub Tomcat 8081")

    tg = TaskGraph()
    tg.add_node(
        TaskNode(
            id=TASK_ID,
            label="Detect Tomcat CVE-2017-12615",
            task_type=TaskType.VULNERABILITY_VALIDATION,
            status=TaskStatus.READY,
            source_action_id="manual-vulhub-12615",
            input_bindings={
                "target_url": TARGET_URL,
                "service_id": SERVICE_ID,
                "port": 8081,
                "cve": "CVE-2017-12615",
            },
            target_refs=[service_ref],
            resource_keys={f"service:{SERVICE_ID}"},
            estimated_risk=0.1,
            estimated_noise=0.1,
            goal_relevance=0.95,
        )
    )

    trace.append({"layer": "Graph State Layer", "status": "initialized"})
    trace.append({"layer": "PlannerAgent", "decision": "create one vulnerability_validation TG task"})
    trace.append({"layer": "SchedulerAgent", "decision": f"schedule {TASK_ID}"})

    tg.transition_task(TASK_ID, TaskStatus.QUEUED, reason="SchedulerAgent selected task")
    tg.transition_task(TASK_ID, TaskStatus.RUNNING, reason="LLMWorkerAgent started task")

    trace.append({
        "layer": "LLMWorkerAgent",
        "action": "call_mcp_tool",
        "tool_name": "vuln.tomcat_cve_2017_12615_probe",
    })

    tool_result = safe_probe(TARGET_URL)
    trace.append(tool_result)

    critic = critic_agent(tool_result)
    trace.append(critic)

    agent_task_result = build_result(tool_result, critic)

    applied = PhaseTwoResultApplier().apply(
        agent_task_result,
        state,
        task_graph=tg,
    )

    output = {
        "operation_id": OPERATION_ID,
        "target_url": TARGET_URL,
        "trace": trace,
        "worker_result": {
            "status": agent_task_result.status.value,
            "summary": agent_task_result.summary,
            "outcome_payload": agent_task_result.outcome_payload,
        },
        "result_applier": {
            "kg_state_delta_count": len(applied.kg_state_deltas),
            "runtime_event_count": len(applied.runtime_event_refs),
            "kg_state_deltas": applied.kg_state_deltas,
            "tg_graph": applied.tg_graph,
        },
        "runtime_audit": state.execution.metadata.get("audit_log", []),
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print()
    print("=== AEGRA_SINGLE_ROUND_RESULT ===")
    print("PASS" if agent_task_result.status.value == "succeeded" else "CHECK")
    print(f"validated={tool_result['validated']}")
    print(f"critic_confidence={critic['confidence']}")
    print(f"kg_delta_count={len(applied.kg_state_deltas)}")


if __name__ == "__main__":
    main()