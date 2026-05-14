from __future__ import annotations

from src.core.agents.agent_protocol import GraphRef as ProtocolGraphRef, GraphScope
from src.core.models.ag import GraphRef
from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntime
from src.core.models.tg import TaskNode, TaskType
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.workers.vulnerability_validation_worker import VulnerabilityValidationWorker


def _state() -> RuntimeState:
    state = RuntimeState(operation_id="op-findings", execution=OperationRuntime(operation_id="op-findings"))
    state.register_task(TaskRuntime(task_id="task-vuln", tg_node_id="task-vuln"))
    state.execution.metadata["control_plane"] = {"audit_redaction_enabled": True}
    return state


def _task() -> TaskNode:
    service_id = "127.0.0.1:8080/tcp"
    return TaskNode(
        id="task-vuln",
        label="Validate Struts2 S2-045",
        task_type=TaskType.VULNERABILITY_VALIDATION,
        source_action_id="action-vuln",
        input_bindings={"host_id": "127.0.0.1", "port": 8080, "service_id": service_id},
        target_refs=[GraphRef(graph="kg", ref_id=service_id, ref_type="Service", label=service_id)],
        resource_keys={"host:127.0.0.1"},
    )


def _apply_validation(status: str) -> RuntimeState:
    state = _state()
    request = VulnerabilityValidationWorker().build_request(
        task=_task(),
        operation_id=state.operation_id,
        metadata={
            "target_url": "http://127.0.0.1:8080/",
            "vulnerability_validator_output": {
                "status": status,
                "confidence": 0.91,
                "cvss": 9.8,
                "epss": 0.8,
                "kev": True,
                "summary": f"Struts2 S2-045 {status}",
            },
        },
    )
    result = VulnerabilityValidationWorker().execute_task(request)
    PhaseTwoResultApplier().apply(
        result,
        state,
        kg_ref=ProtocolGraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
    )
    return state


def test_validated_vulnerability_promotes_to_finding_with_evidence_chain() -> None:
    state = _apply_validation("validated")

    finding = state.execution.metadata["findings"][0]
    evidence = state.execution.metadata["evidence_artifacts"][0]

    assert finding["validation_status"] == "validated"
    assert finding["service_ref"] == "127.0.0.1:8080/tcp"
    assert finding["vulnerability_ref"].startswith("vuln::struts2-s2-045")
    assert finding["evidence_refs"] == [evidence["evidence_id"]]
    assert evidence["task_ref"] == "task-vuln"
    assert evidence["tool_output_ref"].startswith("runtime://worker-results/")
    assert finding["provenance"]["task_ref"] == "task-vuln"
    assert finding["risk_score"]["score"] > 0


def test_not_detected_does_not_generate_finding() -> None:
    state = _apply_validation("not_detected")

    assert state.execution.metadata.get("findings", []) == []


def test_blocked_generates_audit_but_no_vulnerability_finding() -> None:
    state = _state()
    request = VulnerabilityValidationWorker().build_request(
        task=_task(),
        operation_id=state.operation_id,
        metadata={"target_url": "http://example.com:8080/"},
    )
    result = VulnerabilityValidationWorker().execute_task(request)
    PhaseTwoResultApplier().apply(
        result,
        state,
        kg_ref=ProtocolGraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
    )

    assert state.execution.metadata.get("findings", []) == []
    assert state.execution.metadata["finding_audit"][0]["event_type"] == "validation_blocked"
