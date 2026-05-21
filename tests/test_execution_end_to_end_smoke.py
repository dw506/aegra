from __future__ import annotations

import sys
from typing import Any

from src.core.agents.agent_protocol import AgentContext, AgentInput, GraphRef, GraphScope
from src.core.agents.perception import PerceptionAgent
from src.core.execution import ExecutionExecutor, LocalShellAdapter
from src.core.models.ag import GraphRef as EventGraphRef
from src.core.models.events import (
    AgentResultStatus,
    AgentRole,
    AgentTaskResult,
    EvidenceArtifact,
    FactWriteRequest,
    ObservationRecord,
    ProjectionRequest,
    RuntimeControlRequest,
)
from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntime
from src.core.models.tg import TaskType
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.workers.access_validation_worker import AccessValidationWorker
from src.core.workers.base import WorkerTaskSpec
from src.core.workers.services.access_validation_service import AccessValidationService


def test_access_execution_perception_and_runtime_audit_smoke() -> None:
    state = RuntimeState(operation_id="op-e2e-smoke", execution=OperationRuntime(operation_id="op-e2e-smoke"))
    state.register_task(TaskRuntime(task_id="task-access-1", tg_node_id="task-access-1"))
    target_ref = GraphRef(graph=GraphScope.KG, ref_id="host-1", ref_type="Host")
    agent_input = AgentInput(
        graph_refs=[target_ref],
        task_ref="task-access-1",
        context=AgentContext(operation_id=state.operation_id),
        raw_payload={
            "task_type": TaskType.IDENTITY_CONTEXT_CONFIRMATION.value,
            "task_label": "Access validation smoke",
            "metadata": {
                "require_session": True,
                "execution_adapter": "local_shell",
                "probe_command": [sys.executable, "-c", "print('ok')"],
            },
        },
    )
    task_spec = WorkerTaskSpec(
        task_id="task-access-1",
        task_type=TaskType.IDENTITY_CONTEXT_CONFIRMATION.value,
        target_refs=[target_ref],
    )
    worker = AccessValidationWorker(
        service=AccessValidationService(executor=ExecutionExecutor([LocalShellAdapter()]))
    )

    worker_result = worker.run(agent_input)

    assert worker_result.success is True
    worker_outcome = worker_result.output.outcomes[0]
    worker_payload = dict(worker_outcome["payload"])
    assert worker_payload["tool_execution"]["adapter"] == "local_shell"
    assert worker_payload["tool_execution"]["tool"] == "session_probe"
    assert worker_payload["tool_execution"]["success"] is True

    perception_result = PerceptionAgent().run(
        AgentInput(
            graph_refs=[target_ref],
            task_ref=task_spec.task_id,
            context=AgentContext(operation_id=state.operation_id),
            raw_payload={
                "outcome": worker_outcome,
                "raw_result": worker_payload,
            },
        )
    )

    assert perception_result.success is True
    assert any("perception parser=tool_execution_parser" in log for log in perception_result.output.logs)
    assert any(
        observation["payload"].get("category") == "tool_execution"
        for observation in perception_result.output.observations
    )
    assert any(
        evidence["payload"].get("kind") == "tool_execution_evidence"
        for evidence in perception_result.output.evidence
    )

    canonical_result = AgentTaskResult(
        request_id="request-access-smoke",
        agent_role=AgentRole.ACCESS_WORKER,
        operation_id=state.operation_id,
        task_id=task_spec.task_id,
        tg_node_id=task_spec.task_id,
        status=AgentResultStatus.SUCCEEDED,
        summary=str(worker_outcome["summary"]),
        observations=[_observation_record(item) for item in perception_result.output.observations],
        evidence=[_evidence_artifact(item, tool_execution=worker_payload["tool_execution"]) for item in perception_result.output.evidence],
        fact_write_requests=[FactWriteRequest.model_validate(item) for item in worker_payload["fact_write_requests"]],
        projection_requests=[ProjectionRequest.model_validate(item) for item in worker_payload["projection_requests"]],
        runtime_requests=[RuntimeControlRequest.model_validate(item) for item in worker_payload["runtime_requests"]],
        outcome_payload=worker_payload,
    )

    apply_result = PhaseTwoResultApplier().apply(canonical_result, state)

    audit_log = state.execution.metadata["audit_log"]
    tool_audits = [entry for entry in audit_log if entry["event_type"] == "tool_execution_recorded"]
    assert len(tool_audits) == 1
    assert tool_audits[0]["adapter"] == "local_shell"
    assert tool_audits[0]["tool"] == "session_probe"
    assert tool_audits[0]["stdout_excerpt"].strip() == "ok"
    assert state.execution.tasks[task_spec.task_id].status.value == "succeeded"
    assert apply_result.kg_state_deltas


def _observation_record(item: dict[str, Any]) -> ObservationRecord:
    payload = dict(item.get("payload", {}))
    return ObservationRecord(
        category=str(payload.get("category") or payload.get("branch") or "perception"),
        summary=str(item["summary"]),
        confidence=float(item.get("confidence", 0.5)),
        refs=_event_refs(item.get("refs", [])),
        payload=payload,
    )


def _evidence_artifact(item: dict[str, Any], *, tool_execution: dict[str, Any]) -> EvidenceArtifact:
    payload = dict(item.get("payload", {}))
    metadata = {"perception_payload": payload}
    if payload.get("kind") == "tool_execution_evidence":
        metadata["tool_execution"] = dict(tool_execution)
    return EvidenceArtifact(
        kind=str(payload.get("kind") or payload.get("evidence_kind") or "perception_evidence"),
        summary=str(item["summary"]),
        payload_ref=str(item.get("payload_ref") or "runtime://perception/evidence"),
        refs=_event_refs(item.get("refs", [])),
        metadata=metadata,
    )


def _event_refs(items: list[dict[str, Any]]) -> list[EventGraphRef]:
    refs: list[EventGraphRef] = []
    for item in items:
        refs.append(
            EventGraphRef(
                graph=str(item.get("graph", "kg")),
                ref_id=str(item["ref_id"]),
                ref_type=item.get("ref_type"),
            )
        )
    return refs
