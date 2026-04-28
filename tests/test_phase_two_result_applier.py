from __future__ import annotations

import sys

from src.core.agents.agent_protocol import (
    AgentContext,
    AgentInput,
    AgentKind,
    AgentOutput,
    GraphRef as ProtocolGraphRef,
    GraphScope,
)
from src.core.models.ag import GraphRef
from src.core.models.events import (
    AgentResultStatus,
    AgentRole,
    AgentTaskResult,
    CheckpointHint,
    CriticSignal,
    CriticSignalSeverity,
    EvidenceArtifact,
    FactWriteKind,
    FactWriteRequest,
    ObservationRecord,
    ProjectionRequest,
    ProjectionRequestKind,
    ReplanHint,
    ReplanScope,
    RuntimeBudgetDelta,
    RuntimeControlRequest,
    RuntimeControlType,
)
from src.core.models.runtime import (
    CredentialKind,
    CredentialRuntime,
    CredentialStatus,
    OperationRuntime,
    PivotRouteRuntime,
    PivotRouteStatus,
    ResourceLock,
    RuntimeState,
    SessionLeaseRuntime,
    SessionRuntime,
    SessionStatus,
    TaskRuntime,
    TaskRuntimeStatus,
    WorkerRuntime,
    WorkerStatus,
    utc_now,
)
from src.core.models.tg import TaskNode, TaskType
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.workers.access_worker import AccessWorker
from src.core.workers.goal_worker import GoalWorker
from src.core.workers.recon_worker import ReconWorker


def build_state() -> RuntimeState:
    state = RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))
    state.register_task(TaskRuntime(task_id="task-1", tg_node_id="task-1"))
    return state


def build_task(task_type: TaskType) -> TaskNode:
    return TaskNode(
        id="task-1",
        label="Task Label",
        task_type=task_type,
        source_action_id="action-1",
        input_bindings={"host_id": "host-1"},
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host", label="host-1")],
        resource_keys={"host:host-1"},
    )


def test_result_applier_routes_runtime_kg_and_projection_effects() -> None:
    applier = PhaseTwoResultApplier()
    state = build_state()
    result = AgentTaskResult(
        request_id="request-1",
        agent_role=AgentRole.RECON_WORKER,
        operation_id="op-1",
        task_id="task-1",
        tg_node_id="task-1",
        status=AgentResultStatus.SUCCEEDED,
        summary="recon completed",
        observations=[
            ObservationRecord(
                category="recon",
                summary="host observed",
                confidence=0.8,
                refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host", label="host-1")],
                payload={"reachable": True},
            )
        ],
        evidence=[
            EvidenceArtifact(
                kind="recon_result",
                summary="nmap output",
                payload_ref="runtime://worker-results/recon/host-1",
                refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host", label="host-1")],
                metadata={"tool": {"category": "success"}},
            )
        ],
        fact_write_requests=[
            FactWriteRequest(
                kind=FactWriteKind.ASSERTION,
                source_task_id="task-1",
                subject_ref=GraphRef(graph="kg", ref_id="host-1", ref_type="Host", label="host-1"),
                attributes={"reachable": True},
                confidence=0.9,
                summary="Host reachability asserted",
            )
        ],
        projection_requests=[
            ProjectionRequest(
                kind=ProjectionRequestKind.REFRESH_LOCAL_FRONTIER,
                source_task_id="task-1",
                reason="new recon facts available",
                target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host", label="host-1")],
            )
        ],
        runtime_requests=[
            RuntimeControlRequest(
                request_type=RuntimeControlType.OPEN_SESSION,
                source_task_id="task-1",
                lease_seconds=120,
                reason="session needed",
                metadata={"bound_target": "host-1", "bound_identity": "operator"},
            ),
            RuntimeControlRequest(
                request_type=RuntimeControlType.CONSUME_BUDGET,
                source_task_id="task-1",
                budget_delta=RuntimeBudgetDelta(operations=1, noise=0.2),
                reason="consume budget",
            ),
        ],
        checkpoint_hints=[
            CheckpointHint(
                checkpoint_id="cp-1",
                source_task_id="task-1",
                summary="checkpoint after recon",
                created_after_tasks=["task-1"],
            )
        ],
        replan_hints=[
            ReplanHint(
                hint_id="hint-1",
                source_task_id="task-1",
                scope=ReplanScope.LOCAL,
                reason="follow-up planning needed",
                task_ids=["task-1"],
            )
        ],
        critic_signals=[
            CriticSignal(
                signal_id="critic-1",
                source_task_id="task-1",
                kind="new_recon_fact",
                severity=CriticSignalSeverity.MEDIUM,
                reason="new fact may affect planning",
                task_ids=["task-1"],
            )
        ],
    )

    applied = applier.apply(
        result,
        state,
        kg_ref=ProtocolGraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
    )

    assert state.sessions
    session = next(iter(state.sessions.values()))
    assert session.bound_target == "host-1"
    assert state.execution.tasks["task-1"].metadata["session_id"] == session.session_id
    assert state.execution.tasks["task-1"].metadata["session_lease_id"] == f"lease::{session.session_id}::task-1"
    assert state.session_leases[f"lease::{session.session_id}::task-1"].session_id == session.session_id
    assert state.budgets.operation_budget_used == 1
    assert state.budgets.noise_budget_used == 0.2
    assert "cp-1" in state.checkpoints
    assert state.replan_requests
    assert state.execution.metadata["critic_signals"][0]["kind"] == "new_recon_fact"
    assert state.recent_outcomes[0].task_id == "task-1"
    assert state.recent_outcomes[0].payload_ref == "runtime://worker-results/recon/host-1"

    assert applied.state_writer_result is not None
    assert applied.graph_projection_result is not None
    assert applied.kg_event_batch is not None
    assert len(applied.kg_state_deltas) >= 3
    assert applied.ag_state_deltas
    assert {delta["scope"] for delta in applied.kg_state_deltas} == {"kg"}
    assert {delta["scope"] for delta in applied.ag_state_deltas} == {"ag"}
    assert {ref.event_type for ref in applied.runtime_event_refs} >= {
        "SessionOpened",
        "BudgetConsumed",
        "CheckpointCreated",
        "ReplanRequested",
    }
    assert any(entry["event_type"] == "tool_invocation" for entry in state.execution.metadata["audit_log"])
    assert any(entry["event_type"] == "fact_write" for entry in state.execution.metadata["audit_log"])
    assert any(entry["event_type"] == "evidence_chain" for entry in state.execution.metadata["audit_log"])


def test_result_applier_converts_relation_fact_write_into_kg_relation_delta() -> None:
    applier = PhaseTwoResultApplier()
    state = build_state()
    result = AgentTaskResult(
        request_id="request-2",
        agent_role=AgentRole.ACCESS_WORKER,
        operation_id="op-1",
        task_id="task-1",
        tg_node_id="task-1",
        status=AgentResultStatus.SUCCEEDED,
        summary="access completed",
        fact_write_requests=[
            FactWriteRequest(
                kind=FactWriteKind.RELATION_UPSERT,
                source_task_id="task-1",
                subject_ref=GraphRef(graph="kg", ref_id="identity-1", ref_type="Identity", label="alice"),
                relation_type="AUTHENTICATES_AS",
                object_ref=GraphRef(graph="kg", ref_id="cred-1", ref_type="Credential", label="cred"),
                attributes={"validated": True},
                confidence=0.95,
                summary="identity authenticates with credential",
            )
        ],
    )

    applied = applier.apply(
        result,
        state,
        kg_ref=ProtocolGraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
    )

    assert applied.state_writer_result is None
    assert applied.graph_projection_result is None
    assert applied.kg_event_batch is not None
    assert len(applied.kg_state_deltas) == 1
    delta = applied.kg_state_deltas[0]
    assert delta["delta_type"] == "upsert_relation"
    assert delta["patch"]["relation_type"] == "AUTHENTICATES_AS"
    assert delta["patch"]["source"] == "identity-1"
    assert delta["patch"]["target"] == "cred-1"
    assert delta["patch"]["attributes"]["evidence_chain"]["source_task_id"] == "task-1"
    assert any(entry["event_type"] == "evidence_chain" for entry in state.execution.metadata["audit_log"])


def test_result_applier_accepts_worker_agent_output_via_canonical_adapter() -> None:
    applier = PhaseTwoResultApplier()
    state = build_state()
    agent_input = AgentInput(
        graph_refs=[ProtocolGraphRef(graph=GraphScope.KG, ref_id="host-1", ref_type="Host")],
        task_ref="task-1",
        decision_ref="decision-1",
        context=AgentContext(operation_id="op-1"),
        raw_payload={"task_type": "service_validation"},
    )
    agent_output = AgentOutput(
        evidence=[
            {
                "task_id": "task-1",
                "source_agent": "compat_worker",
                "result_type": "probe_result",
                "summary": "probe completed",
                "payload_ref": "runtime://worker-results/task-1",
                "refs": [ProtocolGraphRef(graph=GraphScope.KG, ref_id="host-1", ref_type="Host").model_dump(mode="json")],
                "extra": {"tool": {"category": "success"}},
            }
        ],
        outcomes=[
            {
                "task_id": "task-1",
                "source_agent": "compat_worker",
                "outcome_type": "execution_result",
                "success": True,
                "summary": "probe completed",
                "raw_result_ref": "runtime://worker-results/task-1",
                "confidence": 0.8,
                "refs": [ProtocolGraphRef(graph=GraphScope.KG, ref_id="host-1", ref_type="Host").model_dump(mode="json")],
                "payload": {"status": "ok"},
            }
        ],
    )

    applied = applier.apply(
        agent_output,
        state,
        agent_input=agent_input,
        agent_name="compat_worker",
        agent_kind=AgentKind.WORKER,
        kg_ref=ProtocolGraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
    )

    assert applied.state_writer_result is not None
    assert applied.kg_state_deltas
    assert any(entry["event_type"] == "tool_invocation" for entry in state.execution.metadata["audit_log"])


def test_result_applier_audit_log_redacts_sensitive_payloads() -> None:
    applier = PhaseTwoResultApplier()
    state = build_state()
    state.execution.metadata["control_plane"] = {
        "audit_redaction_enabled": True,
        "audit_max_entries": 10,
    }
    result = AgentTaskResult(
        request_id="request-redaction",
        agent_role=AgentRole.RECON_WORKER,
        operation_id="op-1",
        task_id="task-1",
        tg_node_id="task-1",
        status=AgentResultStatus.SUCCEEDED,
        summary="runtime token=top-secret " + ("x" * 320),
        evidence=[
            EvidenceArtifact(
                kind="tool_result",
                summary="curl output",
                payload_ref="runtime://worker-results/task-1",
                refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host", label="host-1")],
                metadata={
                    "tool": "curl",
                    "command": "curl -H 'Authorization: Bearer super-secret-token' password=pw-1 api_key=key-1 "
                    + ("y" * 700),
                },
            )
        ],
        fact_write_requests=[
            FactWriteRequest(
                kind=FactWriteKind.ASSERTION,
                source_task_id="task-1",
                subject_ref=GraphRef(graph="kg", ref_id="host-1", ref_type="Host", label="host-1"),
                attributes={"reachable": True},
                confidence=0.8,
                summary="fact token=raw-secret " + ("z" * 320),
            )
        ],
    )

    applier.apply(
        result,
        state,
        kg_ref=ProtocolGraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
    )

    audit_log = state.execution.metadata["audit_log"]
    tool_entry = next(entry for entry in audit_log if entry["event_type"] == "tool_invocation")
    fact_entry = next(entry for entry in audit_log if entry["event_type"] == "fact_write")
    evidence_entry = next(entry for entry in audit_log if entry["event_type"] == "evidence_chain")

    assert "[REDACTED]" in tool_entry["command"]
    assert "super-secret-token" not in tool_entry["command"]
    assert "pw-1" not in tool_entry["command"]
    assert "(truncated" in tool_entry["command"]
    assert "[REDACTED]" in fact_entry["summary"]
    assert "raw-secret" not in fact_entry["summary"]
    assert "(truncated" in fact_entry["summary"]
    assert "[REDACTED]" in evidence_entry["evidence_chain"]["summary"]
    assert "raw-secret" not in evidence_entry["evidence_chain"]["summary"]


def test_worker_results_apply_into_formal_kg_entity_and_relation_updates() -> None:
    applier = PhaseTwoResultApplier()
    state = build_state()

    recon_worker = ReconWorker()
    recon_request = recon_worker.build_request(
        task=build_task(TaskType.SERVICE_VALIDATION),
        operation_id="op-1",
        metadata={
            "tool_command": [
                sys.executable,
                "-c",
                (
                    "import json; "
                    "print(json.dumps({'summary':'service probe ok','reachable':True,"
                    "'confidence':0.91,"
                    "'entities':[{'id':'host-1','type':'Host'},{'id':'host-1:22/tcp','type':'Service','host_id':'host-1','port':22,'protocol':'tcp'}],"
                    "'relations':[{'type':'HOSTS','source':'host-1','target':'host-1:22/tcp','attributes':{'port':22,'protocol':'tcp'}}],"
                    "'service':{'id':'host-1:22/tcp','port':22,'banner':'ssh','protocol':'tcp'}}))"
                ),
            ],
        },
    )
    recon_result = recon_worker.execute_task(recon_request)
    assert {item.kind.value for item in recon_result.fact_write_requests} >= {"entity_upsert", "relation_upsert"}
    recon_applied = applier.apply(
        recon_result,
        state,
        kg_ref=ProtocolGraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
    )
    assert any(delta["patch"]["entity_type"] == "Host" for delta in recon_applied.kg_state_deltas if delta["delta_type"] == "upsert_entity")
    assert any(delta["patch"]["entity_type"] == "Service" for delta in recon_applied.kg_state_deltas if delta["delta_type"] == "upsert_entity")
    assert any(delta["patch"]["relation_type"] == "HOSTS" for delta in recon_applied.kg_state_deltas if delta["delta_type"] == "upsert_relation")

    access_worker = AccessWorker()
    access_request = access_worker.build_request(
        task=build_task(TaskType.PRIVILEGE_CONFIGURATION_VALIDATION),
        operation_id="op-1",
        metadata={
            "runtime_session": {"session_id": "sess-1", "status": "active"},
            "runtime_credential": {"credential_id": "cred-1", "status": "valid"},
            "host_reachability": {"reachable": True, "source_id": "host-0", "source_type": "Host", "via": "pivot"},
            "privilege_validation": {"validated": True, "required_level": "admin"},
        },
    )
    access_result = access_worker.execute_task(access_request)
    access_applied = applier.apply(
        access_result,
        state,
        kg_ref=ProtocolGraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
    )
    assert any(delta["patch"]["entity_type"] == "Session" for delta in access_applied.kg_state_deltas if delta["delta_type"] == "upsert_entity")
    assert any(delta["patch"]["entity_type"] == "Credential" for delta in access_applied.kg_state_deltas if delta["delta_type"] == "upsert_entity")
    assert any(delta["patch"]["entity_type"] == "PrivilegeState" for delta in access_applied.kg_state_deltas if delta["delta_type"] == "upsert_entity")
    assert any(delta["patch"]["relation_type"] == "CAN_REACH" for delta in access_applied.kg_state_deltas if delta["delta_type"] == "upsert_relation")
    assert any(delta["patch"]["relation_type"] == "HAS_PRIVILEGE_STATE" for delta in access_applied.kg_state_deltas if delta["delta_type"] == "upsert_relation")

    goal_worker = GoalWorker()
    goal_request = goal_worker.build_request(
        task=build_task(TaskType.GOAL_CONDITION_VALIDATION),
        operation_id="op-1",
        metadata={"goal_evaluation": {"satisfied": False, "missing_requirements": ["proof-of-access"]}},
    )
    goal_result = goal_worker.execute_task(goal_request)
    goal_applied = applier.apply(
        goal_result,
        state,
        kg_ref=ProtocolGraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
    )
    assert any(delta["patch"]["entity_type"] == "Goal" for delta in goal_applied.kg_state_deltas if delta["delta_type"] == "upsert_entity")
    assert any(delta["patch"]["entity_type"] == "Finding" for delta in goal_applied.kg_state_deltas if delta["delta_type"] == "upsert_entity")
    assert any(delta["patch"]["relation_type"] == "SUPPORTED_BY" for delta in goal_applied.kg_state_deltas if delta["delta_type"] == "upsert_relation")


def test_result_applier_blocked_open_session_keeps_task_retryable_and_creates_lease() -> None:
    applier = PhaseTwoResultApplier()
    state = build_state()
    state.execution.tasks["task-1"].status = TaskRuntimeStatus.RUNNING
    state.execution.tasks["task-1"].assigned_worker = "worker-1"
    state.workers["worker-1"] = WorkerRuntime(
        worker_id="worker-1",
        status=WorkerStatus.BUSY,
        current_task_id="task-1",
    )
    result = AgentTaskResult(
        request_id="request-blocked",
        agent_role=AgentRole.ACCESS_WORKER,
        operation_id="op-1",
        task_id="task-1",
        tg_node_id="task-1",
        status=AgentResultStatus.BLOCKED,
        summary="session required before task can proceed",
        runtime_requests=[
            RuntimeControlRequest(
                request_type=RuntimeControlType.OPEN_SESSION,
                source_task_id="task-1",
                lease_seconds=120,
                reuse_policy="shared",
                metadata={"bound_target": "host-1", "bound_identity": "alice"},
            )
        ],
        outcome_payload={"blocked_on": "session"},
    )

    applier.apply(result, state, kg_ref=ProtocolGraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"))

    session = next(iter(state.sessions.values()))
    lease = next(iter(state.session_leases.values()))
    assert state.execution.tasks["task-1"].status == TaskRuntimeStatus.PENDING
    assert state.execution.tasks["task-1"].assigned_worker is None
    assert state.workers["worker-1"].status == WorkerStatus.IDLE
    assert session.status == SessionStatus.ACTIVE
    assert lease.session_id == session.session_id


def test_result_applier_failure_cleans_session_lease_credential_and_route() -> None:
    applier = PhaseTwoResultApplier()
    state = build_state()
    task = state.execution.tasks["task-1"]
    task.status = TaskRuntimeStatus.RUNNING
    task.assigned_worker = "worker-1"
    task.metadata["session_id"] = "sess-1"
    state.workers["worker-1"] = WorkerRuntime(
        worker_id="worker-1",
        status=WorkerStatus.BUSY,
        current_task_id="task-1",
    )
    state.sessions["sess-1"] = SessionRuntime(
        session_id="sess-1",
        status=SessionStatus.ACTIVE,
        bound_target="host-1",
        metadata={"bound_task_ids": ["task-1"]},
    )
    state.session_leases["lease-1"] = SessionLeaseRuntime(
        lease_id="lease-1",
        session_id="sess-1",
        owner_task_id="task-1",
    )
    state.credentials["cred-1"] = CredentialRuntime(
        credential_id="cred-1",
        principal="alice",
        kind=CredentialKind.TOKEN,
        status=CredentialStatus.VALID,
        source_session_id="sess-1",
    )
    state.pivot_routes["route-1"] = PivotRouteRuntime(
        route_id="route-1",
        destination_host="host-1",
        source_host="host-0",
        via_host="jump-1",
        session_id="sess-1",
        status=PivotRouteStatus.ACTIVE,
    )
    state.locks["host:host-1"] = ResourceLock(
        lock_key="host:host-1",
        owner_type="task",
        owner_id="task-1",
        acquired_at=utc_now(),
    )
    result = AgentTaskResult(
        request_id="request-failed",
        agent_role=AgentRole.ACCESS_WORKER,
        operation_id="op-1",
        task_id="task-1",
        tg_node_id="task-1",
        status=AgentResultStatus.FAILED,
        summary="credential validation failed",
        error_message="credential validation failed",
        outcome_payload={
            "session_id": "sess-1",
            "credential_validation": {"credential_id": "cred-1", "status": "invalid", "target_id": "host-1"},
            "selected_route": {"route_id": "route-1", "destination_host": "host-1", "session_id": "sess-1"},
            "reachability": {"reachable": False, "via": "pivot", "route_id": "route-1", "source_id": "host-0"},
        },
    )

    applier.apply(result, state, kg_ref=ProtocolGraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"))

    assert state.execution.tasks["task-1"].status == TaskRuntimeStatus.FAILED
    assert state.workers["worker-1"].status == WorkerStatus.IDLE
    assert state.locks["host:host-1"].status.value == "released"
    assert state.sessions["sess-1"].status == SessionStatus.FAILED
    assert state.session_leases["lease-1"].metadata["release_reason"] == "credential validation failed"
    assert state.credentials["cred-1"].status == CredentialStatus.EXPIRED
    assert state.pivot_routes["route-1"].status == PivotRouteStatus.FAILED
