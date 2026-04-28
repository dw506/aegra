from __future__ import annotations

import sys

from src.core.models.ag import GraphRef
from src.core.models.runtime import (
    CredentialKind,
    CredentialRuntime,
    CredentialStatus,
    OperationRuntime,
    PivotRouteRuntime,
    PivotRouteStatus,
    RuntimeState,
    SessionLeaseRuntime,
    SessionRuntime,
    SessionStatus,
)
from src.core.models.tg import TaskNode, TaskType
from src.core.workers.access_worker import AccessWorker
from src.core.workers.goal_worker import GoalWorker
from src.core.workers.probe_adapters import CustomProbeAdapter, NmapAdapter
from src.core.workers.recon_worker import ReconWorker
from src.core.workers.tool_runner import ToolExecutionResult


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


def test_recon_worker_runs_real_tool_command_and_parses_json() -> None:
    worker = ReconWorker()
    request = worker.build_request(
        task=build_task(TaskType.SERVICE_VALIDATION),
        operation_id="op-1",
        metadata={
            "tool_command": [
                sys.executable,
                "-c",
                (
                    "import json; "
                    "print(json.dumps({'summary':'service probe ok','reachable':True,"
                    "'confidence':0.91,'service':{'banner':'ssh','port':22}}))"
                ),
            ],
        },
    )

    result = worker.execute_task(request)

    assert result.status.value == "succeeded"
    assert result.outcome_payload["tool"]["category"] == "success"
    assert result.outcome_payload["parsed"]["service"]["banner"] == "ssh"
    assert {item.kind.value for item in result.fact_write_requests} >= {"entity_upsert", "relation_upsert"}


def test_recon_worker_blocks_when_no_real_probe_tool_is_available() -> None:
    worker = ReconWorker()
    request = worker.build_request(
        task=build_task(TaskType.SERVICE_VALIDATION),
        operation_id="op-1",
        metadata={"probe_adapter": "custom"},
    )

    result = worker.execute_task(request)

    assert result.status.value == "blocked"
    assert result.outcome_payload["observed"] is False
    assert result.outcome_payload["tool"]["category"] == "command_not_found"


def test_nmap_adapter_parses_text_output_into_unified_result() -> None:
    adapter = NmapAdapter()
    execution_result = ToolExecutionResult(
        command=["nmap", "-sV", "10.0.0.5"],
        success=True,
        category="success",
        exit_code=0,
        stdout=(
            "Nmap scan report for 10.0.0.5\n"
            "Host is up (0.0010s latency).\n"
            "PORT   STATE SERVICE VERSION\n"
            "22/tcp open ssh OpenSSH 8.9\n"
        ),
    )

    parsed = adapter.parse_output(
        execution_result=execution_result,
        target_hint="10.0.0.5",
        mode="service_validation",
        metadata={},
    )

    assert parsed.success is True
    assert parsed.reachable is True
    assert parsed.service["port"] == 22
    assert parsed.service["banner"] == "OpenSSH 8.9"
    assert parsed.entities[0]["type"] == "Host"
    assert parsed.entities[1]["type"] == "Service"
    assert parsed.relations[0]["type"] == "HOSTS"


def test_custom_probe_adapter_parses_json_output_into_unified_result() -> None:
    adapter = CustomProbeAdapter()
    execution_result = ToolExecutionResult(
        command=[sys.executable, "-c", "print('ok')"],
        success=True,
        category="success",
        exit_code=0,
        stdout=(
            '{"summary":"custom probe ok","reachable":true,"confidence":0.93,'
            '"entities":[{"id":"host-1","type":"Host"},{"id":"svc-22","type":"Service","port":22}],'
            '"relations":[{"type":"HOSTS","source":"host-1","target":"svc-22"}],'
            '"evidence":{"kind":"custom-json"},"runtime_hints":{"reachable":true},'
            '"service":{"service_id":"svc-22","port":22,"banner":"ssh"}}'
        ),
    )

    parsed = adapter.parse_output(
        execution_result=execution_result,
        target_hint="host-1",
        mode="service_validation",
        metadata={},
    )

    assert parsed.success is True
    assert parsed.confidence == 0.93
    assert parsed.entities[1]["port"] == 22
    assert parsed.relations[0]["target"] == "svc-22"
    assert parsed.evidence["kind"] == "custom-json"
    assert parsed.runtime_hints["reachable"] is True


def test_custom_probe_adapter_marks_timeout_as_blocked() -> None:
    adapter = CustomProbeAdapter()
    execution_result = ToolExecutionResult(
        command=["missing-probe"],
        success=False,
        category="timeout",
        stdout="",
        stderr="",
        timed_out=True,
        error_message="command timed out after 5s",
    )

    parsed = adapter.parse_output(
        execution_result=execution_result,
        target_hint="host-1",
        mode="service_validation",
        metadata={},
    )

    assert parsed.blocked is True
    assert parsed.blocked_reason == "timeout"
    assert parsed.failure_reason == "command timed out after 5s"


def test_recon_worker_marks_nonzero_with_structured_output_as_failed_partial_success() -> None:
    worker = ReconWorker()
    request = worker.build_request(
        task=build_task(TaskType.SERVICE_VALIDATION),
        operation_id="op-1",
        metadata={
            "tool_command": [
                sys.executable,
                "-c",
                (
                    "import json,sys; "
                    "print(json.dumps({'summary':'partial probe','reachable':True,"
                    "'entities':[{'id':'host-1','type':'Host'},{'id':'svc-22','type':'Service','port':22}],"
                    "'relations':[{'type':'HOSTS','source':'host-1','target':'svc-22'}],"
                    "'service':{'service_id':'svc-22','port':22}})); "
                    "sys.exit(2)"
                ),
            ],
        },
    )

    result = worker.execute_task(request)

    assert result.status.value == "failed"
    assert result.outcome_payload["parsed"]["partial_success"] is True
    assert result.outcome_payload["parsed"]["failure_reason"] is not None
    assert result.fact_write_requests


def test_access_worker_uses_runtime_session_and_credential_views() -> None:
    worker = AccessWorker()
    runtime_state = RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))
    runtime_state.add_session(
        SessionRuntime(
            session_id="sess-1",
            status=SessionStatus.ACTIVE,
            bound_target="host-1",
            bound_identity="alice",
            metadata={"reuse_policy": "shared"},
        )
    )
    runtime_state.add_credential(
        CredentialRuntime(
            credential_id="cred-1",
            principal="alice",
            kind=CredentialKind.PASSWORD,
            status=CredentialStatus.VALID,
        )
    )
    request = worker.build_request(
        task=build_task(TaskType.PRIVILEGE_CONFIGURATION_VALIDATION),
        operation_id="op-1",
        metadata={
            "require_session": True,
            "require_credential": True,
            "runtime_snapshot": runtime_state,
            "session_probe": {"session_id": "sess-1", "status": "active"},
            "credential_validation": {"credential_id": "cred-1", "status": "valid"},
            "host_reachability": {"reachable": True},
            "privilege_validation": {"validated": False, "required_level": "admin"},
        },
    )

    result = worker.execute_task(request)

    assert result.status.value == "succeeded"
    assert result.outcome_payload["credential_status"] == "valid"
    assert result.outcome_payload["privilege_validation"]["validated"] is False
    assert result.outcome_payload["session_id"] == "sess-1"
    assert {item.kind.value for item in result.fact_write_requests} >= {"entity_upsert", "relation_upsert"}
    assert len(result.critic_signals) == 1


def test_access_worker_requests_new_session_when_runtime_snapshot_has_none() -> None:
    worker = AccessWorker()
    runtime_state = RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))
    request = worker.build_request(
        task=build_task(TaskType.IDENTITY_CONTEXT_CONFIRMATION),
        operation_id="op-1",
        metadata={
            "require_session": True,
            "runtime_snapshot": runtime_state,
            "host_reachability": {"reachable": True},
        },
    )

    result = worker.execute_task(request)

    assert result.status.value == "blocked"
    assert result.outcome_payload["blocked_on"] == "session"
    assert result.runtime_requests[0].request_type.value == "open_session"


def test_access_worker_blocks_when_session_probe_command_is_unavailable() -> None:
    worker = AccessWorker()
    request = worker.build_request(
        task=build_task(TaskType.IDENTITY_CONTEXT_CONFIRMATION),
        operation_id="op-1",
        metadata={
            "require_session": True,
            "session_probe_command": ["definitely-missing-session-probe"],
            "host_reachability": {"reachable": True},
        },
    )

    result = worker.execute_task(request)

    assert result.status.value == "blocked"
    assert result.outcome_payload["blocked_on"] == "session_probe"
    assert "session_probe" in (result.error_message or "")


def test_access_worker_fails_when_credential_validation_is_invalid() -> None:
    worker = AccessWorker()
    runtime_state = RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))
    runtime_state.add_credential(
        CredentialRuntime(
            credential_id="cred-1",
            principal="alice",
            kind=CredentialKind.PASSWORD,
            status=CredentialStatus.UNKNOWN,
        )
    )
    request = worker.build_request(
        task=build_task(TaskType.PRIVILEGE_CONFIGURATION_VALIDATION),
        operation_id="op-1",
        metadata={
            "require_session": False,
            "require_credential": True,
            "runtime_snapshot": runtime_state,
            "credential_validation": {"credential_id": "cred-1", "status": "invalid", "reason": "auth_failed"},
            "host_reachability": {"reachable": True},
        },
    )

    result = worker.execute_task(request)

    assert result.status.value == "failed"
    assert result.outcome_payload["credential_status"] == "invalid"


def test_access_worker_blocks_when_credential_validator_command_is_unavailable() -> None:
    worker = AccessWorker()
    request = worker.build_request(
        task=build_task(TaskType.PRIVILEGE_CONFIGURATION_VALIDATION),
        operation_id="op-1",
        metadata={
            "require_session": False,
            "require_credential": True,
            "credential_validator_command": ["definitely-missing-credential-validator"],
            "credential_validation": {"credential_id": "cred-1"},
            "host_reachability": {"reachable": True},
        },
    )

    result = worker.execute_task(request)

    assert result.status.value == "blocked"
    assert result.outcome_payload["blocked_on"] == "credential_validator"


def test_access_worker_selects_pivot_route_from_runtime_snapshot() -> None:
    worker = AccessWorker()
    runtime_state = RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))
    runtime_state.add_session(
        SessionRuntime(
            session_id="sess-1",
            status=SessionStatus.ACTIVE,
            bound_target="host-1",
            metadata={"reuse_policy": "shared"},
        )
    )
    runtime_state.add_pivot_route(
        PivotRouteRuntime(
            route_id="route-1",
            destination_host="host-1",
            source_host="host-0",
            via_host="pivot-1",
            session_id="sess-1",
            status=PivotRouteStatus.ACTIVE,
            protocol="ssh",
        )
    )
    request = worker.build_request(
        task=build_task(TaskType.IDENTITY_CONTEXT_CONFIRMATION),
        operation_id="op-1",
        metadata={
            "require_session": False,
            "runtime_snapshot": runtime_state,
            "prefer_pivot_route": True,
            "host_reachability": {"reachable": True, "via": "pivot", "source_id": "host-0"},
        },
    )

    result = worker.execute_task(request)

    assert result.status.value == "succeeded"
    assert result.outcome_payload["selected_route"]["route_id"] == "route-1"
    assert result.outcome_payload["reachability"]["via"] == "pivot"
    assert any(
        item.relation_type == "CAN_REACH" and item.attributes.get("route_id") == "route-1"
        for item in result.fact_write_requests
    )


def test_access_worker_privilege_gap_emits_replan_runtime_request() -> None:
    worker = AccessWorker()
    runtime_state = RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))
    runtime_state.add_session(
        SessionRuntime(
            session_id="sess-1",
            status=SessionStatus.ACTIVE,
            bound_target="host-1",
            metadata={"reuse_policy": "shared"},
        )
    )
    request = worker.build_request(
        task=build_task(TaskType.PRIVILEGE_CONFIGURATION_VALIDATION),
        operation_id="op-1",
        metadata={
            "require_session": True,
            "runtime_snapshot": runtime_state,
            "session_probe": {"session_id": "sess-1", "status": "active"},
            "host_reachability": {"reachable": True},
            "privilege_validation": {"validated": False, "required_level": "admin"},
        },
    )

    result = worker.execute_task(request)

    assert result.status.value == "succeeded"
    assert len(result.critic_signals) == 1
    assert len(result.replan_hints) == 1
    assert any(item.request_type.value == "request_replan" for item in result.runtime_requests)


def test_goal_worker_prefers_structured_goal_evaluation() -> None:
    worker = GoalWorker()
    request = worker.build_request(
        task=build_task(TaskType.GOAL_CONDITION_VALIDATION),
        operation_id="op-1",
        metadata={
            "goal_validator_output": {
                "satisfied": False,
                "missing_requirements": ["proof-of-access"],
                "validated_ref_ids": ["host-1"],
                "supporting_evidence": [{"payload_ref": "runtime://validators/goal/task-1"}],
                "confidence": 0.87,
            }
        },
    )

    result = worker.execute_task(request)

    assert result.status.value == "needs_replan"
    assert result.outcome_payload["goal_evaluation"]["missing_requirements"] == ["proof-of-access"]
    assert result.outcome_payload["goal_evaluation"]["supporting_evidence"][0]["payload_ref"] == "runtime://validators/goal/task-1"
    assert any(item.request_type.value == "request_replan" for item in result.runtime_requests)
    assert len(result.replan_hints) == 1
    assert {item.kind.value for item in result.fact_write_requests} >= {"entity_upsert", "relation_upsert"}


def test_goal_worker_succeeds_with_validator_output() -> None:
    worker = GoalWorker()
    request = worker.build_request(
        task=build_task(TaskType.GOAL_CONDITION_VALIDATION),
        operation_id="op-1",
        metadata={
            "goal_validator_output": {
                "satisfied": True,
                "missing_requirements": [],
                "validated_ref_ids": ["host-1"],
                "supporting_evidence": [{"payload_ref": "runtime://validators/goal/success"}],
                "confidence": 0.95,
            }
        },
    )

    result = worker.execute_task(request)

    assert result.status.value == "succeeded"
    assert result.outcome_payload["goal_satisfied"] is True
    assert result.outcome_payload["goal_evaluation"]["validated_ref_ids"] == ["host-1"]
    assert result.checkpoint_hints


def test_goal_worker_blocks_when_validator_command_is_unavailable() -> None:
    worker = GoalWorker()
    request = worker.build_request(
        task=build_task(TaskType.GOAL_CONDITION_VALIDATION),
        operation_id="op-1",
        metadata={"goal_validator_command": ["definitely-missing-goal-validator"]},
    )

    result = worker.execute_task(request)

    assert result.status.value == "blocked"
    assert result.outcome_payload["goal_evaluation"]["blocked"] is True


def test_goal_worker_uses_command_validator_output() -> None:
    worker = GoalWorker()
    request = worker.build_request(
        task=build_task(TaskType.GOAL_CONDITION_VALIDATION),
        operation_id="op-1",
        metadata={
            "goal_validator_command": [
                sys.executable,
                "-c",
                (
                    "import json; "
                    "print(json.dumps({'satisfied': False,"
                    "'missing_requirements':['proof-of-access'],"
                    "'validated_ref_ids':['host-1'],"
                    "'supporting_evidence':[{'payload_ref':'runtime://validators/goal/cmd'}],"
                    "'confidence':0.88}))"
                ),
            ]
        },
    )

    result = worker.execute_task(request)

    assert result.status.value == "needs_replan"
    assert result.outcome_payload["goal_evaluation"]["supporting_evidence"][0]["payload_ref"] == "runtime://validators/goal/cmd"


def test_runtime_state_tracks_credentials_leases_and_pivot_routes() -> None:
    state = RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))

    credential = CredentialRuntime(
        credential_id="cred-1",
        principal="operator",
        kind=CredentialKind.PASSWORD,
        status=CredentialStatus.VALID,
    )
    session = SessionRuntime(session_id="sess-1", status=SessionStatus.ACTIVE, bound_target="host-1")
    lease = SessionLeaseRuntime(lease_id="lease-1", session_id="sess-1", owner_task_id="task-1")
    route = PivotRouteRuntime(
        route_id="route-1",
        destination_host="host-2",
        via_host="host-1",
        session_id="sess-1",
        status=PivotRouteStatus.ACTIVE,
    )

    state.add_credential(credential)
    state.add_session(session)
    state.add_session_lease(lease)
    state.add_pivot_route(route)

    assert state.credentials["cred-1"].is_usable() is True
    assert state.session_leases["lease-1"].is_active() is True
    assert state.pivot_routes["route-1"].is_usable() is True
