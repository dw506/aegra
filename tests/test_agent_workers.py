from __future__ import annotations

import sys

from src.core.models.ag import GraphRef
from src.core.models.events import (
    AgentExecutionContext,
    AgentResultStatus,
    AgentRole,
    AgentTaskIntent,
    AgentTaskRequest,
    ProjectionRequestKind,
    RuntimeControlType,
)
from src.core.models.runtime import OperationRuntime, RuntimeState, SessionRuntime, SessionStatus, TaskRuntime
from src.core.models.tg import TaskNode, TaskType
from src.core.workers.access_worker import AccessWorker
from src.core.workers.goal_worker import GoalWorker
from src.core.workers.recon_worker import ReconWorker


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


def test_build_request_links_tg_task_and_runtime_task() -> None:
    worker = ReconWorker()
    task = build_task(TaskType.SERVICE_VALIDATION)
    runtime_task = TaskRuntime(
        task_id="task-1",
        tg_node_id="tg-node-1",
        attempt_count=1,
        max_attempts=3,
        assigned_worker="worker-1",
        checkpoint_ref="cp-1",
        resource_keys={"host:host-1", "service:svc-1"},
        metadata={"session_id": "sess-1"},
    )

    request = worker.build_request(
        task=task,
        operation_id="op-1",
        task_runtime=runtime_task,
    )

    assert request.agent_role == AgentRole.RECON_WORKER
    assert request.intent == AgentTaskIntent.COLLECT_EVIDENCE
    assert request.context.task_id == "task-1"
    assert request.context.tg_node_id == "tg-node-1"
    assert request.context.assigned_worker_id == "worker-1"
    assert request.context.session_id == "sess-1"
    assert request.context.resource_keys == {"host:host-1", "service:svc-1"}


def test_recon_worker_returns_structured_write_and_projection_intents() -> None:
    worker = ReconWorker()
    request = worker.build_request(
        task=build_task(TaskType.REACHABILITY_VALIDATION),
        operation_id="op-1",
        metadata={
            "require_checkpoint": True,
            "tool_command": [
                sys.executable,
                "-c",
                (
                    "import json; "
                    "print(json.dumps({'summary':'reachability confirmed','reachable':True,"
                    "'confidence':0.88,'entities':[{'id':'host-1','type':'Host'}],"
                    "'relations':[],'runtime_hints':{'reachable':True}}))"
                ),
            ],
        },
    )

    result = worker.execute_task(request)

    assert result.status == AgentResultStatus.SUCCEEDED
    assert len(result.observations) == 1
    assert len(result.evidence) == 1
    assert result.fact_write_requests
    assert {item.kind.value for item in result.fact_write_requests} >= {"entity_upsert", "relation_upsert"}
    assert result.projection_requests[0].kind == ProjectionRequestKind.REFRESH_LOCAL_FRONTIER
    assert result.runtime_requests[0].request_type == RuntimeControlType.CONSUME_BUDGET
    assert len(result.checkpoint_hints) == 1


def test_access_worker_blocks_when_session_is_required() -> None:
    worker = AccessWorker()
    request = worker.build_request(
        task=build_task(TaskType.IDENTITY_CONTEXT_CONFIRMATION),
        operation_id="op-1",
        metadata={"require_session": True, "lease_seconds": 120},
    )

    result = worker.execute_task(request)

    assert result.status == AgentResultStatus.BLOCKED
    assert result.runtime_requests[0].request_type == RuntimeControlType.OPEN_SESSION
    assert result.runtime_requests[0].lease_seconds == 120


def test_access_worker_succeeds_with_session_and_may_emit_replan_signal() -> None:
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
            "runtime_snapshot": runtime_state,
            "session_probe": {"session_id": "sess-1", "status": "active"},
            "privilege_gap_detected": True,
        },
    )

    result = worker.execute_task(request)

    assert result.status == AgentResultStatus.SUCCEEDED
    assert result.outcome_payload["validated"] is True
    assert result.outcome_payload["session_id"] == "sess-1"
    assert {item.kind.value for item in result.fact_write_requests} >= {"entity_upsert", "relation_upsert"}
    assert len(result.critic_signals) == 1
    assert len(result.replan_hints) == 1
    assert any(item.request_type == RuntimeControlType.REQUEST_REPLAN for item in result.runtime_requests)


def test_goal_worker_requests_local_replan_when_goal_unsatisfied() -> None:
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
                "confidence": 0.82,
            }
        },
    )

    result = worker.execute_task(request)

    assert result.status == AgentResultStatus.NEEDS_REPLAN
    assert {item.kind.value for item in result.fact_write_requests} >= {"entity_upsert", "relation_upsert"}
    assert result.replan_hints[0].scope.value == "local"
    assert result.critic_signals[0].kind == "goal_unsatisfied"
    assert any(item.request_type == RuntimeControlType.REQUEST_REPLAN for item in result.runtime_requests)


def test_request_validation_rejects_wrong_agent_role() -> None:
    worker = GoalWorker()
    request = AgentTaskRequest(
        agent_role=AgentRole.RECON_WORKER,
        intent=AgentTaskIntent.VALIDATE_GOAL,
        context=AgentExecutionContext(
            operation_id="op-1",
            task_id="task-1",
            tg_node_id="task-1",
            task_type=TaskType.GOAL_CONDITION_VALIDATION,
        ),
        task_label="Goal validation",
    )

    result = worker.execute_task(request)

    assert result.status == AgentResultStatus.FAILED
    assert "does not match" in (result.error_message or "")
