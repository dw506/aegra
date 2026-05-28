from __future__ import annotations

import sys

from src.core.agents.agent_protocol import AgentContext, AgentInput, GraphRef as ProtocolGraphRef, GraphScope
from src.core.execution import ExecutionExecutor, LocalShellAdapter
from src.core.models.ag import GraphRef
from src.core.models.tg import TaskType
from src.core.workers.access_validation_worker import AccessValidationWorker
from src.core.workers.base import WorkerTaskSpec
from src.core.workers.services.access_validation_service import AccessValidationRequest, AccessValidationService


def _request(metadata: dict | None = None) -> AccessValidationRequest:
    return AccessValidationRequest(
        operation_id="op-1",
        task_id="task-1",
        task_type=TaskType.IDENTITY_CONTEXT_CONFIRMATION.value,
        task_label="Access validation",
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host", label="host-1")],
        metadata=metadata or {"require_session": False, "host_reachability": {"reachable": True}},
    )


def test_access_service_without_executor_keeps_existing_behavior() -> None:
    service = AccessValidationService()

    result = service.validate(_request())

    assert result.status == "succeeded"
    assert result.raw_payload["validated"] is True
    assert "tool_execution" not in result.raw_payload


def test_access_service_with_local_shell_executor_records_tool_result() -> None:
    service = AccessValidationService(executor=ExecutionExecutor([LocalShellAdapter()]))
    request = _request(
        {
            "require_session": True,
            "execution_adapter": "local_shell",
            "probe_command": [sys.executable, "-c", "print('ok')"],
        }
    )

    result = service.validate(request)

    assert result.status == "succeeded"
    assert result.raw_payload["tool_execution"]["adapter"] == "local_shell"
    assert result.raw_payload["tool_execution"]["tool"] == "session_probe"
    assert result.raw_payload["tool_execution"]["success"] is True
    assert result.raw_payload["session_probe"]["usable"] is True
    assert result.evidence[0]["metadata"]["tool_execution"]["stdout"].strip() == "ok"


def test_access_validation_worker_preserves_tool_result_in_agent_output() -> None:
    service = AccessValidationService(executor=ExecutionExecutor([LocalShellAdapter()]))
    worker = AccessValidationWorker(service=service)
    task_spec = WorkerTaskSpec(
        task_id="task-1",
        task_type=TaskType.IDENTITY_CONTEXT_CONFIRMATION.value,
        target_refs=[ProtocolGraphRef(graph=GraphScope.KG, ref_id="host-1", ref_type="Host")],
    )
    agent_input = AgentInput(
        graph_refs=task_spec.target_refs,
        task_ref="task-1",
        context=AgentContext(operation_id="op-1"),
        raw_payload={
            "task_type": task_spec.task_type,
            "task_label": "Access validation",
            "metadata": {
                "require_session": True,
                "execution_adapter": "local_shell",
                "probe_command": [sys.executable, "-c", "print('ok')"],
            },
        },
    )

    output = worker.execute_task(task_spec, agent_input)

    payload = output.outcomes[0]["payload"]
    assert payload["tool_execution"]["adapter"] == "local_shell"
    assert payload["tool_execution"]["success"] is True
    assert output.evidence[0]["metadata"]["tool_execution"]["tool"] == "session_probe"
