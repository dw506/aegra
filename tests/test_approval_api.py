from __future__ import annotations

import pytest

from src.app import api as app_api
from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntime, TaskRuntimeStatus

try:
    from fastapi.testclient import TestClient
except Exception as exc:  # pragma: no cover
    TestClient = None
    _TESTCLIENT_IMPORT_ERROR = exc
else:  # pragma: no cover
    _TESTCLIENT_IMPORT_ERROR = None


def _client(tmp_path):
    if app_api.FastAPI is None:
        pytest.skip(app_api.FASTAPI_UNAVAILABLE_MESSAGE)
    if TestClient is None:
        pytest.skip(f"FastAPI TestClient unavailable: {_TESTCLIENT_IMPORT_ERROR}")
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    orchestrator = AppOrchestrator(settings=settings)
    state = RuntimeState(operation_id="op-approval", execution=OperationRuntime(operation_id="op-approval"))
    state.register_task(
        TaskRuntime(
            task_id="task-1",
            tg_node_id="task-1",
            status=TaskRuntimeStatus.WAITING_APPROVAL,
            metadata={
                "policy_decision": {
                    "decision": "requires_approval",
                    "gate": "risk",
                    "reason": "active_exploit requires approval",
                    "task_id": "task-1",
                    "approval_id": "task:task-1:approved",
                    "metadata": {},
                }
            },
        )
    )
    orchestrator.runtime_store.create_operation("op-approval", initial_state=state)
    return TestClient(app_api.create_app(orchestrator=orchestrator, settings=settings)), orchestrator


def test_approval_api_lists_and_applies_approvals(tmp_path) -> None:
    client, orchestrator = _client(tmp_path)

    listed = client.get("/operations/op-approval/approvals")
    assert listed.status_code == 200
    assert listed.json()[0]["approval_id"] == "task:task-1:approved"
    assert listed.json()[0]["status"] == "pending"

    approved = client.post(
        "/operations/op-approval/approve",
        json={
            "approval_id": "task:task-1:approved",
            "task_id": "task-1",
            "decision": "approve",
            "reason": "reviewed",
        },
    )

    assert approved.status_code == 200
    state = orchestrator.get_operation_state("op-approval")
    assert state.budgets.approval_cache["task:task-1:approved"] is True
    assert state.execution.tasks["task-1"].status == TaskRuntimeStatus.PENDING
    assert any(entry["event_type"] == "approval_decision" for entry in state.execution.metadata["audit_log"])
