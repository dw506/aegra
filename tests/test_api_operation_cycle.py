from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.app import api as app_api
from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.agents.agent_pipeline import AgentPipeline
from src.core.agents.critic import CriticAgent
from src.core.agents.scheduler_agent import SchedulerAgent
from src.core.agents.task_builder import TaskBuilderAgent
from src.core.models.runtime import WorkerRuntime, WorkerStatus
from src.core.execution.mcp_client import MCPToolCallResult
from src.core.workers.llm_worker import LLMWorkerAgent
from src.core.workers.llm_worker_models import LLMWorkerDecision

from test_app_orchestrator import FakePlannerAgent, FakeSchedulerAdvisor, FakeWorkerAgent, build_graph_refs


class FakeAnyTaskWorkerAgent(FakeWorkerAgent):
    def supports_task(self, task_spec) -> bool:
        del task_spec
        return True


class FakeLLMWorkerAdvisor:
    def advise(self, **kwargs):
        return LLMWorkerDecision(
            action="call_mcp_tool",
            server_id="lab",
            tool_name="noop",
            arguments={},
            summary="fake llm worker executed",
        )


class FakeMCPClient:
    def list_tools(self):
        return {"available": True, "tools": [{"server_id": "lab", "name": "noop"}]}

    def is_available(self, server_id=None):
        return True

    def call_tool(self, **kwargs):
        return MCPToolCallResult(success=True, stdout="ok", metadata={})


try:
    from fastapi.testclient import TestClient
except Exception as exc:  # pragma: no cover - depends on optional HTTP deps
    TestClient = None
    _TESTCLIENT_IMPORT_ERROR = exc
else:  # pragma: no cover - depends on optional HTTP deps
    _TESTCLIENT_IMPORT_ERROR = None


def _require_test_client():
    if app_api.FastAPI is None:
        pytest.skip(app_api.FASTAPI_UNAVAILABLE_MESSAGE)
    if TestClient is None:
        pytest.skip(f"FastAPI TestClient unavailable: {_TESTCLIENT_IMPORT_ERROR}")
    return TestClient


def _build_client(tmp_path, *, runtime_policy: dict | None = None):
    client_cls = _require_test_client()
    settings = AppSettings(
        runtime_store_backend="file",
        runtime_store_dir=tmp_path / "runtime-store",
        runtime_policy=runtime_policy or {"authorized_hosts": ["127.0.0.1"]},
    )
    pipeline = AgentPipeline(
        agents=[
            FakePlannerAgent(),
            TaskBuilderAgent(),
            SchedulerAgent(advisor=FakeSchedulerAdvisor()),
            LLMWorkerAgent(advisor=FakeLLMWorkerAdvisor(), mcp_client=FakeMCPClient()),
            CriticAgent(),
        ]
    )
    orchestrator = AppOrchestrator(settings=settings, pipeline=pipeline)
    return client_cls(app_api.create_app(orchestrator=orchestrator, settings=settings)), orchestrator


def _bootstrap_operation(client, orchestrator: AppOrchestrator, *, operation_id: str = "op-api-cycle") -> None:
    response = client.post("/operations", json={"operation_id": operation_id, "metadata": {}})
    assert response.status_code == 200

    response = client.post(
        f"/operations/{operation_id}/targets",
        json={
            "targets": [
                {
                    "address": "127.0.0.1",
                    "hostname": "localhost",
                    "platform": "linux",
                    "tags": ["lab"],
                    "metadata": {"port": 8080},
                }
            ]
        },
    )
    assert response.status_code == 200

    response = client.post(f"/operations/{operation_id}/start")
    assert response.status_code == 200

    state = orchestrator.get_operation_state(operation_id)
    state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    orchestrator.runtime_store.save_state(state)


def _cycle_payload() -> dict:
    return {
        "graph_refs": [ref.model_dump(mode="json") for ref in build_graph_refs()],
        "planner_payload": {"goal_refs": [], "planning_context": {"top_k": 1, "max_depth": 1}},
        "worker_overrides": {"SERVICE_VALIDATION": "fake_worker"},
    }


def test_api_operation_cycle_runs_and_exposes_summary_and_audit(tmp_path) -> None:
    client, orchestrator = _build_client(tmp_path)
    _bootstrap_operation(client, orchestrator)

    response = client.post("/operations/op-api-cycle/cycle", json=_cycle_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["operation_id"] == "op-api-cycle"
    assert payload["cycle_index"] == 1
    assert payload["planning"]["success"] is True
    assert payload["execution"]["success"] is True
    assert payload["execution"]["cycle_name"] == "stage_dispatch"
    assert "task_graph" not in payload["runtime_state"]["execution"]["metadata"]
    assert payload["feedback"]["success"] is True

    summary_response = client.get("/operations/op-api-cycle/summary")
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert summary["operation_id"] == "op-api-cycle"
    assert summary["last_cycle_phase"] == "cycle_completed"
    assert summary["audit_event_count"] > 0

    audit_response = client.get("/operations/op-api-cycle/audit-report")
    assert audit_response.status_code == 200
    audit = audit_response.json()
    assert audit["operation_id"] == "op-api-cycle"
    assert audit["control_cycle_history"]


def test_api_operation_run_executes_bounded_cycles(tmp_path) -> None:
    client, orchestrator = _build_client(tmp_path)
    _bootstrap_operation(client, orchestrator, operation_id="op-api-run")
    payload = _cycle_payload() | {"max_cycles": 2, "stop_when_quiescent": False}

    response = client.post("/operations/op-api-run/run", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["operation"]["operation_id"] == "op-api-run"
    assert len(body["cycles"]) == 2


def test_api_operation_stop_marks_operation_cancelled(tmp_path) -> None:
    client, orchestrator = _build_client(tmp_path)
    _bootstrap_operation(client, orchestrator, operation_id="op-api-stop")

    response = client.post("/operations/op-api-stop/stop", json={"reason": "test_stop"})

    assert response.status_code == 200
    body = response.json()
    assert body["operation_status"] == "cancelled"
    assert body["execution"]["metadata"]["stop_request"]["reason"] == "test_stop"

    cycle_response = client.post("/operations/op-api-stop/cycle", json=_cycle_payload())
    assert cycle_response.status_code == 409


def test_api_operation_cycle_requires_imported_targets(tmp_path) -> None:
    client, orchestrator = _build_client(tmp_path)
    response = client.post("/operations", json={"operation_id": "op-no-target", "metadata": {}})
    assert response.status_code == 200
    response = client.post("/operations/op-no-target/start")
    assert response.status_code == 200
    state = orchestrator.get_operation_state("op-no-target")
    state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    orchestrator.runtime_store.save_state(state)

    response = client.post("/operations/op-no-target/cycle", json=_cycle_payload())

    assert response.status_code == 409
    assert "no imported authorized targets" in response.json()["detail"]


def test_api_operation_cycle_rejects_policy_blocked_target(tmp_path) -> None:
    client, orchestrator = _build_client(
        tmp_path,
        runtime_policy={"authorized_hosts": ["127.0.0.1"], "blocked_hosts": ["127.0.0.1"]},
    )
    _bootstrap_operation(client, orchestrator, operation_id="op-blocked")

    response = client.post("/operations/op-blocked/cycle", json=_cycle_payload())

    assert response.status_code == 409
    assert "blocked by runtime policy" in response.json()["detail"]
