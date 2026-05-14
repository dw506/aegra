from __future__ import annotations

from datetime import timezone

import pytest

from src.app import api as app_api
from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.agents.agent_pipeline import PipelineCycleResult, PipelineStepResult
from src.core.agents.agent_protocol import AgentContext, AgentInput, AgentKind, AgentOutput
from src.core.agents.agent_protocol import utc_now as agent_utc_now

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


def _validation(*, accepted: bool, reason: str = "accepted") -> dict[str, object]:
    return {
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "reason": reason,
        "sanitized_payload": {},
    }


def _cycle(agent_kind: AgentKind, output: AgentOutput) -> PipelineCycleResult:
    now = agent_utc_now().astimezone(timezone.utc)
    step = PipelineStepResult(
        step_name=agent_kind.value,
        agent_name=f"fake_{agent_kind.value}",
        agent_kind=agent_kind,
        success=True,
        agent_input=AgentInput(context=AgentContext(operation_id="op-api")),
        agent_output=output,
        started_at=now,
        finished_at=now,
        duration_ms=0,
    )
    return PipelineCycleResult(
        cycle_name=agent_kind.value,
        operation_id="op-api",
        success=True,
        steps=[step],
        final_output=output,
    )


def _client(tmp_path):
    client_cls = _require_test_client()
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    orchestrator = AppOrchestrator(settings=settings)
    orchestrator.create_operation("op-api")
    planner_output = AgentOutput(logs=["planner llm strategy decision rejected: invalid candidate"])
    supervisor_output = AgentOutput(
        decisions=[
            {
                "decision_type": "supervisor_strategy",
                "payload": {
                    "supervisor_decision": {"strategy": "pause_for_review"},
                    "llm_decision_validation": _validation(accepted=True),
                },
            }
        ]
    )
    orchestrator.record_llm_decision_cycle("op-api", cycle_index=1, cycle=_cycle(AgentKind.PLANNER, planner_output))
    orchestrator.record_llm_decision_cycle("op-api", cycle_index=2, cycle=_cycle(AgentKind.SUPERVISOR, supervisor_output))
    state = orchestrator.get_operation_state("op-api")
    state.execution.metadata["control_cycle_history"] = [
        {"cycle_index": 1, "stopped": False},
        {"cycle_index": 2, "stopped": True},
    ]
    orchestrator.runtime_store.save_state(state)
    return client_cls(app_api.create_app(orchestrator=orchestrator, settings=settings))


def test_api_llm_decisions_default_query(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.get("/operations/op-api/llm-decisions")

    assert response.status_code == 200
    assert [item["agent_kind"] for item in response.json()] == ["planner", "supervisor"]


def test_api_llm_decisions_filters_agent_kind_and_accepted(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.get("/operations/op-api/llm-decisions?agent_kind=supervisor&accepted=true")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["agent_kind"] == "supervisor"
    assert payload[0]["accepted"] is True


def test_api_control_cycles_applies_limit(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.get("/operations/op-api/control-cycles?limit=1")

    assert response.status_code == 200
    assert response.json() == [{"cycle_index": 2, "stopped": True}]


def test_api_audit_report_returns_filtered_report(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.get("/operations/op-api/audit-report?limit=1&agent_kind=planner&accepted=false")

    assert response.status_code == 200
    payload = response.json()
    assert payload["operation_id"] == "op-api"
    assert payload["filters"] == {"limit": 1, "agent_kind": "planner", "accepted": False}
    assert [item["agent_kind"] for item in payload["llm_decision_history"]] == ["planner"]
    assert [item["cycle_index"] for item in payload["control_cycle_history"]] == [2]


def test_api_operation_not_found_returns_404(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.get("/operations/missing/llm-decisions")

    assert response.status_code == 404


def test_api_rejects_out_of_range_limit(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.get(f"/operations/op-api/llm-decisions?limit={app_api.LLM_DECISION_QUERY_LIMIT_MAX + 1}")

    assert response.status_code == 422
