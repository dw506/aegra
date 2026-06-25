from __future__ import annotations


import pytest

from src.app import api as app_api
from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings

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


def _client(tmp_path):
    client_cls = _require_test_client()
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    orchestrator = AppOrchestrator(settings=settings)
    orchestrator.create_operation("op-api")
    state = orchestrator.get_operation_state("op-api")
    state.execution.metadata["control_cycle_history"] = [
        {"cycle_index": 1, "stopped": False},
        {"cycle_index": 2, "stopped": True},
    ]
    orchestrator.runtime_store.save_state(state)
    return client_cls(app_api.create_app(orchestrator=orchestrator, settings=settings))


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
    assert [item["cycle_index"] for item in payload["control_cycle_history"]] == [2]


def test_api_operation_not_found_returns_404(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.get("/operations/missing/control-cycles")

    assert response.status_code == 404


def test_api_rejects_out_of_range_limit(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.get(f"/operations/op-api/control-cycles?limit={app_api.CONTROL_CYCLE_QUERY_LIMIT_MAX + 1}")

    assert response.status_code == 422
