from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app import api as app_api
from src.app.orchestrator import AppOrchestrator, TargetHost
from src.app.settings import AppSettings
from src.core.models.runtime import RuntimeStatus
from src.core.planning.models import PlannerDecision

try:
    from fastapi.testclient import TestClient
except Exception as exc:  # pragma: no cover - depends on optional HTTP deps
    TestClient = None
    _TESTCLIENT_IMPORT_ERROR = exc
else:  # pragma: no cover - depends on optional HTTP deps
    _TESTCLIENT_IMPORT_ERROR = None


SUCCESS_PROGRESS = {
    "conditions": {
        "dmz_service_discovered": {"satisfied": True, "evidence_ids": ["ev-service"]},
        "goal_check_recorded": {"satisfied": True, "evidence_ids": ["ev-goal"]},
    },
    "all_required_satisfied": True,
    "recommended_planner_action": "stop_success",
}


class StopSuccessPlanner:
    def run(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any] | None = None,
        recent_stage_results: list[dict[str, Any]] | None = None,
    ) -> PlannerDecision:
        del policy_context, recent_stage_results
        return PlannerDecision(
            operation_id=str(graph_context["operation_id"]),
            cycle_index=int(graph_context["cycle_index"]),
            decision="stop_success",
            selected_agent=None,
            selected_stage=None,
            objective=goal,
            risk_level="low",
            max_steps=1,
            reasoning_summary="required success conditions are satisfied",
            confidence=1.0,
        )


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        runtime_store_backend="memory",
        runtime_store_dir=tmp_path / "runtime",
        lab_profile={"profile_id": "full-vulhub-multihost-pentest"},
    )


def test_operation_run_summary_contract_uses_success_condition_progress(tmp_path: Path) -> None:
    orchestrator = AppOrchestrator(settings=_settings(tmp_path))
    state = orchestrator.create_operation("op-contract")
    state.operation_status = RuntimeStatus.COMPLETED
    state.execution.status = RuntimeStatus.COMPLETED
    state.execution.metadata["success_condition_progress"] = SUCCESS_PROGRESS
    orchestrator.runtime_store.save_state(state)

    summary = orchestrator.get_operation_run_summary("op-contract")

    assert summary.model_dump(mode="json") == {
        "operation_id": "op-contract",
        "status": "success",
        "stop_reason": "success_conditions_satisfied",
        "success": True,
        "success_condition_progress": SUCCESS_PROGRESS,
        "evidence_ids": ["ev-goal", "ev-service"],
        "findings_url": "/operations/op-contract/findings",
        "evidence_url": "/operations/op-contract/evidence",
        "graph_url": "/operations/op-contract/graph",
        "audit_url": "/operations/op-contract/audit-report",
    }


def test_full_pentest_activation_uses_lab_profile_not_mcp_toolset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AEGRA_MCP_TOOLSET", raising=False)
    orchestrator = AppOrchestrator(settings=_settings(tmp_path))

    state = orchestrator.create_operation("op-lab-profile")

    assert state.execution.metadata["lab_activation"] == {
        "full_pentest_active": True,
        "profile_id": "full-vulhub-multihost-pentest",
        "source": "lab_profile.profile_id",
    }


def test_run_endpoint_returns_operation_run_summary_contract(tmp_path: Path) -> None:
    if app_api.FastAPI is None:
        pytest.skip(app_api.FASTAPI_UNAVAILABLE_MESSAGE)
    if TestClient is None:
        pytest.skip(f"FastAPI TestClient unavailable: {_TESTCLIENT_IMPORT_ERROR}")
    settings = _settings(tmp_path)
    orchestrator = AppOrchestrator(settings=settings)
    orchestrator.mission_planner = StopSuccessPlanner()  # type: ignore[assignment]
    client = TestClient(app_api.create_app(orchestrator=orchestrator, settings=settings))

    assert client.post("/operations", json={"operation_id": "op-api-contract", "metadata": {}}).status_code == 200
    assert client.post(
        "/operations/op-api-contract/targets",
        json={"targets": [{"address": "10.20.0.0/24", "kind": "cidr"}]},
    ).status_code == 200
    assert client.post("/operations/op-api-contract/start").status_code == 200
    state = orchestrator.get_operation_state("op-api-contract")
    state.execution.metadata["success_condition_progress"] = SUCCESS_PROGRESS
    orchestrator.runtime_store.save_state(state)

    response = client.post("/operations/op-api-contract/run", json={"max_cycles": 1})

    assert response.status_code == 200
    body = response.json()
    for payload in (body, body["result"]):
        assert payload["operation_id"] == "op-api-contract"
        assert payload["status"] == "success"
        assert payload["stop_reason"] == "success_conditions_satisfied"
        assert payload["success"] is True
        assert payload["success_condition_progress"] == SUCCESS_PROGRESS
        assert payload["evidence_ids"] == ["ev-goal", "ev-service"]
        assert payload["findings_url"] == "/operations/op-api-contract/findings"
        assert payload["evidence_url"] == "/operations/op-api-contract/evidence"
        assert payload["graph_url"] == "/operations/op-api-contract/graph"
        assert payload["audit_url"] == "/operations/op-api-contract/audit-report"
