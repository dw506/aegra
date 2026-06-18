from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app import api as app_api
from src.app.orchestrator import AppOrchestrator, TargetHost
from src.app.settings import AppSettings
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.models.runtime import RuntimeStatus
from src.core.planning.models import PlannerDecision
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.stage.models import StageResult, ToolTrace

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


def test_success_condition_progress_is_derived_from_configured_stage_signals(tmp_path: Path) -> None:
    orchestrator = AppOrchestrator(
        settings=AppSettings(
            runtime_store_backend="memory",
            runtime_store_dir=tmp_path / "runtime",
            lab_profile={
                "profile_id": "inline-progress-test",
                "success_conditions": {
                    "require_all": [
                        "dmz_service_discovered",
                        "vulnerability_candidate_recorded",
                    ]
                },
            },
        )
    )
    state = orchestrator.create_operation("op-progress")
    recon_result = StageResult(
        operation_id="op-progress",
        stage_task_id="stage-recon",
        stage_type="RECON_STAGE",
        agent_name="recon_agent",
        status="succeeded",
        summary="service discovery completed",
        findings=[{"type": "service_discovery", "summary": "one service discovered"}],
        evidence_refs=["ev-recon"],
    )
    vuln_result = StageResult(
        operation_id="op-progress",
        stage_task_id="stage-vuln",
        stage_type="VULN_ANALYSIS_STAGE",
        agent_name="vuln_analysis_agent",
        status="succeeded",
        summary="candidate vulnerability analysis completed",
        findings=[{"kind": "candidate_finding", "summary": "candidate recorded"}],
        evidence_refs=["ev-vuln"],
        tool_trace=[ToolTrace(tool_name="web_fingerprint", success=True, summary="candidate")],
    )

    kg = KnowledgeGraph()
    ag = AttackGraph()
    applier = PhaseTwoResultApplier()
    applier.apply_stage_result(recon_result, state, kg, ag)
    orchestrator._update_success_condition_progress(state=state, kg=kg, ag=ag)
    applier.apply_stage_result(vuln_result, state, kg, ag)
    orchestrator._update_success_condition_progress(state=state, kg=kg, ag=ag)

    progress = state.execution.metadata["success_condition_progress"]
    assert progress["conditions"]["dmz_service_discovered"]["satisfied"] is True
    assert progress["conditions"]["vulnerability_candidate_recorded"]["satisfied"] is True
    assert progress["all_required_satisfied"] is True


def test_success_condition_progress_does_not_match_internal_service_from_dmz_recon(tmp_path: Path) -> None:
    orchestrator = AppOrchestrator(
        settings=AppSettings(
            runtime_store_backend="memory",
            runtime_store_dir=tmp_path / "runtime",
            lab_profile={
                "profile_id": "inline-progress-specific-test",
                "success_conditions": {
                    "require_all": [
                        "dmz_service_discovered",
                        "internal_service_discovered_after_authorized_route",
                    ]
                },
            },
        )
    )
    state = orchestrator.create_operation("op-progress-specific")
    recon_result = StageResult(
        operation_id="op-progress-specific",
        stage_task_id="stage-recon",
        stage_type="RECON_STAGE",
        agent_name="recon_agent",
        status="needs_replan",
        summary="service discovery needs replan",
        findings=[{"type": "service_discovery", "summary": "one service discovered"}],
        tool_trace=[ToolTrace(tool_name="nmap_scan", success=True, raw_output_ref="raw-nmap.json")],
    )

    kg = KnowledgeGraph()
    ag = AttackGraph()
    PhaseTwoResultApplier().apply_stage_result(recon_result, state, kg, ag)
    orchestrator._update_success_condition_progress(state=state, kg=kg, ag=ag)

    progress = state.execution.metadata["success_condition_progress"]
    assert progress["conditions"]["dmz_service_discovered"]["satisfied"] is True
    assert progress["conditions"]["internal_service_discovered_after_authorized_route"]["satisfied"] is False
    assert progress["all_required_satisfied"] is False


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


def test_unified_operation_endpoint_creates_imports_starts_and_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if app_api.FastAPI is None:
        pytest.skip(app_api.FASTAPI_UNAVAILABLE_MESSAGE)
    if TestClient is None:
        pytest.skip(f"FastAPI TestClient unavailable: {_TESTCLIENT_IMPORT_ERROR}")
    settings = _settings(tmp_path)
    orchestrator = AppOrchestrator(settings=settings)
    run_context: dict[str, Any] = {}

    def fake_run_until_quiescent(operation_id: str, **kwargs: Any) -> list[Any]:
        state = orchestrator.get_operation_state(operation_id)
        run_context["operation_status_before_run"] = state.operation_status
        run_context["target_count_before_run"] = state.execution.metadata.get("target_count")
        run_context["max_cycles"] = kwargs["max_cycles"]
        state.operation_status = RuntimeStatus.COMPLETED
        state.execution.status = RuntimeStatus.COMPLETED
        state.execution.metadata["success_condition_progress"] = SUCCESS_PROGRESS
        orchestrator.runtime_store.save_state(state)
        return []

    monkeypatch.setattr(orchestrator, "run_until_quiescent", fake_run_until_quiescent)
    client = TestClient(app_api.create_app(orchestrator=orchestrator, settings=settings))

    response = client.post(
        "/operations",
        json={
            "operation_id": "op-unified",
            "metadata": {
                "operation_input": {
                    "target": "10.20.0.0/24",
                    "profile_id": "full-vulhub-multihost-pentest",
                    "mode": "authorized_blackbox_lab",
                    "goal": "perform authorized multi-host assessment under supplied policy",
                }
            },
            "targets": [{"address": "10.20.0.0/24", "kind": "cidr"}],
            "max_cycles": 1,
        },
    )

    assert response.status_code == 200
    assert run_context == {
        "operation_status_before_run": RuntimeStatus.READY,
        "target_count_before_run": 1,
        "max_cycles": 1,
    }
    body = response.json()
    assert body["result"]["operation_id"] == "op-unified"
    assert body["result"]["status"] == "success"
    assert body["result"]["success"] is True
    assert body["operation"]["target_count"] == 1
