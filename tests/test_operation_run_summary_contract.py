from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app import api as app_api
from src.app.orchestrator import AppOrchestrator, TargetHost
from src.app.settings import AppSettings
from src.core.models.runtime import OutcomeCacheEntry, RuntimeStatus
from src.core.planning.models import PlannerDecision
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


def test_operation_run_summary_merges_stage_and_tool_evidence_when_progress_empty(tmp_path: Path) -> None:
    orchestrator = AppOrchestrator(settings=_settings(tmp_path))
    state = orchestrator.create_operation("op-evidence-merge")
    state.record_outcome(
        OutcomeCacheEntry(
            outcome_id="outcome-stage",
            task_id="stage-recon",
            outcome_type="RECON_STAGE",
            summary="stage wrote evidence",
            payload_ref="runtime://outcome/stage-recon",
            metadata={
                "outcome_payload": {
                    "stage_result": {
                        "evidence_refs": ["ev-stage-ref"],
                        "evidence": [{"evidence_id": "ev-stage-record", "payload_ref": "raw-stage.json"}],
                        "tool_trace": [{"raw_output_ref": "tool-output.json", "evidence_refs": ["ev-tool-ref"]}],
                    }
                }
            },
        )
    )
    state.execution.metadata["evidence_artifacts"] = [{"evidence_id": "ev-kg"}]
    orchestrator.runtime_store.save_state(state)

    summary = orchestrator.get_operation_run_summary("op-evidence-merge")

    assert summary.evidence_ids == [
        "ev-kg",
        "ev-stage-record",
        "ev-stage-ref",
        "ev-tool-ref",
        "raw-stage.json",
        "runtime://outcome/stage-recon",
        "tool-output.json",
    ]


def test_planner_dead_end_constraint_overrides_repeated_exploit_validation(tmp_path: Path) -> None:
    orchestrator = AppOrchestrator(settings=_settings(tmp_path))
    state = orchestrator.create_operation("op-dead-end")
    state.execution.metadata["validation_dead_ends"] = [
        {"target": "10.20.0.10:8080", "reason": "no_supported_profile"}
    ]
    decision = PlannerDecision(
        operation_id="op-dead-end",
        cycle_index=5,
        decision="dispatch_agent",
        selected_agent="exploit_validation_agent",
        selected_stage="EXPLOIT_STAGE",
        objective="Validate candidate on 10.20.0.10:8080",
        required_context={"target_url": "http://10.20.0.10:8080/"},
        risk_level="medium",
        max_steps=1,
        confidence=0.8,
    )

    constrained = orchestrator._apply_planner_dead_end_constraints(
        decision=decision,
        state=state,
        cycle_index=5,
    )

    assert constrained.selected_agent == "recon_agent"
    assert constrained.selected_stage == "RECON_STAGE"
    assert constrained.metadata["dead_end_override"] is True


def test_full_pentest_activation_uses_lab_profile_not_mcp_toolset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AEGRA_MCP_TOOLSET", raising=False)
    orchestrator = AppOrchestrator(settings=_settings(tmp_path))

    state = orchestrator.create_operation("op-lab-profile")

    assert state.execution.metadata["lab_activation"] == {
        "full_pentest_active": True,
        "profile_id": "full-vulhub-multihost-pentest",
        "source": "lab_profile.profile_id",
    }


def test_success_condition_progress_is_derived_from_configured_stage_signals(tmp_path: Path) -> None:
    orchestrator = AppOrchestrator(
        settings=AppSettings(
            runtime_store_backend="memory",
            runtime_store_dir=tmp_path / "runtime",
            lab_profile={
                "profile_id": "full-vulhub-multihost-pentest",
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

    orchestrator._update_success_condition_progress(state=state, stage_result=recon_result)
    orchestrator._update_success_condition_progress(state=state, stage_result=vuln_result)

    progress = state.execution.metadata["success_condition_progress"]
    assert progress["conditions"]["dmz_service_discovered"]["evidence_ids"] == ["ev-recon"]
    assert progress["conditions"]["vulnerability_candidate_recorded"]["evidence_ids"] == ["ev-vuln"]
    assert progress["all_required_satisfied"] is True


def test_success_condition_progress_does_not_match_internal_service_from_dmz_recon(tmp_path: Path) -> None:
    orchestrator = AppOrchestrator(
        settings=AppSettings(
            runtime_store_backend="memory",
            runtime_store_dir=tmp_path / "runtime",
            lab_profile={
                "profile_id": "full-vulhub-multihost-pentest",
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

    orchestrator._update_success_condition_progress(state=state, stage_result=recon_result)

    progress = state.execution.metadata["success_condition_progress"]
    assert progress["conditions"]["dmz_service_discovered"]["evidence_ids"] == ["raw-nmap.json"]
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
