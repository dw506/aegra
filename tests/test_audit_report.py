from __future__ import annotations

from copy import deepcopy

from src.core.models.runtime import OperationRuntime, ReplanRequest, RuntimeState, RuntimeStatus
from src.core.runtime.audit_report import build_operation_audit_report
from src.core.runtime.observability import append_operation_log


def _state() -> RuntimeState:
    state = RuntimeState(operation_id="op-audit-report", execution=OperationRuntime(operation_id="op-audit-report"))
    state.operation_status = RuntimeStatus.PAUSED
    state.execution.status = RuntimeStatus.PAUSED
    state.execution.metadata["pause_reason"] = "operator review"
    state.execution.metadata["last_supervisor_control_strategy"] = {
        "cycle_index": 2,
        "strategy": "pause_for_review",
        "accepted": True,
        "reason": "operator review",
    }
    state.execution.metadata["control_cycle_history"] = [
        {"cycle_index": 1, "stopped": False},
        {
            "cycle_index": 2,
            "stopped": True,
            "supervisor_control_strategy": {
                "strategy": "pause_for_review",
                "accepted": True,
            },
        },
    ]
    state.request_replan(
        ReplanRequest(
            request_id="replan-1",
            reason="supervisor requested existing replan flow",
            metadata={"source": "supervisor", "cycle_index": 2},
        )
    )
    append_operation_log(state, event_type="operation_created", operation_id=state.operation_id)
    append_operation_log(state, event_type="operation_paused_for_review", reason="operator review")
    return state


def test_operation_audit_report_contains_required_sections() -> None:
    state = _state()

    report = build_operation_audit_report(state)

    assert report["operation_id"] == "op-audit-report"
    assert report["operation_status"] == "paused"
    assert report["execution_status"] == "paused"
    assert report["pause_reason"] == "operator review"
    assert report["latest_supervisor_control_strategy"]["strategy"] == "pause_for_review"
    assert [item["cycle_index"] for item in report["control_cycle_history"]] == [1, 2]
    assert report["replan_requests"][0]["request_id"] == "replan-1"
    assert report["operation_log"][-1]["event_type"] == "operation_paused_for_review"
    assert report["budget_summary"]["requires_human_review"] is False
    assert "correlations" in report["derived"]


def test_operation_audit_report_applies_limit_to_recent_sections() -> None:
    state = _state()

    report = build_operation_audit_report(state, limit=1)

    assert [item["cycle_index"] for item in report["control_cycle_history"]] == [2]
    assert [item["event_type"] for item in report["operation_log"]] == ["operation_paused_for_review"]
    assert report["filters"]["limit"] == 1


def test_operation_audit_report_sanitizes_sensitive_and_raw_llm_content() -> None:
    state = _state()
    state.execution.metadata["operation_log"].append(
        {
            "event_type": "llm_debug",
            "authorization": "Bearer secret-token",
            "long_context": "x" * 800,
        }
    )

    report = build_operation_audit_report(state)
    report_text = str(report)

    assert "Bearer secret-token" not in report_text
    assert "truncated" in report["operation_log"][-1]["long_context"]


def test_operation_audit_report_includes_budget_guard_summary() -> None:
    state = _state()
    state.budgets.operation_budget_used = 2
    state.budgets.operation_budget_max = 2
    state.budgets.token_budget_used = 10
    state.budgets.token_budget_max = 20

    report = build_operation_audit_report(state)

    assert report["budget_summary"]["operation_budget_used"] == 2
    assert report["budget_summary"]["operation_budget_max"] == 2
    assert report["budget_summary"]["guards"]["operation"] is True
    assert report["budget_summary"]["guards"]["token"] is False
    assert report["budget_summary"]["requires_human_review"] is True


def test_operation_audit_report_does_not_mutate_runtime_state() -> None:
    state = _state()
    before = deepcopy(state.model_dump(mode="json"))

    build_operation_audit_report(state, limit=1, agent_kind="planner", accepted=True)

    assert state.model_dump(mode="json") == before


def test_operation_audit_report_correlates_accepted_request_replan_strategy() -> None:
    state = _state()
    state.execution.metadata["control_cycle_history"] = [
        {
            "cycle_index": 2,
            "supervisor_control_strategy": {
                "cycle_index": 2,
                "strategy": "request_replan",
                "accepted": True,
                "reason": "use existing replan flow",
            },
        }
    ]
    state.execution.metadata["last_supervisor_control_strategy"] = {
        "cycle_index": 2,
        "strategy": "request_replan",
        "accepted": True,
        "reason": "use existing replan flow",
    }

    report = build_operation_audit_report(state)
    correlation = report["derived"]["correlations"]["accepted_request_replans"][0]

    assert correlation["strategy"]["strategy"] == "request_replan"
    assert correlation["replan_request"]["request_id"] == "replan-1"
    assert correlation["replan_request"]["metadata"]["source"] == "supervisor"


def test_operation_audit_report_correlates_accepted_pause_strategy() -> None:
    state = _state()
    state.execution.metadata["pause_cycle_index"] = 2

    report = build_operation_audit_report(state)
    correlation = report["derived"]["correlations"]["accepted_pauses"][0]

    assert correlation["strategy"]["strategy"] == "pause_for_review"
    assert correlation["pause_reason"] == "operator review"
    assert correlation["pause_cycle_index"] == 2


def test_operation_audit_report_correlates_budget_guard_strategy() -> None:
    state = _state()
    state.execution.metadata["control_cycle_history"] = [
        {
            "cycle_index": 3,
            "cycle_type": "control_guard",
            "supervisor_control_strategy": {
                "cycle_index": 3,
                "strategy": "budget_guard",
                "accepted": True,
                "reason": "budget guard triggered",
            },
        }
    ]
    state.execution.metadata["last_supervisor_control_strategy"] = {
        "cycle_index": 3,
        "strategy": "budget_guard",
        "accepted": True,
        "reason": "budget guard triggered",
    }
    state.budgets.risk_budget_used = 1
    state.budgets.risk_budget_max = 1

    report = build_operation_audit_report(state)
    correlation = report["derived"]["correlations"]["budget_guards"][0]

    assert correlation["strategy"]["strategy"] == "budget_guard"
    assert correlation["guards"]["risk"] is True
    assert correlation["requires_human_review"] is True


def test_operation_audit_report_correlates_deterministic_fallback_strategy() -> None:
    state = _state()
    state.execution.metadata["control_cycle_history"] = [
        {
            "cycle_index": 4,
            "supervisor_control_strategy": {
                "cycle_index": 4,
                "strategy": "deterministic_fallback",
                "accepted": False,
                "reason": "consecutive llm rejections threshold reached",
            },
        }
    ]
    state.execution.metadata["last_supervisor_control_strategy"] = {
        "cycle_index": 4,
        "strategy": "deterministic_fallback",
        "accepted": False,
        "reason": "consecutive llm rejections threshold reached",
    }

    report = build_operation_audit_report(state)
    correlation = report["derived"]["correlations"]["deterministic_fallbacks"][0]

    assert correlation["strategy"]["strategy"] == "deterministic_fallback"
