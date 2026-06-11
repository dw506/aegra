from __future__ import annotations

from datetime import timezone

from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.agents.pipeline_results import PipelineCycleResult, PipelineStepResult
from src.core.agents.agent_protocol import AgentContext, AgentInput, AgentKind, AgentOutput
from src.core.agents.agent_protocol import utc_now as agent_utc_now


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
        agent_input=AgentInput(context=AgentContext(operation_id="op-history")),
        agent_output=output,
        started_at=now,
        finished_at=now,
        duration_ms=0,
    )
    return PipelineCycleResult(
        cycle_name=agent_kind.value,
        operation_id="op-history",
        success=True,
        steps=[step],
        final_output=output,
    )


def _orchestrator(tmp_path, **settings_kwargs: object) -> AppOrchestrator:
    settings = AppSettings(
        runtime_store_backend="file",
        runtime_store_dir=tmp_path / "runtime-store",
        **settings_kwargs,
    )
    return AppOrchestrator(settings=settings)


def test_llm_decision_history_is_empty_by_default(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path)
    state = orchestrator.create_operation("op-history")

    assert state.execution.metadata["llm_decision_history"] == []
    assert orchestrator.get_llm_decision_history("op-history", limit=10) == []


def test_llm_decision_history_records_planner_decision(tmp_path) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        llm_api_key="secret-key",
        llm_model="gpt-5.4",
        enable_planner_llm_advisor=True,
    )
    orchestrator.create_operation("op-history")
    output = AgentOutput(
        decisions=[
            {
                "decision_type": "plan_selection",
                "payload": {
                    "planning_candidate": {
                        "metadata": {
                            "llm_decision": {"decision_type": "planner_strategy_decision"},
                            "llm_decision_validation": _validation(accepted=True),
                        }
                    }
                },
            }
        ]
    )

    orchestrator.record_llm_decision_cycle("op-history", cycle_index=1, cycle=_cycle(AgentKind.PLANNER, output))
    history = orchestrator.get_llm_decision_history("op-history")

    assert history[-1]["agent_kind"] == "planner"
    assert history[-1]["decision_type"] == "planner_strategy_decision"
    assert history[-1]["accepted"] is True
    assert history[-1]["enabled"] is True
    assert history[-1]["configured"] is True
    assert history[-1]["model"] == "gpt-5.4"
    assert "secret-key" not in str(history[-1])


def test_llm_decision_history_keeps_distinct_targets_in_same_cycle(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path, enable_planner_rank_llm_advisor=True, llm_api_key="secret-key")
    orchestrator.create_operation("op-history")
    output = AgentOutput(
        decisions=[
            {
                "decision_type": "plan_selection",
                "payload": {
                    "planning_candidate": {
                        "metadata": {
                            "llm_decision": {
                                "decision_id": "llm-1",
                                "decision_type": "planner_strategy_decision",
                                "target_id": "goal-a",
                                "target_kind": "planner_goal",
                            },
                            "llm_decision_validation": _validation(accepted=True),
                        }
                    }
                },
            },
            {
                "decision_type": "plan_selection",
                "payload": {
                    "planning_candidate": {
                        "metadata": {
                            "llm_decision": {
                                "decision_id": "llm-2",
                                "decision_type": "planner_strategy_decision",
                                "target_id": "goal-b",
                                "target_kind": "planner_goal",
                            },
                            "llm_decision_validation": _validation(accepted=True),
                        }
                    }
                },
            },
        ]
    )

    orchestrator.record_llm_decision_cycle("op-history", cycle_index=1, cycle=_cycle(AgentKind.PLANNER, output))
    history = orchestrator.get_llm_decision_history("op-history")

    assert [item["target_id"] for item in history] == ["goal-a", "goal-b"]
    assert [item["decision_id"] for item in history] == ["llm-1", "llm-2"]


def test_llm_decision_history_records_critic_decision(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path, enable_critic_llm_advisor=False)
    orchestrator.create_operation("op-history")
    output = AgentOutput(
        decisions=[
            {
                "decision_type": "cancel_suggestion",
                "payload": {
                    "recommendation": {
                        "metadata": {
                            "llm_decision": {"decision_type": "critic_finding_review"},
                            "llm_decision_validation": _validation(accepted=True),
                        }
                    }
                },
            }
        ]
    )

    orchestrator.record_llm_decision_cycle("op-history", cycle_index=2, cycle=_cycle(AgentKind.CRITIC, output))
    history = orchestrator.get_llm_decision_history("op-history")

    assert history[-1]["cycle_index"] == 2
    assert history[-1]["agent_kind"] == "critic"
    assert history[-1]["advisor_type"] == "injected"
    assert history[-1]["decision_type"] == "critic_finding_review"
    assert history[-1]["accepted"] is True


def test_llm_decision_history_records_supervisor_decision(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_operation("op-history")
    output = AgentOutput(
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

    orchestrator.record_llm_decision_cycle("op-history", cycle_index=3, cycle=_cycle(AgentKind.SUPERVISOR, output))
    history = orchestrator.get_llm_decision_history("op-history")

    assert history[-1]["agent_kind"] == "supervisor"
    assert history[-1]["decision_type"] == "supervisor_strategy"
    assert history[-1]["accepted"] is True


def test_llm_decision_history_records_rejected_reason_from_logs(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_operation("op-history")
    output = AgentOutput(logs=["planner llm strategy decision rejected: planner strategy selected unknown candidate_id"])

    orchestrator.record_llm_decision_cycle("op-history", cycle_index=4, cycle=_cycle(AgentKind.PLANNER, output))
    history = orchestrator.get_llm_decision_history("op-history")

    assert history[-1]["agent_kind"] == "planner"
    assert history[-1]["accepted"] is False
    assert history[-1]["rejected_reason"] == "planner strategy selected unknown candidate_id"


def test_llm_decision_history_query_returns_recent_n(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_operation("op-history")
    for index in range(1, 4):
        output = AgentOutput(logs=[f"planner llm strategy decision rejected: reason-{index}"])
        orchestrator.record_llm_decision_cycle("op-history", cycle_index=index, cycle=_cycle(AgentKind.PLANNER, output))

    history = orchestrator.get_llm_decision_history("op-history", limit=2)

    assert [item["cycle_index"] for item in history] == [2, 3]


def test_llm_decision_history_query_filters_by_agent_kind_and_accepted(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_operation("op-history")
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

    orchestrator.record_llm_decision_cycle("op-history", cycle_index=1, cycle=_cycle(AgentKind.PLANNER, planner_output))
    orchestrator.record_llm_decision_cycle("op-history", cycle_index=2, cycle=_cycle(AgentKind.SUPERVISOR, supervisor_output))

    supervisor_history = orchestrator.get_llm_decision_history("op-history", agent_kind="supervisor", accepted=True)
    rejected_history = orchestrator.get_llm_decision_history("op-history", accepted=False)

    assert len(supervisor_history) == 1
    assert supervisor_history[0]["agent_kind"] == "supervisor"
    assert supervisor_history[0]["accepted"] is True
    assert len(rejected_history) == 1
    assert rejected_history[0]["agent_kind"] == "planner"
    assert rejected_history[0]["accepted"] is False


def test_operation_audit_report_query_reuses_runtime_report(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_operation("op-history")
    output = AgentOutput(logs=["planner llm strategy decision rejected: invalid candidate"])
    orchestrator.record_llm_decision_cycle("op-history", cycle_index=1, cycle=_cycle(AgentKind.PLANNER, output))
    state = orchestrator.get_operation_state("op-history")
    state.execution.metadata["control_cycle_history"] = [
        {"cycle_index": 1, "stopped": False},
        {"cycle_index": 2, "stopped": True},
    ]
    orchestrator.runtime_store.save_state(state)

    report = orchestrator.get_operation_audit_report("op-history", limit=1, agent_kind="planner", accepted=False)
    control_history = orchestrator.get_control_cycle_history("op-history", limit=1)

    assert report["operation_id"] == "op-history"
    assert report["filters"] == {"limit": 1, "agent_kind": "planner", "accepted": False}
    assert [item["cycle_index"] for item in report["llm_decision_history"]] == [1]
    assert [item["cycle_index"] for item in report["control_cycle_history"]] == [2]
    assert control_history == [{"cycle_index": 2, "stopped": True}]


def test_orchestrator_history_queries_return_sanitized_views(tmp_path) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator.create_operation("op-history")
    state = orchestrator.get_operation_state("op-history")
    state.execution.metadata["llm_decision_history"] = [
        {
            "cycle_index": 1,
            "agent_kind": "supervisor",
            "advisor_type": "injected",
            "enabled": True,
            "configured": True,
            "decision_type": "supervisor_strategy",
            "accepted": False,
            "prompt": "full prompt should not leak",
            "raw_response": "raw response should not leak",
            "api_key": "secret-key",
        }
    ]
    state.execution.metadata["control_cycle_history"] = [
        {
            "cycle_index": 1,
            "supervisor_control_strategy": {
                "strategy": "deterministic_fallback",
                "accepted": False,
                "reason": "authorization: Bearer secret-token",
                "raw_response": "raw control response should not leak",
            },
        }
    ]
    orchestrator.runtime_store.save_state(state)

    llm_history = orchestrator.get_llm_decision_history("op-history")
    control_history = orchestrator.get_control_cycle_history("op-history")
    combined = f"{llm_history} {control_history}"

    assert "secret-key" not in combined
    assert "full prompt should not leak" not in combined
    assert "raw response should not leak" not in combined
    assert "Bearer secret-token" not in combined
    assert "raw control response should not leak" not in combined
    assert llm_history[0]["prompt"] == "[REDACTED]"
    assert llm_history[0]["raw_response"] == "[REDACTED]"
    assert control_history[0]["supervisor_control_strategy"]["raw_response"] == "[REDACTED]"
