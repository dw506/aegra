from __future__ import annotations

from datetime import timezone

from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.agents.agent_pipeline import PipelineCycleResult, PipelineStepResult
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
