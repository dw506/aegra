from __future__ import annotations

from types import SimpleNamespace

from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.agents.agent_pipeline import AgentPipeline
from src.core.agents.agent_protocol import AgentKind, GraphRef, GraphScope
from src.core.agents.critic import CriticAgent
from src.core.agents.packy_supervisor_advisor import PackySupervisorAdvisor
from src.core.agents.pipeline_builders import AgentPipelineAssemblyOptions, build_optional_agent_pipeline
from src.core.agents.scheduler_agent import SchedulerAgent
from src.core.agents.supervisor import SupervisorAgent, SupervisorContext, SupervisorDecision, SupervisorStrategy
from src.core.agents.task_builder import TaskBuilderAgent

from test_app_orchestrator import FakePlannerAgent, FakeWorkerAgent, build_runtime_state


class StaticSupervisorAdvisor:
    def __init__(self, decision: SupervisorDecision) -> None:
        self.decision = decision

    def advise(self, *, context: SupervisorContext) -> SupervisorDecision:
        del context
        return self.decision


def _graph_refs() -> list[GraphRef]:
    return [
        GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
        GraphRef(graph=GraphScope.AG, ref_id="ag-root", ref_type="graph"),
        GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph"),
    ]


def test_pipeline_builder_does_not_include_supervisor_by_default() -> None:
    pipeline = build_optional_agent_pipeline()

    assert pipeline.registry.list_by_kind(AgentKind.SUPERVISOR) == []


def test_pipeline_builder_can_include_supervisor_and_emit_strategy_advice() -> None:
    advisor = StaticSupervisorAdvisor(
        SupervisorDecision(
            strategy=SupervisorStrategy.PAUSE_FOR_REVIEW,
            rationale="budget risk needs review",
            confidence=0.8,
            requires_human_review=True,
            metadata={"reason": "budget_risk"},
        )
    )
    pipeline = build_optional_agent_pipeline(supervisor_llm_advisor=advisor)

    result = pipeline.run_supervisor_cycle(
        operation_id="op-supervisor",
        graph_refs=_graph_refs(),
        supervisor_payload={
            "runtime_summary": {"task_count": 1},
            "last_control_cycle": {"cycle_index": 1},
            "planner_summary": {"pending_candidate_count": 0},
            "critic_summary": {"finding_count": 0},
            "budget_summary": {"remaining": 10},
        },
    )

    assert result.success is True
    assert [step.agent_kind for step in result.steps] == [AgentKind.SUPERVISOR]
    payload = result.final_output.decisions[0]["payload"]
    assert payload["supervisor_decision"]["strategy"] == "pause_for_review"
    assert payload["llm_adopted"] is True
    assert payload["llm_decision_validation"]["accepted"] is True
    assert result.final_output.state_deltas == []
    assert result.final_output.replan_requests == []


def test_supervisor_rejects_forbidden_llm_strategy_and_falls_back() -> None:
    advisor = StaticSupervisorAdvisor(
        SupervisorDecision(
            strategy=SupervisorStrategy.PAUSE_FOR_REVIEW,
            rationale="unsafe direct patch",
            confidence=0.9,
            metadata={"tool_command": "run something"},
        )
    )
    agent = SupervisorAgent(llm_advisor=advisor)
    pipeline = AgentPipeline(agents=[agent])

    result = pipeline.run_supervisor_cycle(
        operation_id="op-supervisor-reject",
        graph_refs=_graph_refs(),
        supervisor_payload={"runtime_summary": {"replan_request_count": 1}},
    )

    payload = result.final_output.decisions[0]["payload"]
    assert payload["supervisor_decision"]["strategy"] == "request_replan"
    assert payload["llm_adopted"] is False
    assert payload["llm_decision_validation"]["accepted"] is False
    assert "forbidden field" in payload["llm_decision_validation"]["reason"]


def test_packy_supervisor_advisor_parses_fake_client_response() -> None:
    class FakeClient:
        def complete_chat(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(
                text=(
                    '{"strategy": "continue_planning", "rationale": "planner has candidates", '
                    '"confidence": 0.7, "requires_human_review": false, '
                    '"metadata": {"reason": "candidate_queue"}}'
                )
            )

    advisor = PackySupervisorAdvisor(client=FakeClient())  # type: ignore[arg-type]

    decision = advisor.advise(
        context=SupervisorContext(planner_summary={"pending_candidate_count": 2})
    )

    assert decision is not None
    assert decision.strategy == SupervisorStrategy.CONTINUE_PLANNING
    assert decision.metadata["reason"] == "candidate_queue"


def test_orchestrator_does_not_run_supervisor_in_main_cycle(tmp_path) -> None:
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    pipeline = AgentPipeline(
        agents=[
            FakePlannerAgent(),
            TaskBuilderAgent(),
            SchedulerAgent(),
            FakeWorkerAgent(),
            CriticAgent(),
            SupervisorAgent(),
        ]
    )
    orchestrator = AppOrchestrator(settings=settings, pipeline=pipeline)
    orchestrator.create_operation("op-main-loop")
    runtime_state = build_runtime_state()
    runtime_state.execution.metadata["runtime_policy"] = {"sensitive_task_types": []}
    orchestrator.runtime_store.save_state(runtime_state)

    result = orchestrator.run_operation_cycle(
        "op-main-loop",
        graph_refs=_graph_refs(),
        planner_payload={"goal_refs": [], "planning_context": {"top_k": 1, "max_depth": 1}},
    )

    assert result.planning is not None
    assert result.execution is not None
    assert result.feedback is not None
    assert all(step.agent_kind != AgentKind.SUPERVISOR for step in result.planning.steps)
    assert all(step.agent_kind != AgentKind.SUPERVISOR for step in result.execution.steps)
    assert all(step.agent_kind != AgentKind.SUPERVISOR for step in result.feedback.steps)
    assert result.runtime_state.execution.metadata.get("last_control_cycle", {}).get("cycle_index") == 1
