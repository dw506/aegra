"""End-to-end test for the success-condition verification loop.

Proves the whole flow through the public orchestrator surface: stage evidence
-> _update_success_condition_progress -> eligible_for_stop -> PlannerAgent
stop_success -> loop terminates -> get_operation_run_summary reports success.

The planner stub only stops when the authoritative eligible_for_stop gate is
true, so a green test means the gate (not the LLM) drives termination.
"""

from __future__ import annotations

from typing import Any

from src.app.orchestrator import AppOrchestrator, TargetHost
from src.app.settings import AppSettings
from src.core.planning.models import PlannerDecision
from src.core.stage.models import StageExecutionRequest, StageResult
from src.core.stage.registry import StageAgentRegistry


def _agent(agent_name: str, stage_type: str, **result_kwargs: Any):
    class _StubAgent:
        def __init__(self) -> None:
            self.agent_name = agent_name
            self.stage_type = stage_type

        def run(self, request: StageExecutionRequest) -> StageResult:
            return StageResult(
                operation_id=request.operation_id,
                stage_task_id=f"stage-{request.operation_id}-{request.cycle_index}-{agent_name}",
                stage_type=stage_type,
                agent_name=agent_name,
                status="succeeded",
                **result_kwargs,
            )

    return _StubAgent()


_CHAIN = [
    ("dmz_service_discovered", "recon_agent", "RECON_STAGE"),
    ("vulnerability_candidate_recorded", "vuln_analysis_agent", "VULN_ANALYSIS_STAGE"),
    ("goal_check_recorded", "goal_agent", "GOAL_STAGE"),
]

_DB_READ_CHAIN = [
    ("database_file_read", "access_pivot_agent", "ACCESS_PIVOT_STAGE"),
    ("goal_check_recorded", "goal_agent", "GOAL_STAGE"),
]


class _GateAwarePlanner:
    def __init__(self) -> None:
        self.decisions: list[str] = []

    def run(self, *, goal: str, graph_context: dict[str, Any], **_: Any) -> PlannerDecision:
        progress = graph_context.get("success_condition_progress") or {}
        operation_id = str(graph_context["operation_id"])
        cycle_index = int(graph_context["cycle_index"])
        if progress.get("eligible_for_stop"):
            self.decisions.append("stop_success")
            return PlannerDecision(
                operation_id=operation_id,
                cycle_index=cycle_index,
                decision="stop_success",
                objective=goal,
                risk_level="low",
                max_steps=1,
                stop_condition="contract_satisfied",
                confidence=1.0,
            )
        missing = set(progress.get("missing") or [c for c, _, _ in _CHAIN])
        for condition, agent, stage in _CHAIN:
            if condition in missing:
                self.decisions.append(f"dispatch:{agent}")
                return PlannerDecision(
                    operation_id=operation_id,
                    cycle_index=cycle_index,
                    decision="dispatch_agent",
                    selected_agent=agent,
                    selected_stage=stage,
                    objective=goal,
                    risk_level="low",
                    max_steps=1,
                    confidence=0.9,
                )
        raise AssertionError("no missing condition but gate not eligible")


class _ProfileDrivenPlanner:
    def __init__(self, chain: list[tuple[str, str, str]]) -> None:
        self.chain = list(chain)
        self.decisions: list[str] = []

    def run(self, *, goal: str, graph_context: dict[str, Any], **_: Any) -> PlannerDecision:
        progress = graph_context.get("success_condition_progress") or {}
        operation_id = str(graph_context["operation_id"])
        cycle_index = int(graph_context["cycle_index"])
        if progress.get("eligible_for_stop"):
            self.decisions.append("stop_success")
            return PlannerDecision(
                operation_id=operation_id,
                cycle_index=cycle_index,
                decision="stop_success",
                objective=goal,
                risk_level="low",
                max_steps=1,
                stop_condition="contract_satisfied",
                confidence=1.0,
            )
        missing = set(progress.get("missing") or [condition for condition, _, _ in self.chain])
        for condition, agent, stage in self.chain:
            if condition in missing:
                self.decisions.append(f"dispatch:{agent}")
                return PlannerDecision(
                    operation_id=operation_id,
                    cycle_index=cycle_index,
                    decision="dispatch_agent",
                    selected_agent=agent,
                    selected_stage=stage,
                    objective=goal,
                    risk_level="low",
                    max_steps=1,
                    confidence=0.9,
                )
        raise AssertionError("no missing condition but gate not eligible")


def test_success_contract_loop_terminates_only_when_eligible_for_stop(tmp_path) -> None:
    settings = AppSettings(
        runtime_store_backend="memory",
        runtime_store_dir=tmp_path / "rt",
        runtime_policy={"authorized_hosts": ["10.20.0.20"]},
        lab_profile={
            "profile_id": "loop-e2e",
            "success_conditions": {
                "require_all": [
                    "dmz_service_discovered",
                    "vulnerability_candidate_recorded",
                    "goal_check_recorded",
                ]
            },
        },
    )
    orch = AppOrchestrator(settings=settings)
    planner = _GateAwarePlanner()
    orch.mission_planner = planner  # type: ignore[assignment]
    orch.stage_registry = StageAgentRegistry(  # type: ignore[list-item]
        [
            _agent(
                "recon_agent",
                "RECON_STAGE",
                summary="entry-zone service discovery completed",
                findings=[{"type": "service_discovery", "summary": "http service discovered"}],
                evidence_refs=["evidence::recon-service"],
            ),
            _agent(
                "vuln_analysis_agent",
                "VULN_ANALYSIS_STAGE",
                summary="vulnerability candidate recorded",
                findings=[{"kind": "candidate_finding", "summary": "candidate"}],
                evidence_refs=["evidence::vuln-candidate"],
            ),
            _agent(
                "goal_agent",
                "GOAL_STAGE",
                summary="goal proof validated",
                findings=[{"kind": "GoalCheck", "goal_satisfied": True}],
                evidence_refs=["evidence::goal-proof"],
                runtime_hints={"goal_satisfied": True, "goal_evidence_refs": ["evidence::goal-proof"]},
            ),
        ]
    )

    orch.create_operation("op-loop-e2e")
    orch.import_targets("op-loop-e2e", [TargetHost(address="10.20.0.20")])
    orch.start_operation("op-loop-e2e")

    results = orch.run_until_quiescent(
        "op-loop-e2e",
        graph_refs=[],
        planner_payload={"mission_goal": "complete the authorized success contract"},
        max_cycles=8,
    )

    state = orch.get_operation_state("op-loop-e2e")
    progress = state.execution.metadata["success_condition_progress"]
    summary = orch.get_operation_run_summary("op-loop-e2e", cycle_results=results)

    # The gate, not the planner, drives termination: dispatch all three, then stop.
    assert planner.decisions == [
        "dispatch:recon_agent",
        "dispatch:vuln_analysis_agent",
        "dispatch:goal_agent",
        "stop_success",
    ]
    assert progress["all_required_satisfied"] is True
    assert progress["eligible_for_stop"] is True
    assert progress["conditions"]["dmz_service_discovered"]["evidence_ids"] == ["evidence::recon-service"]
    assert state.operation_status.value == "completed"
    assert summary.success is True
    assert summary.status == "success"
    assert summary.stop_reason == "success_conditions_satisfied"


def test_success_contract_loop_does_not_stop_while_conditions_missing(tmp_path) -> None:
    """If a required condition is never satisfied, the gate stays closed and the
    planner never reaches stop_success within the cycle budget."""

    settings = AppSettings(
        runtime_store_backend="memory",
        runtime_store_dir=tmp_path / "rt",
        runtime_policy={"authorized_hosts": ["10.20.0.20"]},
        lab_profile={
            "profile_id": "loop-e2e-incomplete",
            "success_conditions": {
                "require_all": [
                    "dmz_service_discovered",
                    "vulnerability_candidate_recorded",
                    "goal_check_recorded",
                ]
            },
        },
    )
    orch = AppOrchestrator(settings=settings)
    planner = _GateAwarePlanner()
    orch.mission_planner = planner  # type: ignore[assignment]
    # Only recon is available: vuln/goal conditions can never be satisfied.
    orch.stage_registry = StageAgentRegistry(  # type: ignore[list-item]
        [
            _agent(
                "recon_agent",
                "RECON_STAGE",
                summary="entry-zone service discovery completed",
                findings=[{"type": "service_discovery", "summary": "http service discovered"}],
                evidence_refs=["evidence::recon-service"],
            )
        ]
    )

    orch.create_operation("op-loop-incomplete")
    orch.import_targets("op-loop-incomplete", [TargetHost(address="10.20.0.20")])
    orch.start_operation("op-loop-incomplete")

    orch.run_until_quiescent(
        "op-loop-incomplete",
        graph_refs=[],
        planner_payload={"mission_goal": "incomplete contract"},
        max_cycles=4,
    )

    state = orch.get_operation_state("op-loop-incomplete")
    progress = state.execution.metadata["success_condition_progress"]
    assert progress["all_required_satisfied"] is False
    assert progress["eligible_for_stop"] is False
    assert "vulnerability_candidate_recorded" in progress["missing"]
    assert "stop_success" not in planner.decisions
    assert state.operation_status.value != "completed"


def test_profile_custom_condition_signal_can_satisfy_database_file_read_goal(tmp_path) -> None:
    settings = AppSettings(
        runtime_store_backend="memory",
        runtime_store_dir=tmp_path / "rt",
        runtime_policy={"authorized_hosts": ["db.internal.local"]},
        lab_profile={
            "profile_id": "database-file-read-blackbox",
            "success_conditions": {
                "require_all": [
                    "database_file_read",
                    "goal_check_recorded",
                ]
            },
        },
    )
    orch = AppOrchestrator(settings=settings)
    planner = _ProfileDrivenPlanner(_DB_READ_CHAIN)
    orch.mission_planner = planner  # type: ignore[assignment]
    orch.stage_registry = StageAgentRegistry(  # type: ignore[list-item]
        [
            _agent(
                "access_pivot_agent",
                "ACCESS_PIVOT_STAGE",
                summary="database file read proof recorded",
                findings=[{"kind": "PostAccessObservation", "summary": "database file artifact observed"}],
                evidence_refs=["evidence::database-file-read"],
                runtime_hints={
                    "satisfied_conditions": ["database_file_read"],
                    "artifact_kind": "database_file",
                },
            ),
            _agent(
                "goal_agent",
                "GOAL_STAGE",
                summary="goal proof validated",
                findings=[{"kind": "GoalCheck", "goal_satisfied": True}],
                evidence_refs=["evidence::goal-proof"],
                runtime_hints={"goal_satisfied": True, "goal_evidence_refs": ["evidence::goal-proof"]},
            ),
        ]
    )

    orch.create_operation("op-database-file-read")
    orch.import_targets("op-database-file-read", [TargetHost(hostname="db.internal.local")])
    orch.start_operation("op-database-file-read")

    results = orch.run_until_quiescent(
        "op-database-file-read",
        graph_refs=[],
        planner_payload={"mission_goal": "prove the authorized database file read objective"},
        max_cycles=6,
    )

    state = orch.get_operation_state("op-database-file-read")
    progress = state.execution.metadata["success_condition_progress"]
    summary = orch.get_operation_run_summary("op-database-file-read", cycle_results=results)

    assert planner.decisions == ["dispatch:access_pivot_agent", "dispatch:goal_agent", "stop_success"]
    assert progress["conditions"]["database_file_read"]["satisfied"] is True
    assert progress["conditions"]["database_file_read"]["evidence_ids"] == ["evidence::database-file-read"]
    assert progress["eligible_for_stop"] is True
    assert summary.success is True
    assert summary.stop_reason == "success_conditions_satisfied"
