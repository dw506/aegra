"""Deterministic black-box e2e for the success-condition verification loop.

Drives the public orchestrator surface (create -> import -> start ->
run_until_quiescent) with stub stage agents and a stub planner that obeys the
authoritative gate: it only emits stop_success when the runtime metadata says
success_condition_progress.eligible_for_stop is true. This proves the whole
flow end to end without needing a live LLM, MCP server, or docker lab:

    stage evidence -> ResultApplier/KG -> _update_success_condition_progress
    -> eligible_for_stop -> PlannerAgent stop_success -> loop terminates
    -> get_operation_run_summary reports success.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from _repo_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.app.orchestrator import AppOrchestrator, TargetHost
from src.app.settings import AppSettings
from src.core.stage.models import StageExecutionRequest, StageResult
from src.core.stage.registry import StageAgentRegistry


class ReconAgent:
    agent_name = "recon_agent"
    stage_type = "RECON_STAGE"

    def run(self, request: StageExecutionRequest) -> StageResult:
        return StageResult(
            operation_id=request.operation_id,
            stage_task_id=f"stage-{request.operation_id}-{request.cycle_index}-recon",
            stage_type="RECON_STAGE",
            agent_name=self.agent_name,
            status="succeeded",
            summary="entry-zone service discovery completed",
            findings=[{"type": "service_discovery", "summary": "http service discovered"}],
            evidence_refs=["evidence::recon-service"],
        )


class VulnAgent:
    agent_name = "vuln_analysis_agent"
    stage_type = "VULN_ANALYSIS_STAGE"

    def run(self, request: StageExecutionRequest) -> StageResult:
        return StageResult(
            operation_id=request.operation_id,
            stage_task_id=f"stage-{request.operation_id}-{request.cycle_index}-vuln",
            stage_type="VULN_ANALYSIS_STAGE",
            agent_name=self.agent_name,
            status="succeeded",
            summary="vulnerability candidate recorded",
            findings=[{"kind": "candidate_finding", "summary": "candidate recorded"}],
            evidence_refs=["evidence::vuln-candidate"],
        )


class GoalAgent:
    agent_name = "goal_agent"
    stage_type = "GOAL_STAGE"

    def run(self, request: StageExecutionRequest) -> StageResult:
        return StageResult(
            operation_id=request.operation_id,
            stage_task_id=f"stage-{request.operation_id}-{request.cycle_index}-goal",
            stage_type="GOAL_STAGE",
            agent_name=self.agent_name,
            status="succeeded",
            summary="goal proof validated",
            findings=[{"kind": "GoalCheck", "goal_satisfied": True}],
            evidence_refs=["evidence::goal-proof"],
            runtime_hints={"goal_satisfied": True, "goal_evidence_refs": ["evidence::goal-proof"]},
        )


# condition name -> (selected_agent, selected_stage) used to advance the chain
_CONDITION_TO_STAGE = [
    ("dmz_service_discovered", "recon_agent", "RECON_STAGE"),
    ("vulnerability_candidate_recorded", "vuln_analysis_agent", "VULN_ANALYSIS_STAGE"),
    ("goal_check_recorded", "goal_agent", "GOAL_STAGE"),
]


class GateAwarePlanner:
    """Only stops when the authoritative eligible_for_stop gate is true."""

    def __init__(self) -> None:
        self.calls = 0
        self.decisions: list[str] = []

    def run(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any] | None = None,
        recent_stage_results: list[dict[str, Any]] | None = None,
    ):
        from src.core.planning.models import PlannerDecision

        self.calls += 1
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
                reasoning_summary="success_condition_progress.eligible_for_stop is true",
                confidence=1.0,
            )

        missing = set(progress.get("missing") or [c for c, _, _ in _CONDITION_TO_STAGE])
        for condition, agent, stage in _CONDITION_TO_STAGE:
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
        # Nothing missing but gate not yet set -> dispatch goal as a fallback
        self.decisions.append("dispatch:goal_agent")
        return PlannerDecision(
            operation_id=operation_id,
            cycle_index=cycle_index,
            decision="dispatch_agent",
            selected_agent="goal_agent",
            selected_stage="GOAL_STAGE",
            objective=goal,
            risk_level="low",
            max_steps=1,
            confidence=0.5,
        )


def main() -> int:
    op_id = "blackbox-success-contract-e2e"
    with tempfile.TemporaryDirectory() as td:
        settings = AppSettings(
            runtime_store_backend="memory",
            runtime_store_dir=Path(td) / "rt",
            runtime_policy={"authorized_hosts": ["10.20.0.20"]},
            lab_profile={
                "profile_id": "blackbox-success-contract",
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
        planner = GateAwarePlanner()
        orch.mission_planner = planner  # type: ignore[assignment]
        orch.stage_registry = StageAgentRegistry([ReconAgent(), VulnAgent(), GoalAgent()])  # type: ignore[list-item]

        orch.create_operation(op_id)
        orch.import_targets(op_id, [TargetHost(address="10.20.0.20")])
        orch.start_operation(op_id)

        results = orch.run_until_quiescent(
            op_id,
            graph_refs=[],
            planner_payload={"mission_goal": "complete the authorized success contract"},
            max_cycles=8,
        )

        state = orch.get_operation_state(op_id)
        progress = state.execution.metadata.get("success_condition_progress", {})
        summary = orch.get_operation_run_summary(op_id, cycle_results=results)

        print("=== planner decisions ===")
        print(planner.decisions)
        print("=== cycles ===", len(results))
        print("=== success_condition_progress ===")
        for name, cond in (progress.get("conditions") or {}).items():
            print(f"  {name}: satisfied={cond.get('satisfied')} evidence={cond.get('evidence_ids')}")
        print("  eligible_for_stop:", progress.get("eligible_for_stop"))
        print("  all_required_satisfied:", progress.get("all_required_satisfied"))
        print("=== operation_status ===", state.operation_status.value)
        print("=== run_summary ===", summary.status, "| success=", summary.success, "| stop_reason=", summary.stop_reason)
        print("=== evidence_ids ===", summary.evidence_ids)

        checks = {
            "loop_terminated_via_stop_success": planner.decisions[-1] == "stop_success",
            "eligible_for_stop_true": bool(progress.get("eligible_for_stop")),
            "all_required_satisfied": bool(progress.get("all_required_satisfied")),
            "operation_completed": state.operation_status.value == "completed",
            "summary_success": summary.success is True and summary.status == "success",
            "stop_reason_contract": summary.stop_reason == "success_conditions_satisfied",
        }
        print("\n=== CHECKS ===")
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        ok = all(checks.values())
        print("\nRESULT:", "PASS - success-condition verification loop closed end to end" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
