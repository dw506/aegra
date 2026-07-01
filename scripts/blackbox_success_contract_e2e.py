"""Deterministic black-box e2e for the success-condition verification loop.

Drives the public orchestrator surface (create -> import -> start ->
run_until_quiescent) with stub execution rounds and a stub planner that obeys
the authoritative gate: it only emits stop_success when the runtime metadata says
success_condition_progress.eligible_for_stop is true. This proves the whole
flow end to end without needing a live LLM, MCP server, or docker lab:

    execution evidence -> ResultApplier/KG -> _update_success_condition_progress
    -> eligible_for_stop -> planner stop_success -> loop terminates
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
from src.core.execution.execution_agent import ExecutionAgent
from src.core.execution.models import ExecutionRequest, ExecutionResult, RoundDirective, ToolTrace
from src.core.planning.models import PlannerOutcome


def _agent(agent_name: str, objective_key: str, **result_kwargs: Any):
    """One objective-scoped stub executor returning a fixed ExecutionResult."""

    class _StubAgent:
        def __init__(self) -> None:
            self.agent_name = agent_name
            self.objective_key = objective_key

        def run(self, request: ExecutionRequest) -> ExecutionResult:
            return ExecutionResult(
                operation_id=request.operation_id,
                execution_id=f"execution-{request.operation_id}-{request.cycle_index}-{agent_name}",
                agent_name=agent_name,
                status="succeeded",
                **result_kwargs,
            )

    return _StubAgent()


def _executor(agents) -> ExecutionAgent:
    """Wrap objective-scoped stub agents into the single ExecutionAgent."""

    by_objective = {agent.objective_key: agent for agent in agents}

    class _Dispatch:
        agent_name = "execution_agent"

        def run(self, request: ExecutionRequest) -> ExecutionResult:
            return by_objective[request.objective].run(request)

    return ExecutionAgent(_Dispatch())  # type: ignore[arg-type]


# condition name -> (selected_agent, objective key) used to advance the chain
_CONDITION_CHAIN = [
    ("dmz_service_discovered", "recon_agent", "recon"),
    ("vulnerability_candidate_recorded", "vuln_analysis_agent", "analysis"),
    ("goal_check_recorded", "goal_agent", "goal"),
]


class GateAwarePlanner:
    """Only stops when the authoritative eligible_for_stop gate is true."""

    def __init__(self) -> None:
        self.calls = 0
        self.decisions: list[str] = []

    def decide(self, *, goal: str, graph_context: dict[str, Any], **_: Any) -> PlannerOutcome:
        self.calls += 1
        progress = graph_context.get("success_condition_progress") or {}
        operation_id = str(graph_context["operation_id"])
        cycle_index = int(graph_context["cycle_index"])

        if progress.get("eligible_for_stop"):
            self.decisions.append("stop_success")
            return PlannerOutcome(
                operation_id=operation_id,
                cycle_index=cycle_index,
                action="stop_success",
                reason=goal,
                stop_condition="contract_satisfied",
                confidence=1.0,
            )

        missing = set(progress.get("missing") or [c for c, _, _ in _CONDITION_CHAIN])
        for condition, agent, objective_key in _CONDITION_CHAIN:
            if condition in missing:
                self.decisions.append(f"dispatch:{agent}")
                return _execute_outcome(operation_id, cycle_index, objective_key, 0.9)

        # Nothing missing but gate not yet set -> dispatch goal as a fallback.
        self.decisions.append("dispatch:goal_agent")
        return _execute_outcome(operation_id, cycle_index, "goal", 0.5)


def _execute_outcome(
    operation_id: str,
    cycle_index: int,
    objective: str,
    confidence: float,
) -> PlannerOutcome:
    return PlannerOutcome(
        operation_id=operation_id,
        cycle_index=cycle_index,
        action="execute",
        directive=RoundDirective(
            operation_id=operation_id,
            cycle_index=cycle_index,
            objective=objective,
            max_tools=1,
            risk_level="low",
        ),
        confidence=confidence,
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
        orch.execution_agent = _executor(  # type: ignore[assignment]
            [
                _agent(
                    "recon_agent",
                    "recon",
                    summary="entry-zone service discovery completed",
                    evidence_refs=["evidence::recon-service"],
                    tool_trace=[ToolTrace(tool_name="safe_probe", success=True, summary="http service discovered")],
                ),
                _agent(
                    "vuln_analysis_agent",
                    "analysis",
                    summary="vulnerability candidate recorded",
                    evidence_refs=["evidence::vuln-candidate"],
                    tool_trace=[ToolTrace(tool_name="safe_probe", success=True, summary="candidate recorded")],
                ),
                _agent(
                    "goal_agent",
                    "goal",
                    summary="goal proof validated",
                    evidence_refs=["evidence::goal-proof"],
                    tool_trace=[ToolTrace(tool_name="safe_probe", success=True, summary="goal proof validated")],
                    runtime_hints={"goal_satisfied": True, "goal_evidence_refs": ["evidence::goal-proof"]},
                ),
            ]
        )

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

