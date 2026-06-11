from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.app.orchestrator import AppOrchestrator, TargetHost
from src.app.settings import AppSettings
from src.core.agents.agent_protocol import GraphRef, GraphScope
from src.core.planning.models import PlannerDecision


class CapturingMissionPlanner:
    def __init__(self) -> None:
        self.graph_context: dict[str, Any] | None = None
        self.policy_context: dict[str, Any] | None = None

    def run(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any] | None = None,
        recent_stage_results: list[dict[str, Any]] | None = None,
    ) -> PlannerDecision:
        del recent_stage_results
        self.graph_context = graph_context
        self.policy_context = policy_context
        return PlannerDecision(
            operation_id=str(graph_context["operation_id"]),
            cycle_index=int(graph_context["cycle_index"]),
            decision="replan",
            selected_agent=None,
            selected_stage=None,
            objective=goal,
            risk_level="low",
            max_steps=1,
            reasoning_summary="captured startup context",
            confidence=0.7,
        )


class StaticMCPClient:
    def list_tools(self) -> dict[str, Any]:
        return {"pentest-tools": {"tools": [{"name": "nmap_scan"}, {"name": "http_probe"}]}}


class SequenceMissionPlanner:
    def __init__(self, decisions: list[str]) -> None:
        self.decisions = list(decisions)
        self.calls = 0

    def run(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any] | None = None,
        recent_stage_results: list[dict[str, Any]] | None = None,
    ) -> PlannerDecision:
        del policy_context, recent_stage_results
        self.calls += 1
        decision = self.decisions.pop(0) if self.decisions else "stop_failed"
        return PlannerDecision(
            operation_id=str(graph_context["operation_id"]),
            cycle_index=int(graph_context["cycle_index"]),
            decision=decision,  # type: ignore[arg-type]
            selected_agent=None,
            selected_stage=None,
            objective=goal,
            risk_level="low",
            max_steps=1,
            reasoning_summary=f"planner returned {decision}",
            stop_condition="goal_satisfied" if decision == "stop_success" else None,
            confidence=0.7,
        )


def test_settings_loads_lab_profile_from_json_file(tmp_path: Path) -> None:
    profile_path = tmp_path / "lab_profile.json"
    profile_path.write_text(json.dumps({"profile_id": "docker-multihost", "zones": ["dmz", "internal"]}), encoding="utf-8")

    settings = AppSettings(runtime_store_backend="memory", lab_profile_path=profile_path)

    assert settings.load_lab_profile() == {
        "profile_id": "docker-multihost",
        "zones": ["dmz", "internal"],
        "loaded_from": str(profile_path.resolve()),
    }


def test_planner_receives_blackbox_policy_and_tool_catalog_without_lab_topology(tmp_path: Path) -> None:
    settings = AppSettings(
        runtime_store_backend="memory",
        runtime_store_dir=tmp_path / "runtime",
        runtime_policy={"authorized_hosts": ["10.20.0.0/24"]},
        lab_profile={"profile_id": "docker-multihost", "zones": ["dmz", "internal"]},
    )
    orchestrator = AppOrchestrator(settings=settings)
    planner = CapturingMissionPlanner()
    orchestrator.mission_planner = planner  # type: ignore[assignment]
    orchestrator.mcp_client = StaticMCPClient()  # type: ignore[assignment]

    orchestrator.create_operation("op-startup-context")
    orchestrator.import_targets("op-startup-context", [TargetHost(address="10.20.0.20")])
    orchestrator.start_operation("op-startup-context")

    orchestrator.run_operation_cycle(
        "op-startup-context",
        graph_refs=[GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph")],
        planner_payload={"mission_goal": "validate authorized internal reachability"},
    )

    assert planner.graph_context is not None
    assert planner.policy_context is not None
    assert "lab_profile" not in planner.graph_context
    assert planner.policy_context["scope_source"] == "imported_targets"
    assert [item["value"] for item in planner.policy_context["authorized_targets"]] == ["10.20.0.20"]
    assert planner.graph_context["mcp_tool_catalog"]["pentest-tools"]["tools"][0]["name"] == "nmap_scan"
    state = orchestrator.get_operation_state("op-startup-context")
    assert state.execution.metadata["lab_profile"] == {"profile_id": "docker-multihost"}


def test_run_until_quiescent_continues_after_planner_replan(tmp_path: Path) -> None:
    settings = AppSettings(
        runtime_store_backend="memory",
        runtime_store_dir=tmp_path / "runtime",
        runtime_policy={"authorized_hosts": ["10.20.0.0/24"]},
        lab_profile={"profile_id": "docker-multihost"},
    )
    orchestrator = AppOrchestrator(settings=settings)
    planner = SequenceMissionPlanner(["replan", "replan", "stop_success"])
    orchestrator.mission_planner = planner  # type: ignore[assignment]

    orchestrator.create_operation("op-replan-loop")
    orchestrator.import_targets("op-replan-loop", [TargetHost(address="10.20.0.20")])
    orchestrator.start_operation("op-replan-loop")

    results = orchestrator.run_until_quiescent(
        "op-replan-loop",
        graph_refs=[GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph")],
        planner_payload={"mission_goal": "validate authorized internal reachability"},
        max_cycles=5,
        max_replans=3,
    )

    assert [result.planner_decision.decision for result in results] == [
        "replan",
        "replan",
        "stop_success",
    ]
    assert planner.calls == 3
    assert results[-1].stopped is True
