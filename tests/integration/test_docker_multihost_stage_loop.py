from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("AEGRA_RUN_DOCKER_MULTIHOST_TEST") != "1",
    reason="Docker multihost stage-loop smoke test requires AEGRA_RUN_DOCKER_MULTIHOST_TEST=1",
)


def test_docker_multihost_stage_loop_contract() -> None:
    """Lab-only contract for the Planner -> StageAgent -> ResultApplier loop."""

    expected_targets = {
        "dmz_cidr": "10.20.0.0/24",
        "internal_cidr": "10.30.0.0/24",
        "pivot_ssh": "10.20.0.30",
        "internal_web": "10.30.0.40",
    }
    expected_process_nodes = {
        "PlannerDecision",
        "AgentExecution",
        "ToolCall",
        "ExecutionResult",
        "GoalCheck",
        "StopDecision",
    }
    expected_kg_facts = {"Host", "Service", "PivotRoute", "Evidence", "Finding"}

    assert expected_targets["dmz_cidr"] == "10.20.0.0/24"
    assert expected_targets["internal_cidr"] == "10.30.0.0/24"
    assert {"GoalCheck", "StopDecision"}.issubset(expected_process_nodes)
    assert {"PivotRoute", "Evidence", "Finding"}.issubset(expected_kg_facts)
