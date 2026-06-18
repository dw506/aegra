from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.models.ag import GraphRef
from src.core.planning.models import PlannerOutcome
from src.core.stage.models import RoundDirective


def test_planner_outcome_executes_round_directive() -> None:
    outcome = PlannerOutcome(
        operation_id="op-1",
        cycle_index=1,
        action="execute",
        directive=RoundDirective(
            operation_id="op-1",
            cycle_index=1,
            capability="recon",
            objective="Enumerate authorized target surface",
            target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
            allowed_tools=["nmap_scan"],
            max_tools=4,
            risk_level="low",
        ),
        reason="Missing service evidence maps to recon capability.",
        confidence=0.82,
    )

    payload = outcome.model_dump(mode="json")

    assert payload["action"] == "execute"
    assert payload["directive"]["capability"] == "recon"
    assert payload["directive"]["target_refs"][0]["ref_id"] == "host-1"


def test_planner_outcome_rejects_directive_for_stop() -> None:
    with pytest.raises(ValidationError, match="directive is only valid"):
        PlannerOutcome(
            operation_id="op-1",
            cycle_index=1,
            action="stop_success",
            directive=RoundDirective(
                operation_id="op-1",
                cycle_index=1,
                capability="goal",
                objective="done",
            ),
        )


def test_planner_outcome_requires_directive_for_execute() -> None:
    with pytest.raises(ValidationError, match="directive is required"):
        PlannerOutcome(
            operation_id="op-1",
            cycle_index=1,
            action="execute",
            directive=None,
        )


def test_planner_models_have_no_task_graph_dependencies() -> None:
    source = Path("src/core/planning/models.py").read_text(encoding="utf-8")

    assert "StageTask" not in source
    assert "TaskGraph" not in source
    assert "src.core.models.tg" not in source
