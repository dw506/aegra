from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.models.ag import GraphRef
from src.core.planning.models import PlannerDecision


def test_planner_decision_dispatches_directly_to_stage_dispatcher_contract() -> None:
    decision = PlannerDecision(
        operation_id="op-1",
        cycle_index=1,
        decision="dispatch_agent",
        selected_agent="recon_agent",
        selected_stage="RECON_STAGE",
        objective="Enumerate authorized target surface",
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        required_context={"scope": "authorized"},
        success_criteria=["host profile is current"],
        risk_level="low",
        max_steps=4,
        reasoning_summary="Recon is the next required stage.",
        handoff_acceptance={"accepted": True},
        stop_condition=None,
        confidence=0.82,
        metadata={"source": "planner_agent"},
    )

    payload = decision.model_dump(mode="json")

    assert payload["selected_agent"] == "recon_agent"
    assert payload["selected_stage"] == "RECON_STAGE"
    assert payload["target_refs"][0]["ref_id"] == "host-1"


def test_planner_decision_allows_empty_agent_for_non_dispatch_decision() -> None:
    decision = PlannerDecision(
        operation_id="op-1",
        cycle_index=2,
        decision="pause_for_review",
        selected_agent=None,
        selected_stage=None,
        objective="Wait for human approval",
        target_refs=[],
        required_context={},
        success_criteria=["review completed"],
        risk_level="medium",
        max_steps=1,
        reasoning_summary="Policy requires review before continuing.",
        handoff_acceptance=None,
        stop_condition="approval required",
        confidence=0.7,
        metadata={},
    )

    assert decision.selected_agent is None
    assert decision.decision == "pause_for_review"


def test_planner_decision_accepts_handoff_acceptance_checklist() -> None:
    decision = PlannerDecision(
        operation_id="op-1",
        cycle_index=1,
        decision="dispatch_agent",
        selected_agent="recon_agent",
        selected_stage="RECON_STAGE",
        objective="Enumerate authorized target surface",
        target_refs=[],
        required_context={},
        success_criteria=["host profile is current"],
        risk_level="low",
        max_steps=3,
        reasoning_summary="Recon is the next required stage.",
        handoff_acceptance=["evidence is recorded", "scope stays authorized"],
        stop_condition=None,
        confidence=0.8,
        metadata={},
    )

    assert decision.handoff_acceptance == ["evidence is recorded", "scope stays authorized"]


def test_planner_decision_accepts_handoff_acceptance_text() -> None:
    decision = PlannerDecision(
        operation_id="op-1",
        cycle_index=1,
        decision="dispatch_agent",
        selected_agent="recon_agent",
        selected_stage="RECON_STAGE",
        objective="Enumerate authorized target surface",
        target_refs=[],
        required_context={},
        success_criteria=["host profile is current"],
        risk_level="low",
        max_steps=3,
        reasoning_summary="Recon is the next required stage.",
        handoff_acceptance="evidence is recorded",
        stop_condition=None,
        confidence=0.8,
        metadata={},
    )

    assert decision.handoff_acceptance == "evidence is recorded"


def test_planner_decision_requires_agent_and_stage_for_dispatch() -> None:
    base_payload = {
        "operation_id": "op-1",
        "cycle_index": 1,
        "decision": "dispatch_agent",
        "selected_agent": "goal_agent",
        "selected_stage": "GOAL_STAGE",
        "objective": "Validate objective completion",
        "target_refs": [],
        "required_context": {},
        "success_criteria": ["objective satisfied"],
        "risk_level": "low",
        "max_steps": 3,
        "reasoning_summary": "Goal validation is ready.",
        "handoff_acceptance": None,
        "stop_condition": None,
        "confidence": 0.9,
        "metadata": {},
    }

    with pytest.raises(ValidationError, match="selected_agent is required"):
        PlannerDecision(**{**base_payload, "selected_agent": None})

    with pytest.raises(ValidationError, match="selected_stage is required"):
        PlannerDecision(**{**base_payload, "selected_stage": None})


def test_planner_decision_model_has_no_task_graph_dependencies() -> None:
    source = Path("src/core/planning/models.py").read_text(encoding="utf-8")

    assert "StageTask" not in source
    assert "TaskGraph" not in source
    assert "src.core.models.tg" not in source
