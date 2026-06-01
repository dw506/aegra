"""Planner decision contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.models.ag import GraphRef


PlannerDecisionType = Literal["dispatch_agent", "replan", "pause_for_review", "stop_success", "stop_failed"]
PlannerSelectedAgent = Literal[
    "recon_agent",
    "vuln_analysis_agent",
    "exploit_validation_agent",
    "access_pivot_agent",
    "goal_agent",
]
PlannerSelectedStage = Literal[
    "RECON_STAGE",
    "VULN_ANALYSIS_STAGE",
    "EXPLOIT_STAGE",
    "ACCESS_PIVOT_STAGE",
    "GOAL_STAGE",
]
PlannerRiskLevel = Literal["low", "medium", "high", "critical"]


class PlannerDecision(BaseModel):
    """PlannerAgent main output consumed by the stage dispatcher."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str = Field(min_length=1)
    cycle_index: int = Field(ge=0)
    decision: PlannerDecisionType
    selected_agent: PlannerSelectedAgent | None = None
    selected_stage: PlannerSelectedStage | None = None
    objective: str = Field(min_length=1)
    target_refs: list[GraphRef] = Field(default_factory=list)
    required_context: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)
    risk_level: PlannerRiskLevel
    max_steps: int = Field(ge=1)
    reasoning_summary: str = ""
    handoff_acceptance: dict[str, Any] | None = None
    stop_condition: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dispatch_fields(self) -> "PlannerDecision":
        """Keep dispatch decisions explicit and non-dispatch decisions unassigned."""

        if self.decision == "dispatch_agent":
            if self.selected_agent is None:
                raise ValueError("selected_agent is required when decision is dispatch_agent")
            if self.selected_stage is None:
                raise ValueError("selected_stage is required when decision is dispatch_agent")
        elif self.selected_agent is None and self.decision == "dispatch_agent":
            raise ValueError("selected_agent cannot be empty for dispatch_agent")
        elif self.selected_agent is None:
            return self
        return self


__all__ = [
    "PlannerDecision",
    "PlannerDecisionType",
    "PlannerRiskLevel",
    "PlannerSelectedAgent",
    "PlannerSelectedStage",
]
