"""Planner decision contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.models.ag import GraphRef
from src.core.stage.models import RoundDirective


PlannerDecisionType = Literal["dispatch_agent", "replan", "pause_for_review", "stop_success", "stop_failed"]
PlannerRiskLevel = Literal["low", "medium", "high", "critical"]
PlannerOutcomeAction = Literal["execute", "replan", "pause_for_review", "stop_success", "stop_failed"]


class PlannerDecision(BaseModel):
    """PlannerAgent main output consumed by the stage dispatcher."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str = Field(min_length=1)
    cycle_index: int = Field(ge=0)
    decision: PlannerDecisionType
    selected_agent: str | None = None
    selected_stage: str | None = None
    objective: str = Field(min_length=1)
    target_refs: list[GraphRef] = Field(default_factory=list)
    required_context: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)
    risk_level: PlannerRiskLevel
    max_steps: int = Field(ge=1)
    task_brief: str | None = None
    autonomy_level: str | None = None
    allowed_tool_names: list[str] | str | None = None
    target_selection: str | None = None
    handoff_policy: str | None = None
    reasoning_summary: str = ""
    handoff_acceptance: dict[str, Any] | list[Any] | str | None = None
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


class PlannerOutcome(BaseModel):
    """P3 planner result: either a RoundDirective or a terminal control action."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str = Field(min_length=1)
    cycle_index: int = Field(ge=0)
    action: PlannerOutcomeAction
    directive: RoundDirective | None = None
    reason: str = ""
    stop_condition: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_action_payload(self) -> "PlannerOutcome":
        if self.action == "execute" and self.directive is None:
            raise ValueError("directive is required when action=execute")
        if self.action != "execute" and self.directive is not None:
            raise ValueError("directive is only valid when action=execute")
        return self


__all__ = [
    "PlannerDecision",
    "PlannerDecisionType",
    "PlannerOutcome",
    "PlannerOutcomeAction",
    "PlannerRiskLevel",
]
