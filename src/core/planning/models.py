"""Planner decision contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.stage.models import RoundDirective


PlannerRiskLevel = Literal["low", "medium", "high", "critical"]
PlannerOutcomeAction = Literal["execute", "replan", "pause_for_review", "stop_success", "stop_failed"]


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
    "PlannerOutcome",
    "PlannerOutcomeAction",
    "PlannerRiskLevel",
]
