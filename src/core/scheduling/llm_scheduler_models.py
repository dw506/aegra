"""Models for LLM-owned task scheduling decisions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.stage.models import GraphUpdateIntent


class ScheduledTask(BaseModel):
    """Minimal scheduled task context passed from SchedulerAgent to WorkerAgent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = Field(min_length=1)
    stage_type: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    target_refs: list[dict[str, Any]] = Field(default_factory=list)
    known_facts: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    policy_context: dict[str, Any] = Field(default_factory=dict)
    runtime_context: dict[str, Any] = Field(default_factory=dict)


class ScheduleDecision(BaseModel):
    """LLM scheduler output consumed by ResultApplier and WorkerAgent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    decision: Literal["dispatch", "defer", "retry", "wait", "blocked", "stop"]
    task_id: str | None = None
    worker_id: str | None = None
    rationale: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    scheduled_task: ScheduledTask | None = None
    runtime_update_intents: list[GraphUpdateIntent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _dispatch_requires_task(self) -> "ScheduleDecision":
        if self.decision == "dispatch":
            if not self.task_id or self.scheduled_task is None:
                raise ValueError("dispatch requires task_id and scheduled_task")
            if self.scheduled_task.task_id != self.task_id:
                raise ValueError("scheduled_task.task_id must match task_id")
        return self


__all__ = ["ScheduleDecision", "ScheduledTask"]
