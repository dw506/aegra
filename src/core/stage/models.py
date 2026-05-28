"""Stage-level task and result contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.ag import GraphRef
from src.core.models.events import utc_now
from src.core.models.tg import BaseTaskNode, TaskType


class StageType(str, Enum):
    """Stage categories executed by dedicated Stage Agents."""

    RECON_STAGE = "RECON_STAGE"
    VULN_ANALYSIS_STAGE = "VULN_ANALYSIS_STAGE"
    EXPLOIT_STAGE = "EXPLOIT_STAGE"
    ACCESS_PIVOT_STAGE = "ACCESS_PIVOT_STAGE"
    GOAL_STAGE = "GOAL_STAGE"

    @property
    def task_type(self) -> TaskType:
        return TaskType(self.value)


class ToolTrace(BaseModel):
    """Audit record for one tool call inside a bounded stage loop."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    step: int = Field(ge=0)
    server_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool = False
    summary: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageTask(BaseModel):
    """TaskGraph stage task: complete a stage objective, not one tool call."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = Field(min_length=1)
    stage_type: StageType
    objective: str = Field(min_length=1)
    target_refs: list[GraphRef] = Field(default_factory=list)
    required_context: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=8, ge=1)
    risk_level: Literal["low", "medium", "high", "critical"] = "medium"
    priority: int = Field(default=50, ge=0, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_task_node(cls, task: BaseTaskNode) -> "StageTask":
        stage_type = StageType(task.task_type.value)
        return cls(
            task_id=task.id,
            stage_type=stage_type,
            objective=str(task.input_bindings.get("objective") or task.label),
            target_refs=list(task.target_refs),
            required_context=dict(task.input_bindings.get("required_context") or {}),
            success_criteria=[
                str(item) for item in task.input_bindings.get("success_criteria", []) if item is not None
            ],
            max_steps=int(task.input_bindings.get("max_steps") or task.max_attempts or 8),
            risk_level=str(task.input_bindings.get("risk_level") or "medium"),  # type: ignore[arg-type]
            priority=task.priority,
            metadata={
                **dict(task.input_bindings.get("metadata") or {}),
                "source_action_id": task.source_action_id,
                "tags": sorted(task.tags),
            },
        )


class StageResult(BaseModel):
    """Stage Agent output consumed by ResultApplier through an adapter."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str = Field(min_length=1)
    stage_task_id: str = Field(min_length=1)
    stage_type: StageType
    agent_name: str = Field(min_length=1)
    status: Literal["succeeded", "failed", "partial", "needs_replan"]
    summary: str = Field(min_length=1)

    observations: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)

    discovered_entities: list[dict[str, Any]] = Field(default_factory=list)
    discovered_relations: list[dict[str, Any]] = Field(default_factory=list)

    capabilities_gained: list[dict[str, Any]] = Field(default_factory=list)
    credentials: list[dict[str, Any]] = Field(default_factory=list)
    sessions: list[dict[str, Any]] = Field(default_factory=list)
    pivot_routes: list[dict[str, Any]] = Field(default_factory=list)
    privilege_contexts: list[dict[str, Any]] = Field(default_factory=list)

    next_stage_candidates: list[dict[str, Any]] = Field(default_factory=list)
    failed_hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    tool_trace: list[ToolTrace] = Field(default_factory=list)
    runtime_hints: dict[str, Any] = Field(default_factory=dict)
    writeback_hints: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())


__all__ = ["StageResult", "StageTask", "StageType", "ToolTrace"]
