"""Stage-level task and result contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.ag import GraphRef, stable_node_id
from src.core.models.events import utc_now
from src.core.models.tg import BaseTaskNode, TaskType


class StageType(str, Enum):
    """Stage categories executed by dedicated Stage Agents."""

    RECON = "recon"
    VULN_ANALYSIS = "vuln_analysis"
    EXPLOIT = "exploit"
    ACCESS_PIVOT = "access_pivot"
    GOAL = "goal"

    RECON_STAGE = "RECON_STAGE"
    VULN_ANALYSIS_STAGE = "VULN_ANALYSIS_STAGE"
    EXPLOIT_STAGE = "EXPLOIT_STAGE"
    ACCESS_PIVOT_STAGE = "ACCESS_PIVOT_STAGE"
    GOAL_STAGE = "GOAL_STAGE"

    @property
    def task_type(self) -> TaskType:
        legacy = {
            StageType.RECON: TaskType.RECON_STAGE,
            StageType.VULN_ANALYSIS: TaskType.VULN_ANALYSIS_STAGE,
            StageType.EXPLOIT: TaskType.EXPLOIT_STAGE,
            StageType.ACCESS_PIVOT: TaskType.ACCESS_PIVOT_STAGE,
            StageType.GOAL: TaskType.GOAL_STAGE,
        }
        if self in legacy:
            return legacy[self]
        return TaskType(self.value)

    @property
    def canonical(self) -> "StageType":
        return {
            StageType.RECON_STAGE: StageType.RECON,
            StageType.VULN_ANALYSIS_STAGE: StageType.VULN_ANALYSIS,
            StageType.EXPLOIT_STAGE: StageType.EXPLOIT,
            StageType.ACCESS_PIVOT_STAGE: StageType.ACCESS_PIVOT,
            StageType.GOAL_STAGE: StageType.GOAL,
        }.get(self, self)

    @property
    def legacy(self) -> "StageType":
        return {
            StageType.RECON: StageType.RECON_STAGE,
            StageType.VULN_ANALYSIS: StageType.VULN_ANALYSIS_STAGE,
            StageType.EXPLOIT: StageType.EXPLOIT_STAGE,
            StageType.ACCESS_PIVOT: StageType.ACCESS_PIVOT_STAGE,
            StageType.GOAL: StageType.GOAL_STAGE,
        }.get(self, self)


class GraphUpdateIntent(BaseModel):
    """A proposed graph/runtime mutation to be validated by ResultApplier."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    target_graph: Literal["KG", "AG", "TG", "Runtime"]
    operation: Literal["add", "update", "link", "mark_status", "append_evidence"]
    entity_type: str = ""
    entity_ref: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: Literal["planner", "stage_agent", "tool_trace", "result_applier"] = "stage_agent"


class GraphStateSnapshot(BaseModel):
    """Read-only graph/runtime/policy snapshot passed into LLM agents."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    kg_summary: dict[str, Any] = Field(default_factory=dict)
    ag_summary: dict[str, Any] = Field(default_factory=dict)
    tg_summary: dict[str, Any] = Field(default_factory=dict)
    runtime_summary: dict[str, Any] = Field(default_factory=dict)
    policy_summary: dict[str, Any] = Field(default_factory=dict)
    recent_evidence: list[dict[str, Any]] = Field(default_factory=list)
    active_sessions: list[dict[str, Any]] = Field(default_factory=list)
    known_assets: list[dict[str, Any]] = Field(default_factory=list)
    known_services: list[dict[str, Any]] = Field(default_factory=list)
    known_identities: list[dict[str, Any]] = Field(default_factory=list)
    current_task_context: dict[str, Any] = Field(default_factory=dict)


class ToolTrace(BaseModel):
    """Audit record for one tool call inside a bounded stage loop."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    trace_id: str = Field(default_factory=lambda: stable_node_id("tool-trace", {"ts": utc_now().isoformat()}))
    step: int = Field(default=0, ge=0)
    server_id: str = "mcp"
    tool_name: str = Field(min_length=1)
    tool_category: str = ""
    input_summary: str = ""
    raw_output_ref: str | None = None
    parsed_output: dict[str, Any] = Field(default_factory=dict)
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool = False
    summary: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int | str | None = None
    started_at: str = Field(default_factory=lambda: utc_now().isoformat())
    ended_at: str | None = None
    policy_check: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageTask(BaseModel):
    """TaskGraph stage task: complete a stage objective, not one tool call."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = Field(min_length=1)
    stage_type: StageType
    objective: str = Field(min_length=1)
    target_scope: dict[str, Any] = Field(default_factory=dict)
    prerequisites: list[str] = Field(default_factory=list)
    input_refs: list[GraphRef] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    status: str = "draft"
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

    result_id: str = Field(default_factory=lambda: stable_node_id("stage-result", {"ts": utc_now().isoformat()}))
    operation_id: str = Field(default="operation", min_length=1)
    stage_task_id: str = Field(min_length=1)
    stage_type: StageType
    agent_name: str = Field(min_length=1)
    status: Literal["success", "succeeded", "partial", "failed", "blocked", "need_more_info", "needs_replan"]
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
    evidence_refs: list[str] = Field(default_factory=list)
    graph_update_intents: list[GraphUpdateIntent] = Field(default_factory=list)
    tool_trace: list[ToolTrace] = Field(default_factory=list)
    tool_traces: list[ToolTrace] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high", "critical"] = "medium"
    policy_notes: list[str] = Field(default_factory=list)
    retry_recommendation: str | None = None
    replan_recommendation: str | None = None
    next_stage_suggestion: dict[str, Any] | None = None
    runtime_hints: dict[str, Any] = Field(default_factory=dict)
    writeback_hints: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())

    def model_post_init(self, __context: Any) -> None:
        if self.tool_traces and not self.tool_trace:
            self.tool_trace = list(self.tool_traces)
        elif self.tool_trace and not self.tool_traces:
            self.tool_traces = list(self.tool_trace)


class PlannerResult(BaseModel):
    """LLM planner output consumed only by ResultApplier."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str = Field(default="operation", min_length=1)
    reasoning_summary: str = ""
    new_stage_tasks: list[StageTask] = Field(default_factory=list)
    selected_next_task: StageTask | None = None
    task_updates: list[dict[str, Any]] = Field(default_factory=list)
    replan_needed: bool = False
    stop_condition: str | None = None
    graph_update_intents: list[GraphUpdateIntent] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def stage_tasks(self) -> list[StageTask]:
        return self.new_stage_tasks


__all__ = [
    "GraphStateSnapshot",
    "GraphUpdateIntent",
    "PlannerResult",
    "StageResult",
    "StageTask",
    "StageType",
    "ToolTrace",
]
