"""Stage-level execution and result contracts."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.models.ag import GraphRef, stable_node_id
from src.core.models.events import utc_now


StageName: TypeAlias = Literal[
    "RECON_STAGE",
    "VULN_ANALYSIS_STAGE",
    "EXPLOIT_STAGE",
    "ACCESS_PIVOT_STAGE",
    "GOAL_STAGE",
]

STAGE_NAMES: tuple[str, ...] = get_args(StageName)

CANONICAL_STAGE_BY_ALIAS: dict[str, str] = {
    "recon": "RECON_STAGE",
    "RECON": "RECON_STAGE",
    "RECON_STAGE": "RECON_STAGE",
    "vuln_analysis": "VULN_ANALYSIS_STAGE",
    "VULN_ANALYSIS": "VULN_ANALYSIS_STAGE",
    "VULN_ANALYSIS_STAGE": "VULN_ANALYSIS_STAGE",
    "exploit": "EXPLOIT_STAGE",
    "EXPLOIT": "EXPLOIT_STAGE",
    "EXPLOIT_STAGE": "EXPLOIT_STAGE",
    "access_pivot": "ACCESS_PIVOT_STAGE",
    "ACCESS_PIVOT": "ACCESS_PIVOT_STAGE",
    "ACCESS_PIVOT_STAGE": "ACCESS_PIVOT_STAGE",
    "goal": "GOAL_STAGE",
    "GOAL": "GOAL_STAGE",
    "GOAL_STAGE": "GOAL_STAGE",
}


def normalize_stage_name(value: Any) -> StageName:
    """Return the canonical stage name accepted by the new stage pipeline."""

    raw = getattr(value, "value", value)
    text = str(raw)
    normalized = CANONICAL_STAGE_BY_ALIAS.get(text)
    if normalized is None:
        raise ValueError(f"unsupported stage_type: {text}")
    return normalized  # type: ignore[return-value]


class GraphUpdateIntent(BaseModel):
    """A proposed graph/runtime mutation to be validated by ResultApplier."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    target_graph: Literal["KG", "AG", "Runtime"]
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


class StageExecutionRequest(BaseModel):
    """PlannerDecision-derived request consumed by a StageAgent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str = Field(min_length=1)
    cycle_index: int = Field(ge=0)
    agent_name: str = Field(min_length=1)
    stage_type: StageName
    objective: str = Field(min_length=1)
    target_refs: list[GraphRef] = Field(default_factory=list)
    required_context: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)
    risk_level: str = "medium"
    max_steps: int = Field(default=8, ge=1)
    kg_snapshot: dict[str, Any] = Field(default_factory=dict)
    ag_process_history: dict[str, Any] = Field(default_factory=dict)
    runtime_context: dict[str, Any] = Field(default_factory=dict)
    policy_context: dict[str, Any] = Field(default_factory=dict)
    mcp_tool_catalog: dict[str, Any] = Field(default_factory=dict)

    @field_validator("stage_type", mode="before")
    @classmethod
    def normalize_stage_type(cls, value: Any) -> StageName:
        return normalize_stage_name(value)


class StageHandoffSuggestion(BaseModel):
    """Suggested next agent/stage handoff emitted by a StageResult."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    suggested_agent: str = Field(min_length=1)
    suggested_stage: StageName
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    required_context_refs: list[Any] = Field(default_factory=list)

    @field_validator("suggested_stage", mode="before")
    @classmethod
    def normalize_suggested_stage(cls, value: Any) -> StageName:
        return normalize_stage_name(value)


class StageResult(BaseModel):
    """Stage Agent output consumed by ResultApplier through an adapter."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    result_id: str = Field(default_factory=lambda: stable_node_id("stage-result", {"ts": utc_now().isoformat()}))
    operation_id: str = Field(default="operation", min_length=1)
    stage_task_id: str = Field(min_length=1)
    stage_type: StageName
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
    handoff_suggestion: StageHandoffSuggestion | None = None
    runtime_hints: dict[str, Any] = Field(default_factory=dict)
    writeback_hints: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())

    @field_validator("stage_type", mode="before")
    @classmethod
    def normalize_stage_type(cls, value: Any) -> StageName:
        return normalize_stage_name(value)

    def model_post_init(self, __context: Any) -> None:
        if self.tool_traces and not self.tool_trace:
            self.tool_trace = list(self.tool_traces)
        elif self.tool_trace and not self.tool_traces:
            self.tool_traces = list(self.tool_trace)


__all__ = [
    "CANONICAL_STAGE_BY_ALIAS",
    "GraphStateSnapshot",
    "GraphUpdateIntent",
    "STAGE_NAMES",
    "StageExecutionRequest",
    "StageHandoffSuggestion",
    "StageName",
    "StageResult",
    "ToolTrace",
    "normalize_stage_name",
]
