"""Stage-level execution and result contracts."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.models.ag import GraphRef, stable_node_id
from src.core.models.graph_common import utc_now
from uuid import uuid4


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


CapabilityName: TypeAlias = Literal["recon", "analysis", "exploit", "pivot", "lateral", "goal", "evidence"]

CAPABILITY_NAMES: tuple[str, ...] = get_args(CapabilityName)


class ExtractedFact(BaseModel):
    """Deterministic fact extracted from a tool trace for KG writeback."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    fact_id: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    label: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)
    source_tool: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RoundDirective(BaseModel):
    """Planner-to-executor contract for one capability-scoped execution round."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str = Field(min_length=1)
    cycle_index: int = Field(ge=0)
    capability: CapabilityName
    objective: str = Field(min_length=1)
    target_refs: list[GraphRef] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    tool_hints: list[dict[str, Any]] = Field(default_factory=list)
    max_tools: int = Field(default=8, ge=1)
    success_hint: str | None = None
    required_context: dict[str, Any] = Field(default_factory=dict)
    risk_level: str = "medium"


class RoundResult(BaseModel):
    """Executor-to-planner result for one bounded execution round."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str = Field(min_length=1)
    cycle_index: int = Field(ge=0)
    capability: str = Field(min_length=1)
    tool_traces: list[ToolTrace] = Field(default_factory=list)
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    raw_summary: str = Field(min_length=1)
    log_ref: str | None = None
    objective_met: bool = False
    execution_result: "ExecutionResult | None" = None


class ExecutionRequest(BaseModel):
    """RoundDirective-derived request consumed by the execution agent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str = Field(min_length=1)
    cycle_index: int = Field(ge=0)
    agent_name: str = Field(min_length=1)
    capability: CapabilityName
    objective: str = Field(min_length=1)
    target_refs: list[GraphRef] = Field(default_factory=list)
    required_context: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)
    risk_level: str = "medium"
    max_steps: int = Field(default=8, ge=1)
    graph_summary: dict[str, Any] = Field(default_factory=dict)
    graph_history: dict[str, Any] = Field(default_factory=dict)
    runtime_context: dict[str, Any] = Field(default_factory=dict)
    policy_context: dict[str, Any] = Field(default_factory=dict)
    mcp_tool_catalog: dict[str, Any] = Field(default_factory=dict)
    # Execution-plane transport state (not surfaced to the LLM): active pivot
    # routes/sessions so the tool boundary can resolve transport behind the call.
    pivot_routes: list[dict[str, Any]] = Field(default_factory=list)
    sessions: list[dict[str, Any]] = Field(default_factory=list)
    task_brief: str | None = None
    autonomy_level: str | None = None
    allowed_tool_names: list[str] | str | None = None
    target_selection: str | None = None
    handoff_policy: str | None = None


class NextRoundSuggestion(BaseModel):
    """Suggested next-capability handoff emitted by a ExecutionResult."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    suggested_agent: str = Field(min_length=1)
    suggested_capability: CapabilityName
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    required_context_refs: list[Any] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    """Execution Agent output consumed directly by ResultApplier."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    result_id: str = Field(default_factory=lambda: f"execution-result-{uuid4().hex}")
    operation_id: str = Field(default="operation", min_length=1)
    execution_id: str = Field(min_length=1)
    # v3 capability tag, authoritative from the planner directive. ExecutionAgent
    # stamps it from RoundDirective.capability; the result tier reads it directly.
    capability: CapabilityName = "evidence"
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

    next_capability_candidates: list[dict[str, Any]] = Field(default_factory=list)
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
    next_capability_suggestion: dict[str, Any] | None = None
    handoff_suggestion: NextRoundSuggestion | None = None
    visual_summary: dict[str, Any] = Field(default_factory=dict)
    runtime_hints: dict[str, Any] = Field(default_factory=dict)
    writeback_hints: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())

    @model_validator(mode="before")
    @classmethod
    def normalize_execution_result_payload(cls, value: Any) -> Any:
        """Accept common finish payload wrappers before strict ExecutionResult validation."""

        if not isinstance(value, dict):
            return value
        payload = dict(value)
        nested = payload.get("execution_result") or payload.get("result")
        if isinstance(nested, dict):
            payload = {
                **{key: item for key, item in payload.items() if key not in {"execution_result", "result"}},
                **nested,
            }
        return payload

    @field_validator("observations", mode="before")
    @classmethod
    def normalize_observations(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [
                {"type": "note", "detail": item}
                if isinstance(item, str)
                else item
                for item in value
            ]
        if isinstance(value, str):
            return [{"type": "note", "detail": value}]
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.tool_traces and not self.tool_trace:
            self.tool_trace = list(self.tool_traces)
        elif self.tool_trace and not self.tool_traces:
            self.tool_traces = list(self.tool_trace)


__all__ = [
    "CAPABILITY_NAMES",
    "CapabilityName",
    "ExtractedFact",
    "GraphUpdateIntent",
    "RoundDirective",
    "RoundResult",
    "ExecutionRequest",
    "NextRoundSuggestion",
    "ExecutionResult",
    "ToolTrace",
]
