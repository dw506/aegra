"""Objective-scoped execution and result contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.models.ag import GraphRef, stable_node_id
from src.core.models.graph_common import utc_now
from uuid import uuid4


class ToolTrace(BaseModel):
    """Audit record for one tool call inside a bounded execution loop."""

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


class RoundDirective(BaseModel):
    """Planner-to-executor contract for one bounded, objective-scoped round.

    The planner states a free-text objective and the executor does whatever that
    objective needs.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str = Field(min_length=1)
    cycle_index: int = Field(ge=0)
    objective: str = Field(min_length=1)
    target_refs: list[GraphRef] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    tool_hints: list[dict[str, Any]] = Field(default_factory=list)
    max_tools: int = Field(default=16, ge=1)
    success_hint: str | None = None
    required_context: dict[str, Any] = Field(default_factory=dict)
    risk_level: str = "medium"

    @field_validator("target_refs", mode="before")
    @classmethod
    def _coerce_target_refs(cls, value: Any) -> Any:
        # The planner LLM tends to emit node ids as bare strings ("kg-host::..",
        # "kg:host::..", "host::..") rather than GraphRef objects. Normalize them.
        if not isinstance(value, list):
            return value
        coerced: list[Any] = []
        for item in value:
            if not isinstance(item, str):
                coerced.append(item)
                continue
            graph, ref_id = "kg", item
            for prefix, resolved in (("kg-", "kg"), ("ag-", "ag"), ("kg:", "kg"), ("ag:", "ag")):
                if item.startswith(prefix):
                    graph, ref_id = resolved, item[len(prefix):]
                    break
            if ref_id:
                coerced.append({"graph": graph, "ref_id": ref_id})
        return coerced

    @field_validator("tool_hints", mode="before")
    @classmethod
    def _coerce_tool_hints(cls, value: Any) -> Any:
        # The planner LLM tends to emit free-text hints; wrap bare strings.
        if not isinstance(value, list):
            return value
        return [{"hint": item} if isinstance(item, str) else item for item in value]


class ExecutionRequest(BaseModel):
    """RoundDirective-derived request consumed by the execution agent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str = Field(min_length=1)
    cycle_index: int = Field(ge=0)
    agent_name: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    target_refs: list[GraphRef] = Field(default_factory=list)
    required_context: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)
    risk_level: str = "medium"
    max_steps: int = Field(default=16, ge=1)
    graph_summary: dict[str, Any] = Field(default_factory=dict)
    graph_history: dict[str, Any] = Field(default_factory=dict)
    runtime_context: dict[str, Any] = Field(default_factory=dict)
    policy_context: dict[str, Any] = Field(default_factory=dict)
    mcp_tool_catalog: dict[str, Any] = Field(default_factory=dict)
    # Execution-plane transport state (not surfaced to the LLM): active pivot
    # routes/sessions so the tool boundary can resolve transport behind the call.
    pivot_routes: list[dict[str, Any]] = Field(default_factory=list)
    sessions: list[dict[str, Any]] = Field(default_factory=list)
    execution_brief: str | None = None
    autonomy_level: str | None = None
    allowed_tool_names: list[str] | str | None = None
    target_selection: str | None = None
    handoff_policy: str | None = None


class ExecutionResult(BaseModel):
    """Execution Agent output consumed directly by ResultApplier."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    result_id: str = Field(default_factory=lambda: f"execution-result-{uuid4().hex}")
    operation_id: str = Field(default="operation", min_length=1)
    execution_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    status: Literal["success", "succeeded", "partial", "failed", "blocked", "need_more_info", "needs_replan"]
    summary: str = Field(min_length=1)

    # All channel-② LLM self-report is GONE (Step 5 F3-KG / A2): observations,
    # findings, discovered_entities/relations and the runtime
    # sessions/pivot_routes/credentials lists no longer ride on the result. KG
    # machine facts derive solely from tool_trace via the deterministic
    # ToolTraceFactExtractor (channel ①, tool = authority); runtime
    # sessions/routes/credentials derive from tool_trace via
    # PhaseTwoResultApplier._harvest_runtime_facts. evidence_refs (tool-derived,
    # load-bearing for success/oracle/AG) stays.
    evidence_refs: list[str] = Field(default_factory=list)
    tool_trace: list[ToolTrace] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    replan_recommendation: str | None = None
    runtime_hints: dict[str, Any] = Field(default_factory=dict)
    writeback_hints: dict[str, Any] = Field(default_factory=dict)

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


__all__ = [
    "RoundDirective",
    "ExecutionRequest",
    "ExecutionResult",
    "ToolTrace",
]
