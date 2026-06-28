"""LangGraph-ready planner agent loop.

The planner is a real tool-use agent: each turn the LLM either calls a read
tool to inspect the graph or emits a final ``PlannerOutcome``. This module keeps
the three pieces deliberately separate so a later LangGraph adoption (Step 6) is
a driver swap, not a rewrite:

- ``PlannerLoopState``       -> the typed state threaded through the loop
- ``decide_node`` / ``act_node`` -> pure ``State -> State`` node functions
- ``run_planner_loop``      -> a thin Python driver (becomes a StateGraph later)

The driver makes no domain decisions; it only iterates decide/act until the LLM
returns a decision or the read-step budget is exhausted.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.core.planning.models import PlannerOutcome


class PlannerLoopState(BaseModel):
    """Typed state for one planner decision turn-sequence (the future LangGraph State)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str
    cycle_index: int
    goal: str
    policy_context: dict[str, Any] = Field(default_factory=dict)
    seed_context: dict[str, Any] = Field(default_factory=dict)
    recent_execution_results: list[dict[str, Any]] = Field(default_factory=list)
    read_log: list[dict[str, Any]] = Field(default_factory=list)
    step: int = 0
    max_steps: int = Field(default=6, ge=1)
    pending_call: dict[str, Any] | None = None
    decision: PlannerOutcome | None = None


class PlannerTurn(Protocol):
    """One LLM turn: inspect state, return a read tool_call or a final outcome."""

    def run_turn(
        self,
        *,
        state: PlannerLoopState,
        read_manifest: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return {"tool_call": {...}} or {"outcome": PlannerOutcome, "raw": {...}}."""


class ReadToolHost(Protocol):
    """Read-tool surface the act node dispatches against (PlannerGraphTools)."""

    def apply_read_call(self, name: str, arguments: dict[str, Any] | None = None) -> Any: ...

    @staticmethod
    def read_tool_manifest() -> list[dict[str, Any]]: ...


def decide_node(
    state: PlannerLoopState,
    *,
    planner: PlannerTurn,
    read_manifest: list[dict[str, Any]],
) -> PlannerLoopState:
    """Run one LLM turn; set either ``pending_call`` (read tool) or ``decision``."""

    turn = planner.run_turn(state=state, read_manifest=read_manifest)
    outcome = turn.get("outcome") if isinstance(turn, dict) else None
    tool_call = turn.get("tool_call") if isinstance(turn, dict) else None
    if isinstance(outcome, PlannerOutcome):
        state.decision = outcome
        state.pending_call = None
    elif isinstance(tool_call, dict):
        state.pending_call = dict(tool_call)
    else:
        # Neither a tool call nor a valid outcome: leave both unset; the driver
        # treats this as "no decision" and bails to the exhausted fallback.
        state.pending_call = None
    return state


def act_node(state: PlannerLoopState, *, graph_tools: ReadToolHost) -> PlannerLoopState:
    """Execute the pending read tool call and append its result to ``read_log``."""

    call = state.pending_call or {}
    name = str(call.get("tool") or call.get("name") or "")
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    result = graph_tools.apply_read_call(name, args)
    state.read_log = [*state.read_log, {"tool": name, "arguments": dict(args), "result": result}]
    state.pending_call = None
    state.step += 1
    return state


def _exhausted_outcome(state: PlannerLoopState) -> PlannerOutcome:
    return PlannerOutcome(
        operation_id=state.operation_id,
        cycle_index=state.cycle_index,
        action="replan",
        reason="planner agent loop ended without a decision",
        stop_condition="planner_loop_exhausted",
        confidence=0.0,
        metadata={"planner": "agent_loop", "accepted": False, "read_steps": len(state.read_log)},
    )


def run_planner_loop(
    state: PlannerLoopState,
    *,
    planner: PlannerTurn,
    graph_tools: ReadToolHost | None,
) -> PlannerOutcome:
    """Thin driver: iterate decide/act until a decision or the read budget runs out.

    Step 6 replaces this for-loop with a compiled LangGraph StateGraph; the node
    functions and ``PlannerLoopState`` are reused unchanged.
    """

    read_manifest = graph_tools.read_tool_manifest() if graph_tools is not None else []
    for _ in range(state.max_steps):
        state = decide_node(state, planner=planner, read_manifest=read_manifest)
        if state.decision is not None:
            return state.decision
        if state.pending_call is None:
            break  # LLM neither called a read tool nor decided
        if graph_tools is None:
            break  # asked to read but no read surface available
        state = act_node(state, graph_tools=graph_tools)
    return _exhausted_outcome(state)


__all__ = [
    "PlannerLoopState",
    "PlannerTurn",
    "ReadToolHost",
    "act_node",
    "decide_node",
    "run_planner_loop",
]
