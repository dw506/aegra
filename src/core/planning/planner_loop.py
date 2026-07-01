"""LangGraph planner agent loop.

The planner is a real tool-use agent: each turn the LLM either calls a read
tool to inspect the graph or emits a final ``PlannerOutcome``. This module keeps
the three pieces deliberately separate so LangGraph is only the loop driver:

- ``PlannerLoopState``       -> the typed state threaded through the loop
- ``decide_node`` / ``act_node`` -> pure ``State -> State`` node functions
- ``run_planner_loop``      -> a thin LangGraph StateGraph driver

The driver makes no domain decisions; it only routes decide/act until the LLM
returns a decision or the read-step budget is exhausted.
"""

from __future__ import annotations

from typing import Any, Protocol

from langgraph.graph import END, StateGraph
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


def build_planner_graph(
    *,
    planner: PlannerTurn,
    graph_tools: ReadToolHost | None,
    read_manifest: list[dict[str, Any]],
):
    """Compile the planner decide/act loop as a LangGraph StateGraph."""

    graph = StateGraph(PlannerLoopState)

    def _decide(state: PlannerLoopState) -> PlannerLoopState:
        return decide_node(state, planner=planner, read_manifest=read_manifest)

    def _act(state: PlannerLoopState) -> PlannerLoopState:
        if graph_tools is None:
            return state
        return act_node(state, graph_tools=graph_tools)

    def _after_decide(state: PlannerLoopState) -> str:
        if state.decision is not None:
            return END
        if state.pending_call is None:
            return END
        if graph_tools is None:
            return END
        return "act"

    def _after_act(state: PlannerLoopState) -> str:
        if state.decision is not None:
            return END
        if state.step >= state.max_steps:
            return END
        return "decide"

    graph.add_node("decide", _decide)
    graph.add_node("act", _act)
    graph.set_entry_point("decide")
    graph.add_conditional_edges("decide", _after_decide, {"act": "act", END: END})
    graph.add_conditional_edges("act", _after_act, {"decide": "decide", END: END})
    return graph.compile()


def run_planner_loop(
    state: PlannerLoopState,
    *,
    planner: PlannerTurn,
    graph_tools: ReadToolHost | None,
) -> PlannerOutcome:
    """Run the compiled LangGraph planner loop until a decision or exhaustion."""

    read_manifest = graph_tools.read_tool_manifest() if graph_tools is not None else []
    graph = build_planner_graph(planner=planner, graph_tools=graph_tools, read_manifest=read_manifest)
    final_state = PlannerLoopState.model_validate(graph.invoke(state))
    if final_state.decision is not None:
        return final_state.decision
    return _exhausted_outcome(final_state)


__all__ = [
    "PlannerLoopState",
    "PlannerTurn",
    "ReadToolHost",
    "act_node",
    "build_planner_graph",
    "decide_node",
    "run_planner_loop",
]
