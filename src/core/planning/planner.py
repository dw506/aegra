"""The planner agent: a bounded LLM tool-use loop over graph-read tools.

`Planner.decide` is the orchestrator entry point — it builds the loop state and
drives `planner_loop.run_planner_loop`, which calls back into `Planner.run_turn`
for each LLM turn. Each turn the LLM either calls a read tool (to inspect the
graph) or emits a final `PlannerOutcome`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.agents.packy_llm import PackyLLMClient, PackyLLMError
from src.core.planning.models import PlannerOutcome
from src.core.planning.planner_loop import PlannerLoopState, run_planner_loop


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PLANNER_GLOBAL_CONTROL_PROMPT = PROMPT_DIR / "planner_global_control.md"


class PlannerConfig(BaseModel):
    """Prompt limits, model options, and read-step budget for the planner."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str | None = None
    max_context_chars: int = Field(default=24000, ge=4000)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_steps: int = Field(default=6, ge=1)


class Planner:
    """Agentic planner driven by an LLM tool-use loop.

    With no LLM client it degrades to a replan outcome (no hard-coded fallback
    plan). The decide/act loop lives in ``planner_loop``; this class supplies the
    orchestrator entry (``decide``) and the per-turn LLM call (``run_turn``).
    """

    def __init__(
        self,
        *,
        client: PackyLLMClient | None = None,
        config: PlannerConfig | None = None,
    ) -> None:
        self._client = client
        self._config = config or PlannerConfig(
            model=getattr(getattr(client, "config", None), "model", None)
        )

    # ------------------------------------------------------------------
    # Orchestrator entry
    # ------------------------------------------------------------------

    def decide(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any] | None = None,
        recent_execution_results: list[dict[str, Any]] | None = None,
        graph_tools: Any | None = None,
    ) -> PlannerOutcome:
        """Run the planner loop and return one PlannerOutcome for this cycle."""

        operation_id = str(graph_context.get("operation_id") or "operation")
        cycle_index = int(
            graph_context.get("cycle_index")
            or graph_context.get("runtime", {}).get("cycle_index")
            or 0
        )
        if self._client is None:
            return PlannerOutcome(
                operation_id=operation_id,
                cycle_index=cycle_index,
                action="replan",
                reason="Planner requires an LLM client; no hard-coded fallback is available.",
                stop_condition="planner_llm_unavailable",
                confidence=0.0,
                metadata={"planner": "llm_required", "accepted": False},
            )

        seed = dict(graph_context)
        if graph_tools is not None:
            seed.setdefault("min_summary", graph_tools.build_min_summary())
            seed["graph_tools"] = graph_tools.tool_manifest()

        state = PlannerLoopState(
            operation_id=operation_id,
            cycle_index=cycle_index,
            goal=goal,
            policy_context=dict(policy_context or {}),
            seed_context=seed,
            recent_execution_results=list(recent_execution_results or []),
            max_steps=self._config.max_steps,
        )
        outcome = run_planner_loop(state, planner=self, graph_tools=graph_tools)
        outcome = _prefer_real_exploit_tools(outcome, seed.get("mcp_tool_catalog") or {})

        # Apply advisory write-tool calls the final turn emitted (record_finding,
        # record_attack_step, link_evidence). Machine facts are written elsewhere.
        if graph_tools is not None and isinstance(outcome.metadata, dict):
            write_calls = outcome.metadata.get("planner_tool_calls")
            if write_calls:
                results = graph_tools.apply_tool_calls(write_calls)
                if results:
                    outcome.metadata = {**dict(outcome.metadata), "planner_graph_tool_results": results}
        if outcome.operation_id == "operation":
            outcome.operation_id = operation_id
        return outcome

    # ------------------------------------------------------------------
    # One LLM turn (called by the loop's decide node)
    # ------------------------------------------------------------------

    def run_turn(
        self,
        *,
        state: PlannerLoopState,
        read_manifest: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run one planner LLM turn.

        Returns ``{"tool_call": {...}}`` to inspect the graph, or
        ``{"outcome": PlannerOutcome, "raw": {...}}`` for a final decision (or a
        fallback outcome on transport/parse/schema failure).
        """

        prompt = self._build_turn_prompt(state, read_manifest)
        try:
            response = self._client.complete_chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                model=self._config.model,
                temperature=self._config.temperature,
            )
        except PackyLLMError as exc:
            return {
                "outcome": _fallback_outcome(
                    operation_id=state.operation_id,
                    cycle_index=state.cycle_index,
                    reason="planner llm unavailable",
                    stop_condition="planner_llm_unavailable",
                    metadata={"planner": "llm", "accepted": False, "error": str(exc)},
                )
            }

        payload = _extract_json_object(response.text)
        if payload is None:
            return {
                "outcome": _fallback_outcome(
                    operation_id=state.operation_id,
                    cycle_index=state.cycle_index,
                    reason="planner llm returned non-json",
                    stop_condition="invalid_planner_json",
                    metadata={"planner": "llm", "accepted": False, "raw_text": response.text[:1000]},
                )
            }

        # A turn is either a read tool call or a final PlannerOutcome.
        tool_call = payload.get("tool_call")
        if isinstance(tool_call, dict):
            return {"tool_call": tool_call}

        payload.setdefault("operation_id", state.operation_id)
        payload.setdefault("cycle_index", state.cycle_index)
        payload.setdefault("metadata", {})
        payload["metadata"] = {
            **dict(payload["metadata"]),
            "planner": "llm",
            "accepted": True,
            "model": response.model,
            "usage": response.usage,
            "read_steps": len(state.read_log),
        }
        # Carry advisory write-tool calls through to decide() via metadata.
        if isinstance(payload.get("planner_tool_calls"), list):
            payload["metadata"]["planner_tool_calls"] = payload["planner_tool_calls"]
        try:
            outcome = PlannerOutcome.model_validate(payload)
        except ValidationError as exc:
            return {
                "outcome": _fallback_outcome(
                    operation_id=state.operation_id,
                    cycle_index=state.cycle_index,
                    reason="planner llm returned invalid PlannerOutcome schema",
                    stop_condition="invalid_planner_schema",
                    metadata={
                        "planner": "llm",
                        "accepted": False,
                        "error": str(exc),
                        "raw_text": response.text[:2000],
                        "normalized_payload": payload,
                    },
                )
            }
        return {"outcome": outcome, "raw": payload}

    def _build_turn_prompt(
        self,
        state: PlannerLoopState,
        read_manifest: list[dict[str, Any]],
    ) -> str:
        outcome_contract = {
            "operation_id": "operation id string",
            "cycle_index": 0,
            "action": "execute | replan | pause_for_review | stop_success | stop_failed",
            "directive": {
                "operation_id": "operation id string",
                "cycle_index": 0,
                "capability": "recon | analysis | exploit | pivot | lateral | goal | evidence",
                "objective": "bounded one-round objective",
                "target_refs": [{"graph": "kg", "ref_id": "node id copied from min_summary", "ref_type": "Host"}],
                "allowed_tools": [],
                "tool_hints": [],
                "max_tools": 16,
                "success_hint": "what is enough for this round",
                "required_context": {},
                "risk_level": "low | medium | high | critical",
            },
            "reason": "short summary, no chain of thought",
            "stop_condition": "contract_satisfied | failure reason | null",
            "confidence": 0.8,
            "metadata": {},
        }
        seed = state.seed_context
        context = _truncate_json(
            {
                "mission_goal": state.goal,
                "operation_id": state.operation_id,
                "cycle_index": state.cycle_index,
                "policy": state.policy_context,
                "min_summary": seed.get("min_summary") or {},
                "success_condition_progress": seed.get("success_condition_progress") or {},
                "graph_tools": seed.get("graph_tools") or {},
                "agent_capabilities": seed.get("agent_capabilities") or [],
                "mcp_tool_catalog": _slim_tool_catalog(seed.get("mcp_tool_catalog") or {}),
                "recent_results": list(state.recent_execution_results or []),
                "read_tools": read_manifest,
                "read_log": state.read_log,
                "read_budget": {"used": state.step, "max": state.max_steps},
                "planner_outcome_contract": outcome_contract,
            },
            self._config.max_context_chars,
        )
        return (
            "You are an agentic planner. Each turn, return strict JSON that is EITHER "
            "(a) a read tool call: {\"tool_call\": {\"tool\": \"<read tool name>\", \"arguments\": {...}}} "
            "to inspect the graph before deciding, OR (b) a final PlannerOutcome object. "
            "The min_summary/success_condition_progress are a starting seed; use read_tools to drill into "
            "specific nodes/edges/runtime state when the seed is insufficient. You have a bounded read budget "
            "(read_budget); prior read results are in read_log. When you have enough, return a final PlannerOutcome. "
            "AG is a result timeline: one ATTACK_STEP per execution round plus terminal outcomes. "
            "Do not output shell commands. Do not output MCP tool arguments. "
            "Do not micro-control tool calls; output one RoundDirective with a capability and bounded objective. "
            "ExecutionAgent may autonomously choose allowed tools, including run_command, inside authorized scope. "
            "PlannerAgent is the only global controller that may output stop_success or stop_failed. "
            "Do not use a fixed stage sequence and do not require every capability to run. "
            "When choosing an exploit round and ToolCatalog contains metasploit_exec, put metasploit_exec "
            "first in directive.allowed_tools/tool_hints for real session proof. "
            "Select the next capability from success_condition_progress.missing, min_summary, Policy, "
            "recent results, and ToolCatalog. "
            "Treat recent ExecutionResult control hints as hard constraints: if a recent result contains "
            "next_step_guidance or capability guidance, "
            "follow it unless it conflicts with Policy or newer evidence. If a recent result contains "
            "supported_bounded_validation_candidate=false, do not choose capability=exploit for that target. "
            "target_refs MUST be a list of objects {\"graph\":\"kg\"|\"ag\", \"ref_id\":\"<exact node id from min_summary>\"}, "
            "never bare id strings; use [] when no specific node applies. Plain ids may go in required_context. "
            "You may emit planner_tool_calls ONLY for write-level judgment records "
            "(record_finding/record_attack_step/link_evidence); these are advisory and do not change this "
            "turn's decision. Machine facts from tools are written deterministically after execution. "
            "If evidence is insufficient, choose action=execute with an appropriate capability or choose replan. "
            "If policy does not allow the next action, choose pause_for_review. "
            "Never select a target whose host appears in policy.blocked_hosts; treat those hosts as strictly out of scope "
            "(they are control-plane infrastructure, not assessment targets). "
            "A blocked_host / target_out_of_scope rejection means ONLY that one host is off-limits, NOT that the chain is "
            "dead: retarget the in-scope assessment hosts (e.g. the discovered entry-zone services) and keep executing. "
            "Do NOT pause_for_review merely because a call against a blocked host was rejected — pause only when no in-scope "
            "target or action remains. "
            "If success_condition_progress.eligible_for_stop=true, choose action=stop_success with stop_condition=contract_satisfied. "
            "For action=execute, directive must be non-null. For stop/replan/pause actions, directive must be null. "
            "Use reason for a concise justification without chain-of-thought.\n\n"
            f"{context}"
        )


def _load_planner_global_control_prompt() -> str:
    return PLANNER_GLOBAL_CONTROL_PROMPT.read_text(encoding="utf-8")


SYSTEM_PROMPT = _load_planner_global_control_prompt()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(1))
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _slim_tool_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    """Project the MCP catalog to tool names + short descriptions for the planner.

    The planner only selects a capability and is told never to emit tool
    arguments, so the per-tool inputSchemas (the bulk of the catalog, ~3.6k
    tokens) are dead weight in its prompt. Keep name/description/availability and
    drop the schemas; the executor still receives the full catalog.
    """

    if not isinstance(catalog, dict):
        return {}
    slim: dict[str, Any] = {}
    for server_id, server in catalog.items():
        if not isinstance(server, dict):
            continue
        slim_tools: list[dict[str, Any]] = []
        for tool in server.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            entry: dict[str, Any] = {"name": tool.get("name")}
            if tool.get("description"):
                entry["description"] = tool.get("description")
            if tool.get("available") is False:
                entry["available"] = False
            slim_tools.append(entry)
        slim[server_id] = {"tools": slim_tools}
    return slim


def _prefer_real_exploit_tools(outcome: PlannerOutcome, catalog: dict[str, Any]) -> PlannerOutcome:
    """Bias exploit rounds toward real session tooling while preserving the LLM decision."""

    directive = outcome.directive
    if outcome.action != "execute" or directive is None or directive.capability != "exploit":
        return outcome
    if not _catalog_has_tool(catalog, "metasploit_exec"):
        return outcome
    allowed = list(directive.allowed_tools)
    for tool in ("metasploit_exec", "session_exec"):
        if _catalog_has_tool(catalog, tool) and tool not in allowed:
            allowed.insert(0 if tool == "metasploit_exec" else len(allowed), tool)
    hints = list(directive.tool_hints)
    hints.append(
        {
            "preferred_tool": "metasploit_exec",
            "purpose": "real exploit/session proof",
            "no_session_policy": "return evidence and replan/tune exploit parameters; do not fabricate success",
        }
    )
    outcome.directive = directive.model_copy(update={"allowed_tools": allowed, "tool_hints": hints})
    return outcome


def _catalog_has_tool(catalog: dict[str, Any], tool_name: str) -> bool:
    if not isinstance(catalog, dict):
        return False
    for server in catalog.values():
        if not isinstance(server, dict):
            continue
        for tool in server.get("tools") or []:
            if isinstance(tool, dict) and tool.get("name") == tool_name and tool.get("available") is not False:
                return True
    return False


def _truncate_json(payload: dict[str, Any], max_chars: int) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80] + "...<truncated>"


def _fallback_outcome(
    *,
    operation_id: str,
    cycle_index: int,
    reason: str,
    stop_condition: str | None,
    metadata: dict[str, Any],
) -> PlannerOutcome:
    return PlannerOutcome(
        operation_id=operation_id,
        cycle_index=cycle_index,
        action="replan",
        directive=None,
        reason=reason,
        stop_condition=stop_condition,
        confidence=0.0,
        metadata=metadata,
    )


__all__ = ["Planner", "PlannerConfig"]
