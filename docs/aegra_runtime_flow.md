# Aegra Runtime Flow

Aegra uses an LLM Multi-Agent Graph-Driven Runtime. Agent means an
LLM-owned reasoning module; deterministic code is named as a service, gate,
validator, applier, store, parser, or adapter.

## Main Path

```text
User Goal / Operation Goal
  -> KG / AG / TG / Runtime / Policy
  -> PlannerAgent [LLM]
  -> ResultApplier
  -> SchedulerAgent [LLM]
  -> ResultApplier
  -> LLMWorkerAgent [LLM]
  -> MCP Tool / ExecutionAdapter
  -> CriticAgent [LLM]
  -> ResultApplier
  -> KG / AG / TG / Runtime update
```

`PlannerAgent` reads graph state and policy context, understands the mission
goal, and emits `PlannerResult` / graph update intents. It does not execute
tools and does not write KG, AG, TG, or Runtime directly.

`SchedulerAgent` reads TG candidates, Runtime constraints, Policy, worker state,
tool catalog, and recent outcomes. Candidate collection and constraint summaries
are deterministic services, but the final `ScheduleDecision` is LLM-owned. If
the scheduler LLM is unavailable, the agent returns blocked/unavailable instead
of falling back to deterministic dispatch.

`LLMWorkerAgent` receives one `ScheduledTask`, compressed context, policy
constraints, and the tool catalog. It may only choose `call_mcp_tool`, `defer`,
or `failed`; it does not create TG tasks or write graph/runtime state.

`CriticAgent` reviews worker evidence, tool traces, and success criteria. It
emits confidence, evidence quality, and retry/replan/change-tool suggestions.
It does not block writeback and does not write graph/runtime state.

## Boundaries

- Agent: LLM-owned reasoning module. Current primary agents are
  `PlannerAgent`, `SchedulerAgent`, `LLMWorkerAgent`, and `CriticAgent`.
- Service: deterministic infrastructure or helper logic, such as
  `CandidateTaskService`, runtime constraint summarizers, and policy helpers.
- Applier: write boundary. `ResultApplier` is the only component that persists
  KG, AG, TG, Runtime, audit log, and LLM decision history changes.
- Adapter: tool execution boundary. Execution adapters and MCP clients execute
  tools and return neutral results.
- Parser: raw result interpretation boundary. Parsers normalize output into
  observations and evidence, but do not persist graph/runtime state.

## Guardrails

- Deterministic helpers may provide candidate lists and constraints, but must
  not be registered or named as Agents unless the core decision is LLM-owned.
- Result writeback remains centralized in `PhaseTwoResultApplier`.
- Execution adapters and parsers do not import graph stores or ResultApplier.
- LLM unavailable states are explicit blocked/unavailable decisions, not hidden
  rule-based fallbacks.
