# Aegra Runtime Flow

Aegra now uses a two-graph runtime: `KG + AG`. `Runtime` and `Policy` are
execution context, not graphs. The main path must not save, load, schedule, or
merge a `TaskGraph`.

## Main Path

```text
User Goal -> KG/AG/Runtime/Policy -> PlannerAgent -> ResultApplier -> StageDispatcher -> StageAgent -> MCP -> ExecutionResult/ToolTrace -> AttackLogExtractor -> ResultApplier -> KG/AG/Runtime -> Next Cycle
```

`PlannerAgent` is the only global planning LLM. It reads `KG`, `AG`,
`Runtime`, and `Policy`, then emits a `PlannerDecision`. The decision selects
the next stage and `selected_agent`; it does not contain shell commands,
payloads, or concrete tool invocations.

Only `PlannerAgent` may end an operation with `stop_success` or `stop_failed`.
`GoalAgent` can record `runtime_hints.goal_satisfied=true` and supporting
evidence, but the current stage cycle returns to `READY`; the next planner cycle
must verify the recorded `GoalCheck`, evidence refs, and AG process nodes before
emitting `stop_success`.

`ResultApplier` is the only write boundary for `KG`, `AG`, `Runtime`, and the
audit log. Planner decisions, stage results, tool traces, extracted attack
events, evidence, findings, session updates, pivot updates, and credential
updates must pass through this boundary before persistence.

`StageDispatcher` reads `PlannerDecision.selected_agent` and invokes exactly one
of the execution agents:

- `ReconAgent`
- `VulnAnalysisAgent`
- `ExploitValidationAgent`
- `AccessPivotAgent`
- `GoalAgent`

There is no fixed `recon -> vuln -> exploit -> pivot -> goal` order. The
planner may select any authorized stage that is justified by the current
`KG/AG/Runtime/Policy` context.

Each `StageAgent` is an independent LLM execution agent. It receives bounded
context and policy constraints, may call authorized MCP tools, and returns
`ExecutionResult`, `ToolTrace`, and optional `handoff_suggestion`. Stage agents do
not write `KG`, `AG`, `Runtime`, or audit logs directly.

`AttackLogExtractor` reads `ExecutionResult`, `ToolTrace`, `PlannerDecision`, and
audit log entries. It extracts attack-process records for `AG` and sends them
to `ResultApplier`. It does not bypass validation or write directly to graph
stores.

## State Model

- `KG`: Knowledge Graph. Stores environment facts only, such as `Host`,
  `Service`, `Port`, `VulnerabilityCandidate`, `ValidatedVulnerability`,
  `Credential`, `Identity`, `Session`, `PivotRoute`, `Evidence`, and
  `Finding`.
- `AG`: Attack Graph. Stores attack-process nodes only. An `AG` node records
  one attack process event or decision, such as `PlannerDecision`,
  `AgentExecution`, `ToolCall`, `ExecutionResult`, `Handoff`, `Blocked`,
  `GoalCheck`, or `AttackCycle`.
- `Runtime`: execution state, including active operation status, session and
  pivot managers, credential manager state, leases, locks, budgets, and current
  cycle metadata. It is not a graph and is exposed to UI consumers as a state
  panel/read model.
- `Policy`: authorization, scope, safety, MCP enforcement, and validation
  constraints. It is not a graph.

## Boundaries

- Agent: LLM-owned reasoning module. The main path agents are `PlannerAgent`
  and the five `StageAgent` implementations.
- Dispatcher: deterministic routing from `PlannerDecision.selected_agent` to a
  execution agent. It does not plan, schedule tasks, or write state.
- MCP: tool capability layer. MCP tools execute authorized actions and return
  neutral tool results. They do not write graphs or runtime state.
- Extractor: deterministic attack-log extraction from planner, stage, tool, and
  audit records into candidate `AG` process nodes.
- Applier: deterministic write boundary. `ResultApplier` validates schemas,
  applies policy-compatible updates, normalizes evidence and findings, merges
  graph facts, updates runtime managers, and records audit history.

## Guardrails

- `TaskGraph` is not a primary graph and must not appear in the main runtime
  path.
- Visualization and graph API defaults publish only `KG`, `AG`, and `Runtime`;
  legacy TG output requires explicit `legacy_tg=true`.
- `SchedulerAgent`, `LLMWorkerAgent`, `TaskGraph`, `StageTaskGraphBuilder`,
  `TaskGraphBuilder`, task-graph merge helpers, and task-graph lifecycle sync
  are legacy compatibility only.
- Legacy compatibility code must not be used as the default planner,
  dispatcher, scheduling, merge, or persistence path for new operation cycles.
- Handoff suggestions are recorded as AG/runtime facts only. They do not trigger
  another stage unless the next `PlannerAgent` decision accepts them.
- JSON schema validation, MCP policy enforcement, Runtime session, pivot, and
  credential managers, audit logging, and evidence/finding normalization remain
  mandatory.
