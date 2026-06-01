# Aegra Two-Graph Architecture

Aegra's primary architecture is `KG + AG`. The system no longer treats a task
graph as a main runtime graph. `Runtime` and `Policy` are required execution
context, but they are not graphs.

## Canonical Runtime Path

```text
User Goal -> KG/AG/Runtime/Policy -> PlannerAgent -> ResultApplier -> StageDispatcher -> StageAgent -> MCP -> StageResult/ToolTrace -> AttackLogExtractor -> ResultApplier -> KG/AG/Runtime -> Next Cycle
```

This path is the only main operation chain. It replaces the older three-graph
model and removes task-graph scheduling, task-graph merge, and task-graph
lifecycle sync from the default flow.

## Knowledge Graph

`KG` is the Knowledge Graph. It stores environment facts only. Typical nodes
include:

- `Host`
- `Service`
- `Port`
- `VulnerabilityCandidate`
- `ValidatedVulnerability`
- `Credential`
- `Identity`
- `Session`
- `PivotRoute`
- `Evidence`
- `Finding`

`KG` does not store planner steps, agent executions, tool calls, handoffs,
blocked states, or attack-cycle history. Those records belong in `AG`.

## Attack Graph

`AG` is the Attack Graph. It stores attack-process nodes only. Each `AG` node
records one attack process event or decision, such as:

- `PlannerDecision`
- `AgentExecution`
- `ToolCall`
- `StageResult`
- `Handoff`
- `Blocked`
- `GoalCheck`
- `AttackCycle`

`AG` is not a traditional task graph. It does not own the scheduling queue, task
dependencies, worker leases, or task merge lifecycle. It is the structured log
of how the attack process unfolded.

## Visualization And API Publishing

The default graph publishing surface exposes only `KG`, `AG`, and `Runtime`.
`Runtime` is rendered as a state panel/read model, not as a primary graph.
`TG` is not included in visual snapshots or deltas unless an explicit legacy
compatibility flag is enabled. The default is `legacy_tg=false`.

## Runtime And Policy

`Runtime` is execution state, not a graph. It contains operation status,
session manager state, pivot manager state, credential manager state, locks,
leases, budgets, current cycle data, and other transient control state.

`Policy` is constraint context, not a graph. It contains authorization scope,
safety rules, MCP policy enforcement inputs, validation requirements, and other
constraints that govern planner and stage behavior.

## PlannerAgent

`PlannerAgent` is the only global planning LLM. It reads `KG`, `AG`,
`Runtime`, and `Policy`, then emits `PlannerDecision`.

`PlannerDecision` chooses the next stage and `selected_agent`. It must not
contain shell commands, payloads, or concrete tool invocations. It does not
create or update task-graph nodes.

The planner must not impose a fixed `recon -> vuln -> exploit -> pivot -> goal`
sequence. It chooses the next authorized stage from current state, prior
attack-process history, runtime constraints, and policy.

## StageDispatcher And Stage Agents

`StageDispatcher` reads `PlannerDecision.selected_agent` and calls one of five
independent LLM stage agents:

- `ReconAgent`
- `VulnAnalysisAgent`
- `ExploitValidationAgent`
- `AccessPivotAgent`
- `GoalAgent`

Each `StageAgent` can call authorized MCP tools. Stage agents do not write
`KG`, `AG`, `Runtime`, or audit logs. They return `StageResult`, `ToolTrace`,
and optional `handoff_suggestion`.

## AttackLogExtractor

`AttackLogExtractor` extracts attack-process nodes from `StageResult`,
`ToolTrace`, `PlannerDecision`, and audit log entries. It writes nothing
directly. Extracted `AG` candidates are sent to `ResultApplier` for validation
and persistence.

## ResultApplier

`ResultApplier` is the only write boundary for:

- `KG`
- `AG`
- `Runtime`
- audit log

It validates JSON schemas, enforces write rules, normalizes evidence and
findings, merges graph updates, updates runtime managers, records provenance,
and persists audit history. All graph writes must pass through this component.

## Legacy Compatibility

`SchedulerAgent`, `LLMWorkerAgent`, `TaskGraph`, `StageTaskGraphBuilder`,
`TaskGraphBuilder`, task-graph merge helpers, and task-graph lifecycle sync are
legacy compatibility only. They may remain for tests, migrations, or old
interfaces, but they must not appear in the main runtime path.
