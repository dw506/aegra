# Aegra Graph-Driven Multi-Agent Architecture

Aegra's graph-driven runtime is now the two-graph architecture described in
`docs/aegra_two_graph_architecture.md`.

The primary graph layer is:

- `KG`: environment facts only.
- `AG`: attack-process nodes only.

`Runtime` and `Policy` are required context for execution and constraints, but
they are not graphs.

## Operation Loop

```text
User Goal -> KG/AG/Runtime/Policy -> PlannerAgent -> ResultApplier -> StageDispatcher -> StageAgent -> MCP -> ExecutionResult/ToolTrace -> AttackLogExtractor -> ResultApplier -> KG/AG/Runtime -> Next Cycle
```

The loop has no task-graph scheduling or task-graph merge step. `PlannerAgent`
is the only global planning LLM and outputs `PlannerDecision`, which selects the
next execution agent without producing commands or payloads.

`StageDispatcher` invokes exactly one of the five execution agents selected by the
planner:

- `ReconAgent`
- `VulnAnalysisAgent`
- `ExploitValidationAgent`
- `AccessPivotAgent`
- `GoalAgent`

Stage agents call authorized MCP tools and return `ExecutionResult`, `ToolTrace`,
and optional `handoff_suggestion`. They do not write graphs or runtime state.

## Write Boundary

`ResultApplier` is the only persistence boundary for `KG`, `AG`, `Runtime`, and
audit log changes. `AttackLogExtractor` extracts `AG` attack-process nodes from
planner, stage, tool, and audit records, then sends candidates through
`ResultApplier`.

## Legacy Compatibility

`SchedulerAgent`, `LLMWorkerAgent`, `TaskGraph`, `StageTaskGraphBuilder`,
`TaskGraphBuilder`, task-graph merge helpers, and task-graph lifecycle sync are
legacy compatibility only. They must not drive the main operation loop.
