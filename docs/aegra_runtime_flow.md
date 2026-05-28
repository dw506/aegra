# Aegra Runtime Flow

This document summarizes the current execution path and ownership boundaries for
Aegra's core runtime loop.

## Main Path

```text
KG / AG / TG
  -> Scheduler
  -> WorkerRegistry
  -> Primary Worker
  -> ValidationService
  -> ExecutionExecutor
  -> ExecutionAdapter
  -> ToolExecutionResult
  -> PerceptionAgent
  -> ParserRegistry
  -> ResultApplier
  -> Runtime audit
```

The current smoke path exercises this chain with access validation:

```text
AccessValidationWorker
  -> AccessValidationService
  -> ExecutionExecutor(LocalShellAdapter)
  -> ToolExecutionResult
  -> PerceptionAgent / ToolExecutionParser
  -> AgentTaskResult
  -> PhaseTwoResultApplier
  -> audit_log.tool_execution_recorded
```

Planner and scheduler decide what should run. Worker selection goes through
`WorkerRegistry.default()`, which registers primary `BaseWorkerAgent`
implementations. The selected worker delegates business behavior to a validation
service. If the service needs a tool, it builds a `ToolPlan` and passes it to
`ExecutionExecutor`, which selects an execution adapter.

Execution adapters return adapter-neutral `ToolExecutionResult` objects. These
results are carried in worker outcomes and evidence metadata so the perception
layer can parse them into observations and evidence. `PhaseTwoResultApplier`
then applies the canonical worker result through runtime, KG, AG, and TG owners
and records runtime audit entries such as `tool_execution_recorded` or
`tool_execution_failed`.

## Ownership Boundaries

- Legacy workers are compatibility wrappers. Primary worker selection uses
  `BaseWorkerAgent` implementations through `WorkerRegistry.default()`.
- Parsers interpret raw results into observation/evidence records. Parsers do
  not write Runtime State, KG, AG, or TG.
- Execution adapters execute `ToolPlan` objects and return
  `ToolExecutionResult`. They do not write KG, AG, TG, or Runtime audit.
- `ResultApplier` is the boundary that turns accepted worker results into
  runtime side effects, KG deltas, AG projection requests, TG lifecycle updates,
  and audit log entries.
- `ToolExecutionParser` belongs to core perception because it handles the
  adapter-neutral result shape.

## Current Guardrails

- `src.core.planner` is the deterministic planning kernel and must not import
  agent wrappers.
- `src.core.perception` must not depend on external control-channel integrations.
- Worker services must not depend on external control-channel clients.
- Execution adapters must not import ResultApplier, graph stores, or KG/AG/TG
  mutation structures.
- Tool execution audit is written by `PhaseTwoResultApplier`, not by parsers or
  adapters.
