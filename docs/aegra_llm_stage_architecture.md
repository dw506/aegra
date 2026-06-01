# Aegra LLM Stage Architecture

The LLM architecture is built around one global planning agent and five
independent execution agents. Aegra does not use a task graph as the main
execution chain.

Canonical main path:

```text
User Goal -> KG/AG/Runtime/Policy -> PlannerAgent -> ResultApplier -> StageDispatcher -> StageAgent -> MCP -> StageResult/ToolTrace -> AttackLogExtractor -> ResultApplier -> KG/AG/Runtime -> Next Cycle
```

```text
KG/AG/Runtime/Policy
        |
        v
PlannerAgent
        |
        v
PlannerDecision
        |
        v
ResultApplier
        |
        v
StageDispatcher
        |
        v
ReconAgent / VulnAnalysisAgent / ExploitValidationAgent / AccessPivotAgent / GoalAgent
        |
        v
MCP
        |
        v
StageResult / ToolTrace / handoff_suggestion
        |
        v
AttackLogExtractor
        |
        v
ResultApplier
        |
        v
KG/AG/Runtime
        |
        v
Next Cycle
```

## PlannerAgent

`PlannerAgent` is the only global planning LLM. It reads:

- `KG`: environment facts.
- `AG`: prior attack-process events and decisions.
- `Runtime`: current execution state.
- `Policy`: scope, authorization, safety, and MCP constraints.

It outputs `PlannerDecision`, which may include:

- selected stage
- `selected_agent`
- rationale
- required context
- expected result shape
- stop, handoff, or blocked state

It must not output shell commands, payloads, or concrete MCP tool calls. It also
must not rely on a fixed stage order. The next agent is selected from current
state and policy, not from a hard-coded sequence.

## StageDispatcher

`StageDispatcher` is deterministic routing. It reads
`PlannerDecision.selected_agent` and invokes one of:

- `ReconAgent`
- `VulnAnalysisAgent`
- `ExploitValidationAgent`
- `AccessPivotAgent`
- `GoalAgent`

It does not schedule task-graph nodes, merge task state, or write graphs.

## StageAgent

Each `StageAgent` is an independent LLM stage executor. A stage agent can call
authorized MCP tools through the policy-enforced tool boundary. It returns:

- `StageResult`
- `ToolTrace`
- `handoff_suggestion`

Stage agents cannot directly write `KG`, `AG`, `Runtime`, or audit logs. Their
outputs are validated and applied later by `ResultApplier`.

## MCP Layer

MCP tools provide capability only. They return tool results, parsed fields,
errors, and evidence references. Policy enforcement happens before tool calls,
and denied calls are recorded as blocked tool traces. MCP tools do not persist
graph or runtime state.

## AttackLogExtractor

`AttackLogExtractor` converts operational records into `AG` attack-process
nodes. Its inputs are:

- `StageResult`
- `ToolTrace`
- `PlannerDecision`
- audit log entries

The extractor emits candidate `AG` nodes such as `PlannerDecision`,
`AgentExecution`, `ToolCall`, `StageResult`, `Handoff`, `Blocked`,
`GoalCheck`, and `AttackCycle`. Each `AG` node records one attack process event
or decision, not an environment fact.

## ResultApplier

`ResultApplier` is the only write boundary. It validates and persists:

- `KG` environment facts
- `AG` attack-process nodes
- `Runtime` session, pivot, credential, lock, and budget updates
- audit log entries
- normalized evidence and findings

The applier performs schema validation, merge rules, provenance tracking,
confidence handling, and policy-compatible writeback. It does not call an LLM
for core judgment.

## Legacy Compatibility

`SchedulerAgent`, `LLMWorkerAgent`, `TaskGraph`, `StageTaskGraphBuilder`,
`TaskGraphBuilder`, task-graph merge helpers, and task-graph lifecycle sync may
remain only as legacy compatibility. They are not part of the main LLM stage
architecture and must not be used to drive new operation cycles.
