# Aegra Graph-Driven Multi-Agent Architecture

Aegra now treats KG, AG, TG and runtime state as the first-class graph layer. This layer does not call LLMs; it only stores, queries and persists structured state. The operation loop is:

1. Load graph memory snapshots: KG facts, AG attack frontier, TG task state and runtime sessions/credentials/pivots.
2. Planner Agent reads the graph context and mission goal, then proposes coarse stage tasks for TG.
3. Stage scheduler selects ready stage tasks from TG dependencies and runtime locks.
4. Dedicated Stage Agents execute bounded ReAct loops through the MCP tool catalog.
5. Stage Agents return structured `StageResult` objects.
6. `StageResultAdapter` and `PhaseTwoResultApplier` deterministically write observations, facts, capabilities, sessions, pivots, task candidates and status transitions back to KG/AG/TG/runtime.
7. Updated graph memory is persisted and snapshotted per cycle.

## Layers

Layer 1 is graph memory, with no LLM decisions in this layer:

- KG: target, service, vulnerability, evidence and capability facts.
- AG: goals, actions, activation status and attack path state.
- TG: stage-level tasks, dependencies, priorities and completion state.
- runtime: operation status, sessions, credentials, pivots, leases, locks, budgets and audit events.

Layer 2 is the mandatory Planner Agent:

- implementation: `MissionPlannerAgent`
- reasoning backend: LLM when configured, deterministic stage planning otherwise
- input: KG/AG/TG/runtime graph context plus policy and mission goal
- output: TG stage tasks and dependencies

Layer 3 is the mandatory execution-agent layer:

- `ReconAgent`
- `VulnAnalysisAgent`
- `ExploitAgent`
- `AccessPivotAgent`
- `GoalAgent`

Each Stage Agent is the execution unit for its stage. When an LLM backend is configured, `LLMStageAdvisor` provides the agent's bounded reasoning over graph/runtime context and the registered MCP tool catalog. Agents do not directly mutate graph stores. They return `StageResult`, and writeback is centralized through the deterministic adapter/applier path.

## Reference Design

The design mirrors the useful split from `nbshenxm/pentest-agent`: planning, reconnaissance and execution are separated into agents with memory. Aegra extends that pattern to multi-host operations by replacing per-script local memory with KG/AG/TG/runtime graph memory and by making stage completion write structured graph updates.

## Legacy Compatibility

The older worker pipeline remains available for tests and compatibility paths, but the application-level `run_operation_cycle` follows the graph-driven stage-agent path by default.
