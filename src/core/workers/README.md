# Worker Boundaries

`BaseWorkerAgent` is the primary worker protocol for the current execution
pipeline. New worker entrypoints should implement this protocol and return
`AgentOutput`.

`BaseWorker` is the legacy compatibility protocol. It remains available for
old `AgentTaskRequest` / `AgentTaskResult` callers, tests, and adapters that
have not migrated yet.

Primary workers only translate the `BaseWorkerAgent` protocol into domain
service calls. They should not own validation business logic.

Services own the single source of business logic for each worker domain. Keep
access, goal, and privilege validation behavior in `services/` so the primary
and legacy protocols stay consistent.

Legacy workers only wrap the old protocol around the same services. Do not add
new business logic to legacy worker modules.
