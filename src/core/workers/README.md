# Worker Boundaries

Legacy worker agents have been removed from the current automation path.
Execution is owned by the five StageAgent implementations and their registered
MCP/tool capabilities.

Reusable low-level helpers such as probe adapters, validators, and tool runners
may remain here, but new execution capabilities should be registered through the
stage/tool capability layer rather than by adding worker-agent protocols.
