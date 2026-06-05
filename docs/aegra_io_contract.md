# Aegra I/O Contract

## Operation Input

API callers and scripts should describe the operation with target, profile,
mode and goal only:

```json
{
  "target": "10.20.0.0/24",
  "profile_id": "full-vulhub-multihost-pentest",
  "mode": "authorized_blackbox_lab",
  "goal": "perform authorized multi-host assessment under supplied policy"
}
```

The input must not include hidden topology, fixture credentials, internal
markers, fixed attack steps, exploit payloads or MCP tool arguments.

## Final Result Contract

`POST /operations/{operation_id}/run` returns the stable operation result
contract at the top level and under `result`:

```json
{
  "operation_id": "op-xxx",
  "status": "success | failed | partial | blocked",
  "stop_reason": "success_conditions_satisfied | max_cycles | blocked | failed",
  "success": true,
  "success_condition_progress": {},
  "evidence_ids": [],
  "findings_url": "/operations/op-xxx/findings",
  "evidence_url": "/operations/op-xxx/evidence",
  "graph_url": "/operations/op-xxx/graph",
  "audit_url": "/operations/op-xxx/audit-report"
}
```

The compatibility fields `operation`, `policy_summary` and `cycles` may also be
present, but consumers should treat `result` or the same top-level fields as the
final API contract.

## Success Ownership

Operation success is generated only by the orchestrator summary layer:

- `/run` assembles `OperationRunSummary`.
- `lab/scripts/run_autopentest.ps1` and `lab/scripts/run_autopentest.sh` print
  `OperationRunSummary`.
- The summary is assembled from
  `state.execution.metadata["success_condition_progress"]`,
  `operation_status`, the final stop reason, and the findings/evidence/graph/audit
  endpoints.
- MCP tools and GoalAgent must not directly output operation success. They can
  contribute evidence, parsed tool output and goal hints only.

`success=true` requires `operation_status=completed` and
`success_condition_progress.all_required_satisfied=true`.

## Full-Pentest Startup Environment

API and script startup for the full-pentest lab must set:

```sh
AEGRA_MCP_ENABLED=1
AEGRA_MCP_CONFIG_PATH=configs/mcp.lab.full-pentest.json
AEGRA_LAB_PROFILE_PATH=lab/profiles/full_pentest_lab.yml
```

`AEGRA_LAB_FIXTURE_PATH` is a hidden fixture path for the MCP server subprocess
only. It belongs in the MCP server config environment and must not enter Planner
payloads, StageAgent prompts, RuntimeState, KG or AG.

The orchestrator determines whether full-pentest mode is active from
`lab_profile.profile_id == "full-vulhub-multihost-pentest"`. It must not depend
on the main process seeing `AEGRA_MCP_TOOLSET`, because that variable may exist
only in the MCP server subprocess environment.
