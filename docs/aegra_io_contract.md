# Aegra I/O Contract

## Operation Input

API callers and scripts should submit one operation request. The API creates
the operation, imports targets, starts the runtime and, by default, runs the
bounded operation loop:

```http
POST /operations
```

```json
{
  "operation_id": "op-xxx",
  "metadata": {
    "operation_input": {
      "target": "10.20.0.0/24",
      "profile_id": "full-vulhub-multihost-pentest",
      "mode": "authorized_blackbox_lab",
      "goal": "perform authorized multi-host assessment under supplied policy"
    }
  },
  "targets": [
    {"address": "10.20.0.0/24", "kind": "cidr", "tags": ["lab", "authorized"]}
  ],
  "max_cycles": 5
}
```

The user-facing operation input should describe target, profile, mode and goal
only:

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

`POST /operations` returns the stable operation result contract at the top level
and under `result` when targets are provided and `run=true`. The compatibility
endpoint `POST /operations/{operation_id}/run` returns the same shape for
already-created operations:

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
AEGRA_LAB_PROFILE_PATH=lab/profiles/full_pentest_lab.yml
```

The orchestrator determines whether full-pentest mode is active from
`lab_profile.profile_id == "full-vulhub-multihost-pentest"`.

The `run_autopentest` scripts are HTTP clients, so environment variables set
inside those scripts do not reconfigure an already-running API server. After the
unified operation request, the scripts read `/operations/{operation_id}/summary`
and require `metadata.lab_activation.full_pentest_active == true`; otherwise
they fail with a message telling the user to start the API server with
`AEGRA_LAB_PROFILE_PATH`.
