# Docker Multihost Lab

This lab runs one Aegra control-plane container, three target web-service
containers, and one simulated internal-only service. It is intended for
repeatable local end-to-end smoke testing of the orchestrator flow.

Use this lab only for authorized, local, controlled validation. It is not a
template for unauthorized scanning, destructive exploitation, persistence,
reverse shells, brute force, stealthy execution, or activity outside the Docker
scope and policy files.

## Layout

- `aegra`: FastAPI control API at `http://localhost:8000`.
- `target-web-1`: nginx target, reachable inside Docker as `http://target-web-1:80/`.
- `target-web-2`: httpd target, reachable inside Docker as `http://target-web-2:80/`.
- `target-web-3`: nginx target, reachable inside Docker as `http://target-web-3:80/`.
- `internal-service`: nginx service on the internal-only network, reachable from Aegra as `http://internal-service:80/` and not published to the host.
- `lab_net`: Docker bridge network shared by Aegra and the three target hosts.
- `internal_net`: Docker bridge network marked `internal: true`, shared only by Aegra and `internal-service`.

The Aegra container can enable configured MCP tools through:

```text
AEGRA_MCP_ENABLED=1
```

LLM configuration is injected from `.env`:

```text
AEGRA_LLM_API_KEY=
AEGRA_LLM_BASE_URL=
AEGRA_LLM_MODEL=
AEGRA_LLM_TIMEOUT_SEC=180
```

Do not commit populated `.env` files or real API keys.

The current multihost compose file includes a DMZ network (`10.20.0.0/24`), an
internal network (`10.30.0.0/24`), a pivot SSH host at `10.20.0.30` /
`10.30.0.30`, and an internal web service at `10.30.0.40`.

## Build And Start

```powershell
docker compose build
docker compose up -d
```

Verify the control API:

```powershell
curl --noproxy "*" http://127.0.0.1:8000/health
curl --noproxy "*" http://127.0.0.1:8000/ready
```

## Run Local Regression Tests

From the host workspace:

```powershell
python -m pytest
```

## Run Docker Multihost Smoke Tests

Run the orchestrator smoke test against the nginx target:

```powershell
docker compose run --rm `
  -e AEGRA_RUN_VULHUB_ORCHESTRATOR_SMOKE=1 `
  -e AEGRA_VULHUB_BASE_URL=http://target-web-1:80/ `
  aegra python -m pytest tests/test_vulhub_orchestrator_smoke.py::test_vulhub_orchestrator_cycle_builds_runtime_and_minimal_kg_chain -q
```

Run the same flow against the httpd target:

```powershell
docker compose run --rm `
  -e AEGRA_RUN_VULHUB_ORCHESTRATOR_SMOKE=1 `
  -e AEGRA_VULHUB_BASE_URL=http://target-web-2:80/ `
  aegra python -m pytest tests/test_vulhub_orchestrator_smoke.py::test_vulhub_orchestrator_cycle_builds_runtime_and_minimal_kg_chain -q
```

Run the same flow against the third target:

```powershell
docker compose run --rm `
  -e AEGRA_RUN_VULHUB_ORCHESTRATOR_SMOKE=1 `
  -e AEGRA_VULHUB_BASE_URL=http://target-web-3:80/ `
  aegra python -m pytest tests/test_vulhub_orchestrator_smoke.py::test_vulhub_orchestrator_cycle_builds_runtime_and_minimal_kg_chain -q
```

Run the same flow against the simulated internal service:

```powershell
docker compose run --rm `
  -e AEGRA_RUN_VULHUB_ORCHESTRATOR_SMOKE=1 `
  -e AEGRA_VULHUB_BASE_URL=http://internal-service:80/ `
  aegra python -m pytest tests/test_vulhub_orchestrator_smoke.py::test_vulhub_orchestrator_cycle_builds_runtime_and_minimal_kg_chain -q
```

Or run the full Docker lab smoke sequence:

```powershell
.\scripts\docker_lab_smoke.ps1
```

Pass `-KeepRunning` if you want the containers to remain up after the smoke run:

```powershell
.\scripts\docker_lab_smoke.ps1 -KeepRunning
```

Successful smoke tests prove that planning, execution, feedback, runtime audit,
and KG evidence deltas work against three Docker-hosted targets and one
internal-only Docker service.

Graph-driven smoke tests should verify the Planner-centered loop:
`PlannerAgent` dispatches execution agents, `ResultApplier` records KG/AG/Runtime
facts, `AccessPivotAgent` records sessions and pivot routes, `GoalAgent` records
`goal_satisfied=true`, and only the next `PlannerAgent` `stop_success` completes
the operation. Tests that require this lab must be skipped by default unless
`AEGRA_RUN_DOCKER_MULTIHOST_TEST=1` is set.

## Stop The Lab

```powershell
docker compose down
```
