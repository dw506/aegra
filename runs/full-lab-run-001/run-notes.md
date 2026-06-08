# Aegra full lab run notes

- Operation ID: `full-lab-run-001`
- Base URL: `http://127.0.0.1:8001`
- Target input: `10.20.0.0/24`
- Command requested:

```powershell
.\lab\scripts\run_autopentest.ps1 `
  -BaseUrl http://127.0.0.1:8001 `
  -OperationId full-lab-run-001 `
  -Target 10.20.0.0/24 `
  -Run
```

## Timeline

- 2026-06-05 21:36:23 +08:00: started `run_autopentest.ps1`.
- 2026-06-05 21:36:24 +08:00: operation created, full lab profile/policy loaded, target `10.20.0.0/24` imported.
- 2026-06-05 21:36:24 +08:00: first control cycle started.
- 2026-06-05 21:37:25 +08:00: initial run failed during planner LLM call with `httpx.RemoteProtocolError: Server disconnected without sending a response`.
- 2026-06-05 21:38:00 +08:00: operation recovered with reason `retry_after_llm_remote_protocol_error`.
- 2026-06-05 21:38:00 +08:00: manual `/run` retry started.
- 2026-06-05 21:40:02 +08:00: cycle 1 completed with `RECON_STAGE` partial result after reaching `max_steps=3`.
- 2026-06-05 21:40:44 +08:00: cycle 2 started, but did not complete. Operation currently remains `running` with `last_cycle_phase=cycle_started` and `unclean_shutdown=true`.

## Loaded Lab Context

- Lab profile: `/app/lab/profiles/full_pentest_lab.yml`
- Policy: `/app/lab/scope/docker_lab.policy.json`
- Full lab activation: `true`
- Entry scope: `10.20.0.0/24`
- Internal pivot scope is present in policy (`10.30.0.0/24`) but should only be used after authorized pivot/capability.

## Completed Recon Result

Nmap scanned `10.20.0.0/24` and reported 256 addresses scanned, 6 hosts up:

- `10.20.0.1`: `111/tcp open rpcbind 2-4 (RPC #100000)`, `8000/tcp filtered http-alt`
- `10.20.0.20`: `80/tcp open http Apache httpd 2.4.25 ((Debian))`
- `10.20.0.21`: `3000/tcp open ppp?`; HTTP fingerprint shows OWASP Juice Shop
- `10.20.0.22`: `8080/tcp open http Jetty 9.2.11.v20150529`
- `10.20.0.30`: `22/tcp open ssh OpenSSH 9.7`
- `10.20.0.10`: `8000/tcp open http Uvicorn`

No findings were generated before the run stalled.

## Exported Artifacts

- `runs/full-lab-run-001/run_autopentest.log`
- `runs/full-lab-run-001/run_autopentest.err.log`
- `runs/full-lab-run-001/run_autopentest.exit.txt`
- `runs/full-lab-run-001/manual-retry.log`
- `runs/full-lab-run-001/recover-after-llm-error.json`
- `runs/full-lab-run-001/summary-final.json`
- `runs/full-lab-run-001/evidence-final.json`
- `runs/full-lab-run-001/findings-final.json`
- `runs/full-lab-run-001/audit-report-final.json`
- `runs/full-lab-run-001/report-final.md`

## Current Blocking Point

The requested run did not complete the full automated penetration-test flow. It successfully reached and recorded the first recon cycle, but the control loop did not advance past cycle 2. The first failure was a transient upstream LLM transport error; after recovery, recon succeeded, but the operation remains in a running/unclean checkpoint state.
