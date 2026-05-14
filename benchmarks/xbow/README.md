# Aegra XBOW Adapter

This directory contains the local adapter used to call Aegra from the
PentestGPT v1.0 standalone XBOW benchmark runner.

## CLI

```powershell
python D:\Aegra\benchmarks\xbow\run_aegra_xbow.py `
  --target-url http://host.docker.internal:32768 `
  --benchmark-id XBEN-001-24 `
  --timeout 900 `
  --operation-dir D:\Aegra\var\xbow-operations `
  --output-dir D:\Aegra\var\xbow-output
```

Supported arguments:

- `--target-url`: target URL supplied by the benchmark runner.
- `--benchmark-id`: benchmark id, for example `XBEN-001-24`.
- `--timeout`: per-benchmark timeout budget in seconds.
- `--operation-dir`: Aegra file runtime store directory.
- `--output-dir`: exported adapter artifacts directory.
- `--dry-run`: validate and export the planned scope without running Aegra.

## Generate Evaluation Manifest

Generate a schema-compatible manifest for the dynamic target URL supplied by the
XBOW runner:

```powershell
python D:\Aegra\benchmarks\xbow\generate_manifest.py `
  --benchmark-json <xbow-benchmarks>\XBEN-104-24\benchmark.json `
  --target-url http://host.docker.internal:32768 `
  --benchmark-id XBEN-104-24 `
  --flag-format "FLAG{...}" `
  --output D:\Aegra\var\xbow-output\XBEN-104-24.manifest.json
```

The generated manifest uses `aegra-xbow-v1`, maps the dynamic URL into
`scope` and `expected_services`, and declares reachability, vulnerability
validation, and flag extraction subtasks. It does not read `solution`
directories, embed concrete flag values, or include exploit write-ups.

## Runner Contract

The adapter writes PentestGPT-runner-compatible lines to stdout:

```text
[FLAG] FLAG{...}
[DONE] Flags: <n>, Cost: $<cost>, Session: <operation_id>
```

On failure it still emits:

```text
[DONE] Flags: 0, Cost: $0.0000, Session: <operation_id>
```

`Cost` is currently `0.0000` because Aegra does not yet expose benchmark billing
metadata through the local orchestrator.

## Safety Scope

For every invocation the adapter builds a strict per-operation runtime policy:

- `authorized_hosts` contains only the parsed `--target-url` host.
- `cidr_whitelist` is a single-host `/32` or `/128` when the host is an IP. For
  hostnames, the adapter resolves the hostname and whitelists resolved single
  addresses when available.
- The imported target inventory contains only the runner-supplied URL.
- The engagement allows only the target URL, target `host:port`, and target host.
- The risk policy blocks active exploit, destructive behavior, command execution,
  file writes, and reverse callbacks.
- Command execution is limited to declared safe probe tools:
  `nmap`, `httpx`, `whatweb`, `sslscan`, and safe-template `nuclei`.
- Known expansion or exploitation tools such as `masscan`, `sqlmap`, and
  `msfconsole` are disabled.
- The adapter injects a scoped recon worker for benchmark runs. That worker uses
  `nmap` only and forces `-n -Pn -p <target-port>` against the parsed target host,
  so host discovery cannot expand into a broad port scan.

Do not use this adapter against non-benchmark or unauthorized targets.

## Artifacts

Each run exports raw adapter artifacts:

- `audit.json`
- `findings.json`
- `evidence.json`
- `graph.json`
- `state.json`
- `summary.json`

Each benchmark output directory also contains the evaluator-ready files:

- `aegra-audit.json`
- `aegra-findings.json`
- `aegra-graph.json`
- `aegra-state.json`
- `aegra-manifest.json`
- `aegra-report.json`
- `aegra-report.md`
- `aegra-report-no-graph.json`
- `aegra-report-no-graph.md`

The adapter generates these reports even when no flag is found or the operation
fails before full artifact export. In that fallback case empty local artifacts
are written so `benchmarks/evaluate.py` can still report steps, loops,
incomplete commands, human interaction, and graph ablation metrics.

The adapter extracts flags only from exported evidence, findings, and operation
state. It does not read or modify benchmark challenge files or ground truth.
