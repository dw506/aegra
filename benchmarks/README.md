# Aegra Benchmarks

This directory contains the Phase 8 benchmark and PentestGPT / AutoPentester comparison infrastructure. It is an offline scoring layer: it reads manifest, audit, findings, evidence, graph, and runtime-state JSON exports, then compares them with declared ground truth.

## Safety and Authorization

Only run benchmarks against systems you own or are explicitly authorized to test. Every manifest target must declare:

- `scope.authorized_hosts`
- `scope.cidr_whitelist`
- `scope.target_url`
- `allowed_tools`
- `risk_level`

The evaluator does not scan, exploit, or contact targets. It only parses local JSON files.

## Manifests

- `vulnhub_local.json`: first concrete local Vulhub Struts2 S2-045 baseline.
- `autopentester_htb10.json`: schema-compatible HTB10 skeleton for later authorized experiments. It intentionally does not include sensitive machine write-up details.
- `custom_vm.json`: template for owned or authorized custom VM targets.
- `manifest.schema.json`: JSON Schema reference for the manifest format.

## Prepare a Local Vulhub Struts2 Target

Use only a local, controlled Vulhub checkout. A typical setup is:

```powershell
cd <your-vulhub-checkout>\struts2\s2-045
docker compose up -d
```

Confirm that the service is mapped only to the intended local address, for example `http://127.0.0.1:8080/`, then keep the manifest scope aligned with that mapping. If your port or host differs, copy `vulnhub_local.json` and change `scope`, `expected_services`, and `allowed_tools` before running Aegra.

## Run Aegra and Export Artifacts

Run the operation with the local target and policy scope matching the manifest. Export or collect these files from the operation store:

- Audit report: `/operations/{id}/audit-report` or `audit.json`
- Findings report: `/operations/{id}/findings` or report JSON
- Evidence: `/operations/{id}/evidence` or embedded `evidence` in the report JSON
- Graph: `/operations/{id}/graph`, with `nodes` and `edges`
- Runtime state if available: `state.json`

The first evaluator version recognizes the current Aegra graph pattern:

```text
Host -> Service -> Vulnerability -> Evidence
```

It also estimates:

- `steps` from `control_cycle_history`, `audit_log`, and `operation_log`
- `loops` from repeated `task_id`, command summaries, or repeated phase/action pairs
- `human_interaction` from `approval_decision`, `waiting_approval`, and `manual_*` audit events
- `incomplete_commands` from failed tool output and known incomplete-command markers

## Run Evaluation

Evaluate an operation directory:

```powershell
python benchmarks/evaluate.py --manifest benchmarks/vulnhub_local.json --operation-dir <runtime-store\op-id> --target-id vulhub-struts2-s2045-local
```

Evaluate explicit exports:

```powershell
python benchmarks/evaluate.py --manifest benchmarks/vulnhub_local.json --audit tests/fixtures/benchmark/audit.json --findings tests/fixtures/benchmark/findings.json --graph tests/fixtures/benchmark/graph.json
```

Write reports:

```powershell
python benchmarks/evaluate.py --manifest benchmarks/vulnhub_local.json --audit tests/fixtures/benchmark/audit.json --findings tests/fixtures/benchmark/findings.json --graph tests/fixtures/benchmark/graph.json --output-json tmp-report.json --output-md tmp-report.md
```

Run the no-graph ablation:

```powershell
python benchmarks/evaluate.py --manifest benchmarks/vulnhub_local.json --audit tests/fixtures/benchmark/audit.json --findings tests/fixtures/benchmark/findings.json --graph tests/fixtures/benchmark/graph.json --ablation no-graph
```

In `no-graph` mode, KG Node Recall, KG Edge Recall, and Evidence Chain Completeness are reported as `null`.

## PentestGPT / AutoPentester Conversion

To compare PentestGPT or AutoPentester runs, convert their logs into the same offline artifact shape:

- Put each action or command into an `audit_log` entry with `event_type`, `source_task_id`, command summary, success/failure, and duration if known.
- Put validated issues into `findings`, using `service_ref`, `vulnerability_ref`, `evidence_refs`, `validation_status`, severity, and CVE when known.
- Put proof snippets into `evidence`, using stable `evidence_id` values.
- Build a minimal `graph.json` with `nodes` and `edges` for Host, Service, Vulnerability, and Evidence.
- Use the same manifest target and ground truth so all tools are scored against identical scope and success criteria.

## Metric Alignment

| Aegra Benchmark Metric | PentestGPT / AutoPentester Alignment |
| --- | --- |
| Target Completion | Overall task success / successful attack completion |
| Subtask Completion % | Reasoning stage or subtask completion rate |
| Service Coverage % | Reconnaissance and service discovery coverage |
| Vulnerability Coverage % | Validated vulnerability discovery coverage |
| Steps | Action count, command count, or attack rounds |
| Loops | Repeated reasoning/action loops |
| Human Interaction | Manual approvals or human intervention |
| Incomplete Commands | Failed, unsupported, placeholder, or malformed commands |
| Time | End-to-end runtime |
| Cost | LLM/tool cost when reported |
| KG Node Recall | Aegra-specific graph recall against ground truth |
| KG Edge Recall | Aegra-specific relationship recall against ground truth |
| Evidence Chain Completeness | Completeness of Host -> Service -> Vulnerability -> Evidence chains |
| False Positive Rate | Unmatched or rejected finding ratio |

## Current Limitations

- HTB10 support is a schema-compatible skeleton only. Add target-specific ground truth only after authorization is recorded.
- The evaluator uses flexible matching heuristics rather than a formal ontology.
- PentestGPT / AutoPentester logs require manual or semi-automated normalization into audit/findings/graph JSON.
- The first version does not execute any live validation and cannot prove authorization by itself.
