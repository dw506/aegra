"""Offline benchmark evaluator for Aegra operation outputs.

The evaluator compares an authorized benchmark manifest with exported Aegra
artifacts. It performs only local JSON parsing and scoring; it never invokes
network tools, scanners, or exploit code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.runtime.repetition_detector import repetition_summary


MetricValue = int | float | str | bool | None

SOURCES = {"HTB", "VulnHub", "Custom", "XBOW"}
RISK_LEVELS = {"low", "medium", "high"}
INCOMPLETE_PATTERNS = (
    "command_error",
    "placeholder command",
    "missing_target",
    "unsupported flag",
    "unsupported_flag",
    "not implemented",
    "not_implemented",
    "timed out",
    "timeout",
)
HUMAN_PATTERNS = (
    "approval_decision",
    "waiting_approval",
    "manual_",
    "human_interaction",
)


def load_json(path: str | Path | None, default: Any = None) -> Any:
    if not path:
        return default
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(data: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def flatten_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True).lower()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 100.0
    return round((numerator / denominator) * 100.0, 2)


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return schema-like validation errors without requiring jsonschema."""

    errors: list[str] = []
    for key in ("benchmark_id", "name", "description", "source", "targets"):
        if key not in manifest:
            errors.append(f"missing manifest field: {key}")
    if manifest.get("source") not in SOURCES:
        errors.append("source must be one of HTB, VulnHub, Custom, XBOW")
    if not isinstance(manifest.get("targets"), list) or not manifest.get("targets"):
        errors.append("targets must be a non-empty list")
        return errors

    target_required = {
        "target_id",
        "name",
        "source",
        "difficulty",
        "os",
        "scope",
        "expected_services",
        "ground_truth_subtasks",
        "ground_truth_vulnerabilities",
        "success_conditions",
        "allowed_tools",
        "max_steps",
        "max_time_minutes",
        "risk_level",
        "notes",
    }
    for index, target in enumerate(manifest["targets"]):
        prefix = f"targets[{index}]"
        if not isinstance(target, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = sorted(target_required - set(target))
        errors.extend(f"{prefix} missing field: {field}" for field in missing)
        if target.get("source") not in SOURCES:
            errors.append(f"{prefix}.source must be one of HTB, VulnHub, Custom, XBOW")
        if target.get("risk_level") not in RISK_LEVELS:
            errors.append(f"{prefix}.risk_level must be low, medium, or high")
        if not isinstance(target.get("allowed_tools"), list) or not target.get("allowed_tools"):
            errors.append(f"{prefix}.allowed_tools must be a non-empty list")
        scope = target.get("scope")
        if not isinstance(scope, dict):
            errors.append(f"{prefix}.scope must be an object")
        else:
            for field in ("authorized_hosts", "cidr_whitelist", "target_url"):
                if field not in scope:
                    errors.append(f"{prefix}.scope missing field: {field}")
            if not isinstance(scope.get("authorized_hosts"), list):
                errors.append(f"{prefix}.scope.authorized_hosts must be a list")
            if not isinstance(scope.get("cidr_whitelist"), list):
                errors.append(f"{prefix}.scope.cidr_whitelist must be a list")
        for service_index, service in enumerate(as_list(target.get("expected_services"))):
            for field in ("service_id", "host", "port", "protocol", "service_name"):
                if field not in service:
                    errors.append(f"{prefix}.expected_services[{service_index}] missing field: {field}")
        for subtask_index, subtask in enumerate(as_list(target.get("ground_truth_subtasks"))):
            for field in ("subtask_id", "category", "description", "success_evidence"):
                if field not in subtask:
                    errors.append(f"{prefix}.ground_truth_subtasks[{subtask_index}] missing field: {field}")
        for vuln_index, vuln in enumerate(as_list(target.get("ground_truth_vulnerabilities"))):
            for field in ("vulnerability_id", "name", "cve", "service_id", "severity", "required_evidence"):
                if field not in vuln:
                    errors.append(f"{prefix}.ground_truth_vulnerabilities[{vuln_index}] missing field: {field}")
    return errors


def select_target(manifest: dict[str, Any], target_id: str | None) -> dict[str, Any]:
    targets = as_list(manifest.get("targets"))
    if target_id:
        for target in targets:
            if target.get("target_id") == target_id:
                return target
        raise ValueError(f"target_id not found in manifest: {target_id}")
    if len(targets) != 1:
        ids = ", ".join(str(target.get("target_id")) for target in targets)
        raise ValueError(f"--target-id is required when manifest has multiple targets: {ids}")
    return dict(targets[0])


def discover_operation_artifacts(operation_dir: str | Path | None) -> dict[str, Path | None]:
    if not operation_dir:
        return {"audit": None, "findings": None, "graph": None, "state": None}
    base = Path(operation_dir)

    def first_existing(names: list[str]) -> Path | None:
        for name in names:
            path = base / name
            if path.exists():
                return path
        return None

    return {
        "audit": first_existing(["audit-report.json", "audit.json"]),
        "findings": first_existing(["findings.json", "report.json", "findings-report.json"]),
        "graph": first_existing(["graph.json", "kg.json"]),
        "state": first_existing(["state.json", "runtime-state.json"]),
    }


def extract_findings(findings_payload: Any, state_payload: Any = None) -> list[dict[str, Any]]:
    if isinstance(findings_payload, list):
        return [item for item in findings_payload if isinstance(item, dict)]
    if isinstance(findings_payload, dict):
        for key in ("findings", "items", "results"):
            if isinstance(findings_payload.get(key), list):
                return [item for item in findings_payload[key] if isinstance(item, dict)]
    metadata = (((state_payload or {}).get("execution") or {}).get("metadata") or {}) if isinstance(state_payload, dict) else {}
    if isinstance(metadata.get("findings"), list):
        return [item for item in metadata["findings"] if isinstance(item, dict)]
    return []


def extract_evidence(findings_payload: Any, state_payload: Any = None) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    if isinstance(findings_payload, dict):
        evidence.extend(item for item in as_list(findings_payload.get("evidence")) if isinstance(item, dict))
    metadata = (((state_payload or {}).get("execution") or {}).get("metadata") or {}) if isinstance(state_payload, dict) else {}
    evidence.extend(item for item in as_list(metadata.get("evidence_artifacts")) if isinstance(item, dict))
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in evidence:
        key = str(item.get("evidence_id") or json.dumps(item, sort_keys=True))
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def extract_audit_events(audit_payload: Any, state_payload: Any = None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if isinstance(audit_payload, list):
        events.extend(item for item in audit_payload if isinstance(item, dict))
    elif isinstance(audit_payload, dict):
        for key in ("audit_log", "operation_log", "events", "control_cycle_history"):
            events.extend(item for item in as_list(audit_payload.get(key)) if isinstance(item, dict))
    metadata = (((state_payload or {}).get("execution") or {}).get("metadata") or {}) if isinstance(state_payload, dict) else {}
    for key in ("audit_log", "operation_log", "control_cycle_history"):
        events.extend(item for item in as_list(metadata.get(key)) if isinstance(item, dict))
    return events


def graph_nodes(graph_payload: Any) -> list[dict[str, Any]]:
    if isinstance(graph_payload, dict):
        return [item for item in as_list(graph_payload.get("nodes")) if isinstance(item, dict)]
    return []


def graph_edges(graph_payload: Any) -> list[dict[str, Any]]:
    if isinstance(graph_payload, dict):
        return [item for item in as_list(graph_payload.get("edges")) if isinstance(item, dict)]
    return []


def node_matches_service(node: dict[str, Any], service: dict[str, Any]) -> bool:
    text = flatten_json(node)
    service_id = normalize_text(service.get("service_id"))
    host = normalize_text(service.get("host"))
    port = str(service.get("port"))
    name = normalize_text(service.get("service_name"))
    protocol = normalize_text(service.get("protocol"))
    if service_id and service_id in text:
        return True
    return host in text and port in text and (name in text or protocol in text)


def finding_matches_service(finding: dict[str, Any], service: dict[str, Any]) -> bool:
    text = flatten_json(finding)
    return any(
        token and token in text
        for token in (
            normalize_text(service.get("service_id")),
            f"{normalize_text(service.get('host'))}:{service.get('port')}",
            normalize_text(service.get("service_name")),
        )
    )


def vulnerability_tokens(vuln: dict[str, Any]) -> list[str]:
    return [
        token
        for token in (
            normalize_text(vuln.get("vulnerability_id")),
            normalize_text(vuln.get("name")),
            normalize_text(vuln.get("cve")),
            normalize_text(vuln.get("required_evidence")),
        )
        if token
    ]


def finding_matches_vulnerability(finding: dict[str, Any], vuln: dict[str, Any]) -> bool:
    text = flatten_json(finding)
    return any(token in text for token in vulnerability_tokens(vuln))


def node_matches_vulnerability(node: dict[str, Any], vuln: dict[str, Any]) -> bool:
    text = flatten_json(node)
    return any(token in text for token in vulnerability_tokens(vuln))


def evidence_matches(evidence_items: list[dict[str, Any]], required: str, findings: list[dict[str, Any]]) -> bool:
    required_text = normalize_text(required)
    if not required_text:
        return True
    return required_text in flatten_json(evidence_items) or required_text in flatten_json(findings)


def count_service_coverage(target: dict[str, Any], findings: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> tuple[int, list[str]]:
    covered: list[str] = []
    for service in as_list(target.get("expected_services")):
        if any(node_matches_service(node, service) for node in nodes) or any(finding_matches_service(item, service) for item in findings):
            covered.append(str(service.get("service_id")))
    return len(covered), covered


def count_vulnerability_coverage(
    target: dict[str, Any],
    findings: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    covered: list[str] = []
    for vuln in as_list(target.get("ground_truth_vulnerabilities")):
        matched = any(finding_matches_vulnerability(item, vuln) for item in findings) or any(
            node_matches_vulnerability(node, vuln) for node in nodes
        )
        if matched and evidence_matches(evidence_items, str(vuln.get("required_evidence") or ""), findings):
            covered.append(str(vuln.get("vulnerability_id")))
    return len(covered), covered


def count_subtask_coverage(target: dict[str, Any], payloads: list[Any]) -> tuple[int, list[str]]:
    text = flatten_json(payloads)
    covered: list[str] = []
    for subtask in as_list(target.get("ground_truth_subtasks")):
        success_evidence = normalize_text(subtask.get("success_evidence"))
        if not success_evidence or success_evidence in text:
            covered.append(str(subtask.get("subtask_id")))
    return len(covered), covered


def canonical_node_id(node: dict[str, Any]) -> str:
    return normalize_text(node.get("id") or node.get("node_id") or node.get("label"))


def edge_endpoints(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_text(edge.get("source") or edge.get("from") or edge.get("src")),
        normalize_text(edge.get("target") or edge.get("to") or edge.get("dst")),
        normalize_text(edge.get("type") or edge.get("relation") or edge.get("label")),
    )


def has_path_edge(edges: list[dict[str, Any]], sources: set[str], targets: set[str], relation_markers: tuple[str, ...]) -> bool:
    for edge in edges:
        source, target, relation = edge_endpoints(edge)
        if source in sources and target in targets and any(marker in relation for marker in relation_markers):
            return True
    return False


def graph_metric_counts(target: dict[str, Any], graph_payload: Any, findings: list[dict[str, Any]], evidence_items: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = graph_nodes(graph_payload)
    edges = graph_edges(graph_payload)
    node_ids = {canonical_node_id(node) for node in nodes if canonical_node_id(node)}
    expected_node_ids: set[str] = set()
    recalled_node_ids: set[str] = set()

    for service in as_list(target.get("expected_services")):
        expected_node_ids.update({normalize_text(service.get("host")), normalize_text(service.get("service_id"))})
        if any(node_matches_service(node, service) for node in nodes):
            recalled_node_ids.add(normalize_text(service.get("host")))
            recalled_node_ids.add(normalize_text(service.get("service_id")))
    for vuln in as_list(target.get("ground_truth_vulnerabilities")):
        expected_node_ids.add(normalize_text(vuln.get("vulnerability_id")))
        expected_node_ids.add(normalize_text(vuln.get("required_evidence")))
        if any(node_matches_vulnerability(node, vuln) for node in nodes):
            recalled_node_ids.add(normalize_text(vuln.get("vulnerability_id")))
        required = normalize_text(vuln.get("required_evidence"))
        if required and (required in flatten_json(nodes) or required in flatten_json(evidence_items) or required in flatten_json(findings)):
            recalled_node_ids.add(required)

    expected_node_ids = {item for item in expected_node_ids if item}
    recalled_node_ids = {item for item in recalled_node_ids if item in expected_node_ids or item in node_ids}

    expected_edges = 0
    recalled_edges = 0
    evidence_complete = 0
    for service in as_list(target.get("expected_services")):
        host_ids = {normalize_text(service.get("host"))}
        service_ids = {normalize_text(service.get("service_id"))}
        expected_edges += 1
        if has_path_edge(edges, host_ids, service_ids, ("exposes", "has_service", "service")):
            recalled_edges += 1
    service_by_id = {str(item.get("service_id")): item for item in as_list(target.get("expected_services"))}
    for vuln in as_list(target.get("ground_truth_vulnerabilities")):
        vuln_ids = {normalize_text(vuln.get("vulnerability_id"))}
        service_ids = {normalize_text(vuln.get("service_id"))}
        evidence_ids = {canonical_node_id(node) for node in nodes if "evidence" in normalize_text(node.get("type"))}
        expected_edges += 2
        if has_path_edge(edges, service_ids, vuln_ids, ("vulnerability", "affects", "derived")):
            recalled_edges += 1
        has_evidence_edge = has_path_edge(edges, vuln_ids, evidence_ids, ("supported", "evidence", "validated"))
        if has_evidence_edge:
            recalled_edges += 1
        service = service_by_id.get(str(vuln.get("service_id")), {})
        has_chain = bool(service) and has_path_edge(
            edges,
            {normalize_text(service.get("host"))},
            {normalize_text(service.get("service_id"))},
            ("exposes", "has_service", "service"),
        ) and has_path_edge(edges, service_ids, vuln_ids, ("vulnerability", "affects", "derived")) and has_evidence_edge
        if has_chain or (
            any(finding_matches_vulnerability(item, vuln) for item in findings)
            and evidence_matches(evidence_items, str(vuln.get("required_evidence") or ""), findings)
        ):
            evidence_complete += 1

    return {
        "kg_node_recalled": len(recalled_node_ids),
        "kg_node_expected": len(expected_node_ids),
        "kg_edge_recalled": recalled_edges,
        "kg_edge_expected": expected_edges,
        "evidence_chain_complete": evidence_complete,
        "evidence_chain_expected": len(as_list(target.get("ground_truth_vulnerabilities"))),
    }


def estimate_steps(audit_payload: Any, state_payload: Any, audit_events: list[dict[str, Any]]) -> int:
    candidates: list[int] = []
    for payload in (audit_payload,):
        if isinstance(payload, dict):
            audit_log = [item for item in as_list(payload.get("audit_log")) if isinstance(item, dict)]
            if audit_log:
                candidates.append(len(audit_log))
            for cycle in as_list(payload.get("control_cycle_history")):
                if isinstance(cycle, dict):
                    candidates.append(len(as_list(cycle.get("selected_task_ids"))))
    for payload in (audit_payload, state_payload):
        if isinstance(payload, dict):
            text = flatten_json(payload)
            for match in re.finditer(r'"step_count"\s*:\s*(\d+)', text):
                candidates.append(int(match.group(1)))
    if not candidates and audit_events:
        candidates.append(len(audit_events))
    return max(candidates) if candidates else 0


def estimate_loops(audit_events: list[dict[str, Any]]) -> int:
    keys: list[str] = []
    for event in audit_events:
        task_id = event.get("task_id") or event.get("source_task_id")
        command = event.get("command") or ((event.get("tool") or {}).get("command") if isinstance(event.get("tool"), dict) else None)
        phase = event.get("phase")
        action = event.get("action") or event.get("event_type")
        key = task_id or command or f"{phase}:{action}"
        if key:
            keys.append(flatten_json(key))
    counts = Counter(keys)
    return sum(count - 1 for count in counts.values() if count > 1)


def count_human_interaction(audit_events: list[dict[str, Any]]) -> int:
    return sum(1 for event in audit_events if any(pattern in flatten_json(event) for pattern in HUMAN_PATTERNS))


def count_incomplete_commands(audit_events: list[dict[str, Any]]) -> int:
    total = 0
    for event in audit_events:
        text = flatten_json(event)
        tool = event.get("tool") if isinstance(event.get("tool"), dict) else {}
        failed_exit = tool.get("exit_code") not in (None, 0) or tool.get("success") is False or bool(tool.get("error_message"))
        if failed_exit or any(pattern in text for pattern in INCOMPLETE_PATTERNS):
            total += 1
    return total


def extract_time_seconds(audit_payload: Any, state_payload: Any, wall_clock_seconds: float) -> float:
    text = flatten_json([audit_payload, state_payload])
    for pattern in (r'"duration_sec"\s*:\s*([0-9.]+)', r'"time_budget_used_sec"\s*:\s*([0-9.]+)'):
        values = [float(match.group(1)) for match in re.finditer(pattern, text)]
        if values:
            return round(sum(values), 3)
    return round(wall_clock_seconds, 3)


def extract_cost(state_payload: Any, audit_payload: Any) -> float | None:
    text = flatten_json([state_payload, audit_payload])
    matches = re.findall(r'"(?:cost|total_cost|usd_cost)"\s*:\s*([0-9.]+)', text)
    if not matches:
        return None
    return round(sum(float(value) for value in matches), 6)


def false_positive_rate(target: dict[str, Any], findings: list[dict[str, Any]]) -> float:
    if not findings:
        return 0.0
    false_positive = 0
    for finding in findings:
        status = normalize_text(finding.get("validation_status") or finding.get("status"))
        if status in {"false_positive", "rejected"}:
            false_positive += 1
            continue
        if not any(finding_matches_vulnerability(finding, vuln) for vuln in as_list(target.get("ground_truth_vulnerabilities"))):
            false_positive += 1
    return round(false_positive / len(findings), 4)


def evaluate(
    manifest: dict[str, Any],
    *,
    target_id: str | None = None,
    audit_payload: Any = None,
    findings_payload: Any = None,
    graph_payload: Any = None,
    state_payload: Any = None,
    ablation: str | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("; ".join(errors))
    target = select_target(manifest, target_id)
    findings = extract_findings(findings_payload, state_payload)
    evidence_items = extract_evidence(findings_payload, state_payload)
    audit_events = extract_audit_events(audit_payload, state_payload)
    repetition_counts = repetition_summary(audit_events)
    nodes = graph_nodes(graph_payload)

    services_total = len(as_list(target.get("expected_services")))
    vulns_total = len(as_list(target.get("ground_truth_vulnerabilities")))
    subtasks_total = len(as_list(target.get("ground_truth_subtasks")))
    services_done, services = count_service_coverage(target, findings, nodes)
    vulns_done, vulns = count_vulnerability_coverage(target, findings, nodes, evidence_items)
    subtasks_done, subtasks = count_subtask_coverage(target, [audit_payload, findings_payload, graph_payload, state_payload])

    no_graph = ablation == "no-graph"
    graph_counts = graph_metric_counts(target, graph_payload, findings, evidence_items) if not no_graph else {}
    target_completion = services_done == services_total and vulns_done == vulns_total and subtasks_done == subtasks_total
    report = {
        "benchmark_id": manifest.get("benchmark_id"),
        "target": {
            "target_id": target.get("target_id"),
            "name": target.get("name"),
            "source": target.get("source"),
            "difficulty": target.get("difficulty"),
            "risk_level": target.get("risk_level"),
        },
        "ablation": ablation,
        "metrics": {
            "target_completion": target_completion,
            "subtask_completion_percent": percent(subtasks_done, subtasks_total),
            "service_coverage_percent": percent(services_done, services_total),
            "vulnerability_coverage_percent": percent(vulns_done, vulns_total),
            "steps": estimate_steps(audit_payload, state_payload, audit_events),
            "loops": estimate_loops(audit_events),
            "repetition_rejections": repetition_counts["rejection_count"],
            "human_interaction": count_human_interaction(audit_events),
            "incomplete_commands": count_incomplete_commands(audit_events),
            "time_seconds": extract_time_seconds(audit_payload, state_payload, perf_counter() - started),
            "cost": extract_cost(state_payload, audit_payload),
            "kg_node_recall_percent": None if no_graph else percent(graph_counts["kg_node_recalled"], graph_counts["kg_node_expected"]),
            "kg_edge_recall_percent": None if no_graph else percent(graph_counts["kg_edge_recalled"], graph_counts["kg_edge_expected"]),
            "evidence_chain_completeness_percent": None
            if no_graph
            else percent(graph_counts["evidence_chain_complete"], graph_counts["evidence_chain_expected"]),
            "false_positive_rate": false_positive_rate(target, findings),
        },
        "matched": {
            "services": services,
            "vulnerabilities": vulns,
            "subtasks": subtasks,
        },
        "counts": {
            "expected_services": services_total,
            "expected_vulnerabilities": vulns_total,
            "expected_subtasks": subtasks_total,
            "findings": len(findings),
            "evidence": len(evidence_items),
            "graph_nodes": len(nodes),
            "graph_edges": len(graph_edges(graph_payload)),
        },
    }
    return report


def format_value(value: MetricValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def report_to_markdown(report: dict[str, Any]) -> str:
    target = report["target"]
    metrics = report["metrics"]
    rows = [
        ("Target Completion", "PentestGPT task success / AutoPentester successful attack", metrics["target_completion"]),
        ("Subtask Completion %", "PentestGPT reasoning task completion", metrics["subtask_completion_percent"]),
        ("Service Coverage %", "Reconnaissance/service discovery coverage", metrics["service_coverage_percent"]),
        ("Vulnerability Coverage %", "Validated vulnerability discovery coverage", metrics["vulnerability_coverage_percent"]),
        ("Steps", "PentestGPT action count / AutoPentester rounds", metrics["steps"]),
        ("Loops", "Repeated task/command/phase count", metrics["loops"]),
        ("Repetition Rejections", "Rejected repeated task proposals", metrics["repetition_rejections"]),
        ("Human Interaction", "Manual approval or intervention count", metrics["human_interaction"]),
        ("Incomplete Commands", "Failed, placeholder, or unsupported command count", metrics["incomplete_commands"]),
        ("Time", "Elapsed/runtime seconds", metrics["time_seconds"]),
        ("Cost", "LLM/tool cost when exported", metrics["cost"]),
        ("KG Node Recall", "Aegra KG node recall against ground truth", metrics["kg_node_recall_percent"]),
        ("KG Edge Recall", "Aegra KG edge recall against ground truth", metrics["kg_edge_recall_percent"]),
        (
            "Evidence Chain Completeness",
            "Host -> Service -> Vulnerability -> Evidence chain coverage",
            metrics["evidence_chain_completeness_percent"],
        ),
        ("False Positive Rate", "Unmatched or rejected finding ratio", metrics["false_positive_rate"]),
    ]
    lines = [
        f"# Benchmark Report: {target.get('target_id')}",
        "",
        f"- Target ID: `{target.get('target_id')}`",
        f"- Target Name: {target.get('name')}",
        f"- Source: {target.get('source')}",
        f"- Risk Level: {target.get('risk_level')}",
        f"- Ablation: {report.get('ablation') or 'none'}",
        "",
        "## PentestGPT / AutoPentester Alignment Metrics",
        "",
        "| Metric | Alignment | Value |",
        "| --- | --- | ---: |",
    ]
    lines.extend(f"| {metric} | {alignment} | {format_value(value)} |" for metric, alignment, value in rows)
    lines.extend(
        [
            "",
            "## Matched Ground Truth",
            "",
            "| Category | IDs |",
            "| --- | --- |",
            f"| Services | {', '.join(report['matched']['services']) or 'none'} |",
            f"| Vulnerabilities | {', '.join(report['matched']['vulnerabilities']) or 'none'} |",
            f"| Subtasks | {', '.join(report['matched']['subtasks']) or 'none'} |",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate offline Aegra benchmark artifacts against a manifest.")
    parser.add_argument("--manifest", required=True, help="Path to benchmark manifest JSON.")
    parser.add_argument("--operation-dir", help="Directory containing exported audit/findings/graph/state JSON.")
    parser.add_argument("--target-id", help="Target id inside the manifest.")
    parser.add_argument("--audit", help="Path to audit report JSON.")
    parser.add_argument("--findings", help="Path to findings/report JSON.")
    parser.add_argument("--graph", help="Path to graph JSON with nodes/edges.")
    parser.add_argument("--state", help="Path to runtime state JSON.")
    parser.add_argument("--output-json", help="Optional JSON report output path.")
    parser.add_argument("--output-md", help="Optional Markdown report output path.")
    parser.add_argument("--ablation", choices=["no-graph"], help="Ablation mode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    discovered = discover_operation_artifacts(args.operation_dir)
    manifest = load_json(args.manifest)
    audit_path = args.audit or discovered["audit"]
    findings_path = args.findings or discovered["findings"]
    graph_path = args.graph or discovered["graph"]
    state_path = args.state or discovered["state"]
    report = evaluate(
        manifest,
        target_id=args.target_id,
        audit_payload=load_json(audit_path, {}),
        findings_payload=load_json(findings_path, {}),
        graph_payload=load_json(graph_path, {}),
        state_payload=load_json(state_path, {}),
        ablation=args.ablation,
    )
    markdown = report_to_markdown(report)
    if args.output_json:
        dump_json(report, args.output_json)
    if args.output_md:
        Path(args.output_md).write_text(markdown, encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        print()
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
