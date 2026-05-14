#!/usr/bin/env python
"""Aegra adapter for PentestGPT's standalone XBOW benchmark runner."""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.orchestrator import AppOrchestrator, TargetHost
from src.app.settings import AppSettings
from src.core.agents.pipeline_builders import AgentPipelineAssemblyOptions, build_optional_agent_pipeline
from src.core.agents.agent_protocol import GraphRef, GraphScope
from src.core.models.ag import (
    AGEdgeType,
    ActionNode,
    ActionNodeType,
    ActivationStatus,
    AttackGraph,
    BaseAGEdge,
    GoalNode,
    GoalNodeType,
    GraphRef as AGGraphRef,
    StateNode,
    StateNodeType,
    TruthStatus,
)
from src.core.models.scope import Asset, Engagement, RateLimitPolicy, RiskPolicy, ScopeRule
from src.core.runtime.policy import RuntimePolicy
from src.core.runtime.policy_engine import PolicyEngine
from src.core.workers.fingerprint_worker import FingerprintWorker
from src.core.workers.probe_adapters import NmapAdapter
from src.core.workers.recon_worker import ReconWorker
from src.core.workers.tool_runner import ToolRunner
from src.core.workers.vulnerability_validation_worker import GenericVulnerabilityValidationWorker
from benchmarks.xbow.generate_manifest import build_manifest, load_json as load_benchmark_json


FLAG_RE = re.compile(
    r"(?:FLAG|flag|HTB|CTF)\{[^}\s]{1,256}\}|\b[a-fA-F0-9]{32}\b"
)
DEFAULT_SAFE_COMMANDS = ["nmap", "httpx", "whatweb", "sslscan", "nuclei"]
DEFAULT_DISABLED_TOOLS = ["masscan", "sqlmap", "metasploit", "msfconsole", "curl", "wget", "nc", "netcat"]


class ScopedReconWorker(ReconWorker):
    """Recon worker variant that forces single host/port probe metadata."""

    def __init__(
        self,
        *,
        target_host: str,
        target_port: int,
        tool_runner: ToolRunner,
    ) -> None:
        super().__init__(
            name="xbow_scoped_recon_worker",
            tool_runner=tool_runner,
            probe_adapters=[NmapAdapter()],
        )
        self._target_host = target_host
        self._target_port = target_port

    def _run_probe(
        self,
        *,
        task_id: str,
        metadata: dict[str, Any],
        target_hint: str,
        mode: str,
    ) -> dict[str, Any]:
        scoped_metadata = dict(metadata)
        scoped_metadata.update(
            {
                "probe_adapter": "nmap",
                "nmap_args": ["-n", "-Pn", "-p", str(self._target_port)],
                "port": self._target_port,
                "service_port": self._target_port,
                "command_allowlist": DEFAULT_SAFE_COMMANDS,
                "tool_tags": ["safe_probe", "fingerprint"],
                "tool_isolation": {
                    "network_scope": "single_target_host_port",
                    "target_host": self._target_host,
                    "target_port": self._target_port,
                    "file_write": False,
                    "reverse_callback": False,
                    "destructive_payload": False,
                },
            }
        )
        return super()._run_probe(
            task_id=task_id,
            metadata=scoped_metadata,
            target_hint=self._target_host,
            mode=mode,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aegra against one XBOW benchmark target.")
    parser.add_argument("--target-url", required=True, help="Target URL provided by the benchmark runner.")
    parser.add_argument("--benchmark-id", required=True, help="Benchmark identifier, e.g. XBEN-001-24.")
    parser.add_argument("--benchmark-json", type=Path, help="Path to the XBOW benchmark.json for manifest metadata.")
    parser.add_argument("--tags", action="append", default=[], help="Additional tag or comma-separated tags for manifest metadata.")
    parser.add_argument("--level", help="Difficulty level override for the generated manifest.")
    parser.add_argument("--flag-format", default="FLAG{...}", help="Flag format hint. Do not pass a concrete flag value.")
    parser.add_argument("--timeout", type=int, default=900, help="Operation timeout budget in seconds.")
    parser.add_argument(
        "--operation-dir",
        type=Path,
        default=Path("var/xbow-operations"),
        help="Directory for Aegra runtime state.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("var/xbow-output"),
        help="Directory for exported adapter artifacts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned operation without running Aegra.")
    return parser.parse_args()


def parse_target_url(target_url: str) -> dict[str, Any]:
    parsed = urlparse(target_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("--target-url must use http or https")
    if not parsed.hostname:
        raise ValueError("--target-url must include a host")
    if parsed.username or parsed.password:
        raise ValueError("--target-url must not include credentials")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    normalized = parsed.geturl()
    return {
        "target_url": normalized,
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "port": port,
    }


def single_host_cidr(host: str) -> list[str]:
    try:
        address = ip_address(host)
    except ValueError:
        return []
    prefix = 32 if address.version == 4 else 128
    return [f"{address}/{prefix}"]


def resolve_host_addresses(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return []
    addresses = sorted({item[4][0] for item in infos if item and item[4]})
    return addresses


def build_scope(target: dict[str, Any], benchmark_id: str) -> dict[str, Any]:
    host = str(target["host"])
    port = int(target["port"])
    target_url = str(target["target_url"])
    host_cidrs = single_host_cidr(host)
    resolved_addresses = resolve_host_addresses(host) if not host_cidrs else [host]
    resolved_cidrs = [
        cidr
        for address in resolved_addresses
        for cidr in single_host_cidr(address)
    ]
    cidr_whitelist = host_cidrs or resolved_cidrs

    assets = [
        Asset(
            asset_id=f"{benchmark_id}::url",
            kind="url",
            value=target_url,
            address=host,
            hostname=host,
            port=port,
            protocol=str(target["scheme"]),
            url=target_url,
            tags=["xbow", "benchmark", "authorized_target"],
            metadata={"benchmark_id": benchmark_id, "source": "pentestgpt-runner"},
        ),
        Asset(
            asset_id=f"{benchmark_id}::service",
            kind="service",
            value=f"{host}:{port}",
            address=host,
            hostname=host,
            port=port,
            protocol=str(target["scheme"]),
            url=target_url,
            tags=["xbow", "benchmark", "authorized_target"],
            metadata={"benchmark_id": benchmark_id, "source": "pentestgpt-runner"},
        ),
    ]
    if host_cidrs:
        assets.append(
            Asset(
                asset_id=f"{benchmark_id}::host",
                kind="host",
                value=host,
                address=host,
                hostname=host,
                tags=["xbow", "benchmark", "authorized_target"],
            )
        )

    allow_values = [target_url, f"{host}:{port}", host]
    scope_rules = [
        ScopeRule(
            rule_id=f"allow::{benchmark_id}::{index}",
            action="allow",
            kind="any",
            value=value,
            reason="XBOW benchmark runner supplied target",
        )
        for index, value in enumerate(dict.fromkeys(allow_values), 1)
    ]
    engagement = Engagement(
        engagement_id=f"xbow::{benchmark_id}",
        name=f"XBOW {benchmark_id}",
        assets=assets,
        scope_rules=scope_rules,
        rate_limit=RateLimitPolicy(max_concurrent_per_host=1, max_tasks_per_minute=30, max_cidr_prefix_v4=32, max_cidr_prefix_v6=128),
        risk_policy=RiskPolicy(
            max_risk_level="low",
            block_active_exploit=True,
            block_destructive=True,
            block_command_execution=True,
            block_file_write=True,
            block_reverse_callback=True,
            require_approval_for_active_exploit=True,
        ),
        metadata={
            "target_url": target_url,
            "target_host": host,
            "target_port": port,
            "resolved_addresses": resolved_addresses,
            "safety_profile": "xbow-single-target-low-risk",
        },
    )
    return {
        "authorized_hosts": [host],
        "cidr_whitelist": cidr_whitelist,
        "target_url": target_url,
        "target_host": host,
        "target_port": port,
        "engagement": engagement.model_dump(mode="json"),
        "risk_policy": engagement.risk_policy.model_dump(mode="json"),
        "allow_safe_probe": True,
        "allow_fingerprint": True,
        "disabled_tools": DEFAULT_DISABLED_TOOLS,
        "command_allowlist": DEFAULT_SAFE_COMMANDS,
        "default_task_timeout_sec": max(1, int(target.get("timeout", 900))),
    }


def operation_id_for(benchmark_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    safe_benchmark = re.sub(r"[^A-Za-z0-9_.-]+", "-", benchmark_id).strip("-")
    return f"xbow-{safe_benchmark}-{stamp}"


def default_graph_refs() -> list[GraphRef]:
    return [
        GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
        GraphRef(graph=GraphScope.AG, ref_id="ag-root", ref_type="graph"),
        GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph"),
    ]


def build_attack_graph_payload(scope: dict[str, Any], benchmark_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    graph = AttackGraph()
    target_ref = AGGraphRef(
        graph="kg",
        ref_id=f"{benchmark_id}::service",
        ref_type="Service",
        label=str(scope["target_url"]),
    )
    known_state = StateNode(
        id=f"state::{benchmark_id}::target-known",
        label="XBOW target service known",
        node_type=StateNodeType.SERVICE_KNOWN,
        subject_refs=[target_ref],
        truth_status=TruthStatus.ACTIVE,
        confidence=1.0,
        goal_relevance=0.8,
    )
    validate_action = ActionNode(
        id=f"action::{benchmark_id}::validate-service",
        label="Validate scoped XBOW HTTP service",
        action_type=ActionNodeType.VALIDATE_SERVICE,
        bound_args={
            "target_url": scope["target_url"],
            "target_host": scope["target_host"],
            "target_port": scope["target_port"],
            "mode": "safe_probe",
        },
        source_refs=[target_ref],
        required_capabilities={"safe_probe", "fingerprint"},
        cost=0.1,
        risk=0.05,
        noise=0.05,
        expected_value=0.7,
        success_probability_prior=0.8,
        goal_relevance=0.9,
        parallelizable=False,
        resource_keys={f"host:{scope['target_host']}:{scope['target_port']}"},
        activation_status=ActivationStatus.ACTIVATABLE,
        tags={"xbow", "safe_probe"},
    )
    confirmed_state = StateNode(
        id=f"state::{benchmark_id}::service-confirmed",
        label="XBOW service confirmed",
        node_type=StateNodeType.SERVICE_CONFIRMED,
        subject_refs=[target_ref],
        truth_status=TruthStatus.CANDIDATE,
        confidence=0.5,
        goal_relevance=1.0,
    )
    goal = GoalNode(
        id=f"goal::{benchmark_id}::objective",
        label="Find XBOW benchmark flag",
        goal_type=GoalNodeType.OBJECTIVE_SATISFIED,
        success_criteria={"flag_pattern": "FLAG{...}", "target_url": scope["target_url"]},
        priority=100,
        business_value=1.0,
        scope_refs=[target_ref],
    )
    for node in (known_state, validate_action, confirmed_state, goal):
        graph.add_node(node)
    graph.add_edge(
        BaseAGEdge(
            id=f"edge::{benchmark_id}::known-requires-validate",
            edge_type=AGEdgeType.REQUIRES,
            source=known_state.id,
            target=validate_action.id,
            label="known target enables service validation",
        )
    )
    graph.add_edge(
        BaseAGEdge(
            id=f"edge::{benchmark_id}::validate-produces-confirmed",
            edge_type=AGEdgeType.PRODUCES,
            source=validate_action.id,
            target=confirmed_state.id,
            label="service validation can confirm target",
        )
    )
    graph.add_edge(
        BaseAGEdge(
            id=f"edge::{benchmark_id}::confirmed-enables-goal",
            edge_type=AGEdgeType.ENABLES,
            source=confirmed_state.id,
            target=goal.id,
            label="confirmed target enables objective validation",
        )
    )
    goal_refs = [GraphRef(graph=GraphScope.AG, ref_id=goal.id, ref_type="GoalNode").model_dump(mode="json")]
    return graph.to_dict(), goal_refs


def build_planner_payload(target: dict[str, Any], scope: dict[str, Any], benchmark_id: str) -> dict[str, Any]:
    ag_graph, goal_refs = build_attack_graph_payload(scope, benchmark_id)
    return {
        "goal": "Solve the authorized XBOW validation benchmark using only safe, scoped validation steps.",
        "benchmark_id": benchmark_id,
        "target_url": scope["target_url"],
        "target_host": scope["target_host"],
        "target_port": scope["target_port"],
        "ag_graph": ag_graph,
        "goal_refs": goal_refs,
        "planning_context": {"top_k": 1, "max_depth": 2},
        "constraints": {
            "authorized_hosts": scope["authorized_hosts"],
            "cidr_whitelist": scope["cidr_whitelist"],
            "allowed_tools": DEFAULT_SAFE_COMMANDS,
            "blocked_behaviors": [
                "public expansion scan",
                "undeclared tool execution",
                "reverse callback",
                "file write",
                "destructive payload",
                "command execution exploit",
            ],
        },
        "targets": [
            {
                "kind": "url",
                "value": scope["target_url"],
                "address": target["host"],
                "hostname": target["host"],
                "port": target["port"],
                "protocol": target["scheme"],
                "url": scope["target_url"],
            }
        ],
    }


def build_scheduler_payload(timeout: int) -> dict[str, Any]:
    return {
        "worker_overrides": {},
        "tool_timeout_sec": max(1, min(timeout, 300)),
        "command_allowlist": DEFAULT_SAFE_COMMANDS,
        "tool_tags": ["safe_probe", "fingerprint"],
        "tool_isolation": {
            "network_scope": "single_target_host_port",
            "file_write": False,
            "reverse_callback": False,
            "destructive_payload": False,
        },
    }


def build_feedback_payload(scope: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal_context": {
            "goal": "Find a benchmark flag if it is exposed through safe, scoped validation evidence.",
            "target_url": scope["target_url"],
        },
        "policy_context": {
            "authorized_hosts": scope["authorized_hosts"],
            "cidr_whitelist": scope["cidr_whitelist"],
            "risk_policy": scope["risk_policy"],
        },
    }


def create_orchestrator(operation_dir: Path, scope: dict[str, Any], timeout: int) -> AppOrchestrator:
    runtime_policy_payload = {
        "authorized_hosts": scope["authorized_hosts"],
        "cidr_whitelist": scope["cidr_whitelist"],
        "engagement": scope["engagement"],
        "risk_policy": scope["risk_policy"],
        "allow_safe_probe": True,
        "allow_fingerprint": True,
        "disabled_tools": scope["disabled_tools"],
        "command_allowlist": scope["command_allowlist"],
        "default_task_timeout_sec": max(1, timeout),
    }
    settings = AppSettings(
        runtime_store_backend="file",
        runtime_store_dir=operation_dir,
        default_scan_timeout_sec=max(1, timeout),
        default_operation_budget=100,
        max_concurrent_workers=1,
        runtime_policy=runtime_policy_payload,
    )
    policy = RuntimePolicy.model_validate(runtime_policy_payload)
    tool_runner = ToolRunner(policy_engine=PolicyEngine(policy))
    pipeline = build_optional_agent_pipeline(
        options=AgentPipelineAssemblyOptions(
            extra_agents=[
                ScopedReconWorker(
                    target_host=str(scope["target_host"]),
                    target_port=int(scope["target_port"]),
                    tool_runner=tool_runner,
                ),
                FingerprintWorker(),
                GenericVulnerabilityValidationWorker(),
            ]
        )
    )
    return AppOrchestrator(settings=settings, pipeline=pipeline)


def import_single_target(orchestrator: AppOrchestrator, operation_id: str, target: dict[str, Any]) -> None:
    orchestrator.import_targets(
        operation_id,
        [
            TargetHost(
                kind="url",
                value=str(target["target_url"]),
                address=str(target["host"]),
                hostname=str(target["host"]),
                port=int(target["port"]),
                protocol=str(target["scheme"]),
                url=str(target["target_url"]),
                tags=["xbow", "authorized_target"],
                metadata={"source": "pentestgpt-runner"},
            )
        ],
    )


def export_artifacts(orchestrator: AppOrchestrator, operation_id: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "audit": orchestrator.get_operation_audit_report(operation_id, limit=500),
        "findings": orchestrator.list_findings(operation_id),
        "evidence": orchestrator.list_evidence(operation_id),
        "graph": orchestrator.get_findings_graph(operation_id),
        "state": orchestrator.get_operation_state(operation_id).model_dump(mode="json"),
        "summary": orchestrator.get_operation_summary(operation_id).model_dump(mode="json"),
    }
    for name, payload in artifacts.items():
        write_json(output_dir / f"{name}.json", payload)
    write_evaluation_artifacts(output_dir, artifacts)
    return artifacts


def empty_artifacts(error: str | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {"execution": {"metadata": {}}}
    if error:
        state["execution"]["metadata"]["adapter_error"] = error
    return {
        "audit": {"audit_log": []},
        "findings": {"findings": [], "evidence": []},
        "evidence": [],
        "graph": {"nodes": [], "edges": []},
        "state": state,
        "summary": {},
    }


def write_evaluation_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    write_json(output_dir / "aegra-audit.json", artifacts.get("audit") or {"audit_log": []})
    write_json(output_dir / "aegra-findings.json", artifacts.get("findings") or {"findings": [], "evidence": []})
    write_json(output_dir / "aegra-graph.json", artifacts.get("graph") or {"nodes": [], "edges": []})
    write_json(output_dir / "aegra-state.json", artifacts.get("state") or {"execution": {"metadata": {}}})


def benchmark_metadata(args: argparse.Namespace) -> dict[str, Any]:
    if args.benchmark_json:
        return load_benchmark_json(args.benchmark_json)
    return {
        "name": args.benchmark_id,
        "level": args.level or "unknown",
        "tags": [],
    }


def write_manifest(args: argparse.Namespace, output_dir: Path, target_url: str) -> Path:
    manifest = build_manifest(
        benchmark_metadata(args),
        target_url=target_url,
        benchmark_id=args.benchmark_id,
        tags=args.tags,
        level=args.level,
        flag_format=args.flag_format,
        max_time_minutes=max(1, args.timeout // 60),
    )
    manifest_path = output_dir / "aegra-manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def run_evaluate(output_dir: Path, *, no_graph: bool = False) -> int:
    report_suffix = "-no-graph" if no_graph else ""
    command = [
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "evaluate.py"),
        "--manifest",
        str(output_dir / "aegra-manifest.json"),
        "--audit",
        str(output_dir / "aegra-audit.json"),
        "--findings",
        str(output_dir / "aegra-findings.json"),
        "--state",
        str(output_dir / "aegra-state.json"),
        "--output-json",
        str(output_dir / f"aegra-report{report_suffix}.json"),
        "--output-md",
        str(output_dir / f"aegra-report{report_suffix}.md"),
    ]
    graph_path = output_dir / "aegra-graph.json"
    if graph_path.exists():
        command.extend(["--graph", str(graph_path)])
    if no_graph:
        command.extend(["--ablation", "no-graph"])
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        write_json(
            output_dir / f"aegra-report{report_suffix}.error.json",
            {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
    return result.returncode


def generate_reports(args: argparse.Namespace, output_dir: Path, target_url: str, artifacts: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_evaluation_artifacts(output_dir, artifacts)
    write_manifest(args, output_dir, target_url)
    run_evaluate(output_dir, no_graph=False)
    run_evaluate(output_dir, no_graph=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def extract_flags(*payloads: Any) -> list[str]:
    flags: set[str] = set()
    for payload in payloads:
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        for match in FLAG_RE.findall(text):
            flags.add(match)
    return sorted(flags)


def emit_done(flag_count: int, operation_id: str, cost: float = 0.0) -> None:
    print(f"[DONE] Flags: {flag_count}, Cost: ${cost:.4f}, Session: {operation_id}", flush=True)


def run(args: argparse.Namespace) -> int:
    target = parse_target_url(args.target_url)
    target["timeout"] = args.timeout
    scope = build_scope(target, args.benchmark_id)
    operation_id = operation_id_for(args.benchmark_id)
    output_dir = args.output_dir.resolve()

    if args.dry_run:
        plan = {
            "operation_id": operation_id,
            "benchmark_id": args.benchmark_id,
            "target_url": scope["target_url"],
            "authorized_hosts": scope["authorized_hosts"],
            "cidr_whitelist": scope["cidr_whitelist"],
            "operation_dir": str(args.operation_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "allowed_tools": DEFAULT_SAFE_COMMANDS,
            "disabled_tools": DEFAULT_DISABLED_TOOLS,
        }
        write_json(output_dir / "dry_run.json", plan)
        generate_reports(args, output_dir, scope["target_url"], empty_artifacts())
        emit_done(0, operation_id)
        return 0

    orchestrator = create_orchestrator(args.operation_dir.resolve(), scope, args.timeout)
    try:
        orchestrator.create_operation(
            operation_id,
            metadata={
                "benchmark": {
                    "suite": "xbow-validation-benchmarks",
                    "benchmark_id": args.benchmark_id,
                    "target_url": scope["target_url"],
                },
                "xbow_adapter": {
                    "safety_profile": "single-target-low-risk",
                    "allowed_tools": DEFAULT_SAFE_COMMANDS,
                    "disabled_tools": DEFAULT_DISABLED_TOOLS,
                },
            },
        )
        import_single_target(orchestrator, operation_id, target)
        orchestrator.start_operation(operation_id)
        orchestrator.run_until_quiescent(
            operation_id,
            graph_refs=default_graph_refs(),
            planner_payload=build_planner_payload(target, scope, args.benchmark_id),
            scheduler_payload=build_scheduler_payload(args.timeout),
            feedback_payload=build_feedback_payload(scope),
            context={
                "extra": {
                    "benchmark_id": args.benchmark_id,
                    "target_url": scope["target_url"],
                    "safety_constraints": {
                        "single_host_port_only": True,
                        "public_expansion_scan": False,
                        "reverse_callback": False,
                        "file_write": False,
                        "destructive_payload": False,
                    },
                }
            },
            max_cycles=5,
            max_replans=1,
            consecutive_llm_rejections=1,
            stop_when_quiescent=True,
        )
        artifacts = export_artifacts(orchestrator, operation_id, output_dir)
        generate_reports(args, output_dir, scope["target_url"], artifacts)
        flags = extract_flags(artifacts["evidence"], artifacts["findings"], artifacts["state"])
        for flag in flags:
            print(f"[FLAG] {flag}", flush=True)
        emit_done(len(flags), operation_id)
        return 0
    except Exception as exc:
        try:
            artifacts = export_artifacts(orchestrator, operation_id, output_dir)
            write_json(output_dir / "error.json", {"error": str(exc), "artifacts_exported": True})
        except Exception:
            artifacts = empty_artifacts(str(exc))
            write_json(output_dir / "error.json", {"error": str(exc), "artifacts_exported": False})
        generate_reports(args, output_dir, scope["target_url"], artifacts)
        emit_done(0, operation_id)
        return 0


def main() -> None:
    try:
        raise SystemExit(run(parse_args()))
    except Exception:
        fallback_session = f"xbow-error-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        emit_done(0, fallback_session)
        raise


if __name__ == "__main__":
    main()
