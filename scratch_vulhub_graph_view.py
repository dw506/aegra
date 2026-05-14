from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.agents.agent_pipeline import AgentPipeline
from src.core.agents.agent_protocol import GraphRef as ProtocolGraphRef, GraphScope
from src.core.agents.critic import CriticAgent
from src.core.agents.scheduler_agent import SchedulerAgent
from src.core.agents.task_builder import TaskBuilderAgent
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import GraphRef as KGGraphRef
from src.core.models.runtime import TaskRuntime
from src.core.models.runtime import WorkerRuntime, WorkerStatus
from src.core.models.tg import TaskNode, TaskType
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.workers.recon_worker import ReconWorker
from src.core.workers.vulnerability_validation_worker import Struts2VulnerabilityValidationWorker


def _load_vulhub_smoke_helpers():
    helper_path = Path(__file__).resolve().parent / "tests" / "test_vulhub_orchestrator_smoke.py"
    spec = importlib.util.spec_from_file_location("aegra_vulhub_orchestrator_smoke_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load Vulhub smoke helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_vulhub_smoke_helpers()
VulhubNmapReconWorkerAgent = _helpers.VulhubNmapReconWorkerAgent
VulhubPlannerAgent = _helpers.VulhubPlannerAgent
VulhubReconWorkerAgent = _helpers.VulhubReconWorkerAgent
_build_graph_refs = _helpers._build_graph_refs
_clean_entity_delta = _helpers._clean_entity_delta
_clean_relation_delta = _helpers._clean_relation_delta
_is_tcp_open = _helpers._is_tcp_open
_resolve_nmap_path = _helpers._resolve_nmap_path
_vulhub_target = _helpers._vulhub_target
_augment_smoke_worker_output = _helpers._augment_smoke_worker_output


class ViewerNmapReconWorkerAgent(VulhubNmapReconWorkerAgent):
    def execute_task(self, task_spec, agent_input=None):  # noqa: ANN001
        if getattr(task_spec, "task_type", None) == TaskType.SERVICE_VALIDATION.value:
            task_spec = task_spec.model_copy(update={"task_type": "service_validation"})
        if agent_input is None:
            return ReconWorker.execute_task(self, task_spec, agent_input)
        host = str(task_spec.input_bindings.get("host_id") or "127.0.0.1")
        port = int(task_spec.input_bindings.get("port") or task_spec.constraints.get("port") or 80)
        bindings = dict(task_spec.input_bindings)
        bindings.update({"host_id": host, "service": host, "port": port})
        task_spec = task_spec.model_copy(update={"input_bindings": bindings})
        payload = dict(agent_input.raw_payload)
        payload.update(
            {
                "probe_adapter": "nmap",
                "nmap_path": self._nmap_path,
                "service_port": port,
                "port": port,
                "tool_timeout_sec": 30,
            }
        )
        modified_input = agent_input.model_copy(deep=True, update={"raw_payload": payload})
        output = ReconWorker.execute_task(self, task_spec, modified_input)
        return _augment_smoke_worker_output(output=output, host=host, port=port, probe_kind="nmap_probe")


def _build_kg_from_apply_results(apply_results) -> KnowledgeGraph:  # noqa: ANN001
    entity_deltas = []
    relation_deltas = []
    for apply_result in apply_results:
        entity_deltas.extend(
            cleaned
            for delta in apply_result.kg_state_deltas
            if (cleaned := _clean_view_entity_delta(delta)) is not None
        )
        relation_deltas.extend(
            cleaned
            for delta in apply_result.kg_state_deltas
            if (cleaned := _clean_view_relation_delta(delta)) is not None
        )

    kg = KnowledgeGraph()
    kg.apply_patch_batch(
        {
            "patch_batch_id": "vulhub-view-entities",
            "base_kg_version": 0,
            "state_deltas": entity_deltas,
            "metadata": {},
        }
    )
    node_ids = {node.id for node in kg.list_nodes()}
    relation_deltas = [
        delta
        for delta in relation_deltas
        if delta.get("patch", {}).get("source") in node_ids and delta.get("patch", {}).get("target") in node_ids
    ]
    kg.apply_patch_batch(
        {
            "patch_batch_id": "vulhub-view-relations",
            "base_kg_version": kg.version,
            "state_deltas": relation_deltas,
            "metadata": {},
        }
    )
    return kg


def _clean_view_entity_delta(delta: dict) -> dict | None:
    cleaned = _clean_entity_delta(delta)
    if cleaned is not None:
        return cleaned
    if delta.get("delta_type") != "upsert_entity":
        return None
    patch = dict(delta.get("patch", {}))
    if str(patch.get("entity_type") or "").lower() != "vulnerability":
        return None
    attributes = dict(patch.get("attributes", {}))
    cleaned_patch = {
        "patch_id": patch.get("patch_id"),
        "entity_id": patch.get("entity_id"),
        "operation": patch.get("operation"),
        "entity_kind": patch.get("entity_kind"),
        "entity_type": patch.get("entity_type"),
        "label": attributes.get("vulnerability_name") or patch.get("entity_id"),
        "vulnerability_name": attributes.get("vulnerability_name"),
        "validation_status": attributes.get("validation_status"),
        "cve": attributes.get("cve"),
        "cwe": attributes.get("cwe"),
        "advisory_refs": attributes.get("advisory_refs", []),
        "summary": attributes.get("summary"),
    }
    return {**delta, "patch": {key: value for key, value in cleaned_patch.items() if value is not None}}


def _clean_view_relation_delta(delta: dict) -> dict | None:
    cleaned = _clean_relation_delta(delta)
    if cleaned is not None:
        return cleaned
    if delta.get("delta_type") != "upsert_relation":
        return None
    patch = dict(delta.get("patch", {}))
    if str(patch.get("relation_type") or "").upper() != "HAS_VULNERABILITY":
        return None
    cleaned_patch = {
        "patch_id": patch.get("patch_id"),
        "relation_id": patch.get("relation_id"),
        "operation": patch.get("operation"),
        "entity_kind": patch.get("entity_kind"),
        "relation_type": patch.get("relation_type"),
        "source": patch.get("source"),
        "target": patch.get("target"),
        "label": patch.get("label"),
        "validation_status": dict(patch.get("attributes", {})).get("validation_status"),
    }
    return {**delta, "patch": {key: value for key, value in cleaned_patch.items() if value is not None}}


def _node_type(node) -> str:  # noqa: ANN001
    value = getattr(node, "type", None) or getattr(node, "node_type", None) or type(node).__name__
    return getattr(value, "value", str(value))


def _edge_type(edge) -> str:  # noqa: ANN001
    value = getattr(edge, "type", None) or getattr(edge, "edge_type", None) or type(edge).__name__
    return getattr(value, "value", str(value))


def _print_nodes(kg: KnowledgeGraph) -> None:
    print("=== KG Nodes ===")
    for node in kg.list_nodes():
        fields = node.model_dump(mode="json")
        compact = {
            "id": fields.get("id"),
            "type": _node_type(node),
            "label": fields.get("label"),
            "status": fields.get("status"),
            "port": fields.get("port"),
            "service_name": fields.get("service_name"),
            "vulnerability_name": fields.get("vulnerability_name"),
            "validation_status": fields.get("validation_status"),
            "cve": fields.get("cve"),
            "summary": fields.get("summary"),
        }
        print(json.dumps({key: value for key, value in compact.items() if value is not None}, ensure_ascii=False))
    print()


def _print_edges(kg: KnowledgeGraph) -> None:
    print("=== KG Edges ===")
    for edge in kg.list_edges():
        fields = edge.model_dump(mode="json")
        compact = {
            "id": fields.get("id"),
            "type": _edge_type(edge),
            "source": fields.get("source"),
            "target": fields.get("target"),
            "label": fields.get("label"),
        }
        print(json.dumps({key: value for key, value in compact.items() if value is not None}, ensure_ascii=False))
    print()


def _print_cycle_summary(result) -> None:  # noqa: ANN001
    print("=== Cycle Summary ===")
    print(f"planning_success={result.planning.success if result.planning else None}")
    print(f"execution_success={result.execution.success if result.execution else None}")
    print(f"feedback_success={result.feedback.success if result.feedback else None}")
    print(f"selected_task_ids={result.selected_task_ids}")
    print(f"applied_task_ids={result.applied_task_ids}")
    print(f"apply_result_count={len(result.apply_results)}")
    print()


def _print_runtime_metadata(result) -> None:  # noqa: ANN001
    metadata = result.runtime_state.execution.metadata
    print("=== Runtime Phase Checkpoints ===")
    for checkpoint in metadata.get("phase_checkpoints", []):
        compact = {
            "phase": checkpoint.get("phase"),
            "cycle_index": checkpoint.get("cycle_index"),
            "success": checkpoint.get("success"),
            "selected_task_ids": checkpoint.get("selected_task_ids"),
            "applied_task_ids": checkpoint.get("applied_task_ids"),
        }
        print(json.dumps({key: value for key, value in compact.items() if value is not None}, ensure_ascii=False))
    print()

    print("=== Audit Log Tail ===")
    for entry in metadata.get("audit_log", [])[-8:]:
        compact = {
            "event_type": entry.get("event_type"),
            "phase": entry.get("phase"),
            "task_id": entry.get("task_id"),
            "tool": entry.get("tool"),
            "summary": entry.get("summary"),
        }
        print(json.dumps({key: value for key, value in compact.items() if value is not None}, ensure_ascii=False))
    print()


def _build_pipeline(*, host: str, port: int, base_url: str, mode: str) -> AgentPipeline:
    if mode == "nmap":
        nmap_path = _resolve_nmap_path()
        if nmap_path is None:
            raise SystemExit("nmap executable is not available; set AEGRA_NMAP_PATH or add nmap to PATH")
        worker = ViewerNmapReconWorkerAgent(nmap_path=nmap_path)
    else:
        worker = VulhubReconWorkerAgent(base_url=base_url)

    return AgentPipeline(
        agents=[
            VulhubPlannerAgent(host=host, port=port),
            TaskBuilderAgent(),
            SchedulerAgent(),
            worker,
            CriticAgent(),
        ]
    )


def _run_vulnerability_validation(*, state, host: str, port: int, base_url: str):  # noqa: ANN001
    service_id = f"{host}:{port}/tcp"
    task = TaskNode(
        id="task-vulhub-vulnerability-validation",
        label="Validate Struts2 S2-045",
        task_type=TaskType.VULNERABILITY_VALIDATION,
        source_action_id="action-vulhub-vulnerability-validation",
        input_bindings={"host_id": host, "port": port, "service_id": service_id, "target_url": base_url},
        target_refs=[KGGraphRef(graph="kg", ref_id=service_id, ref_type="Service", label=service_id)],
        resource_keys={f"host:{host}"},
    )
    if task.id not in state.execution.tasks:
        state.register_task(TaskRuntime(task_id=task.id, tg_node_id=task.id))
    worker = Struts2VulnerabilityValidationWorker()
    request = worker.build_request(
        task=task,
        operation_id=state.operation_id,
        metadata={"target_url": base_url, "allowlist_hosts": [host, "127.0.0.1", "localhost", "::1"]},
    )
    result = worker.execute_task(request)
    applied = PhaseTwoResultApplier().apply(
        result,
        state,
        kg_ref=ProtocolGraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
    )
    return result, applied


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vulhub graph-generation smoke and print the generated KG.")
    parser.add_argument("--mode", choices=["http", "nmap", "vuln"], default="http", help="Probe mode to use.")
    parser.add_argument("--operation-id", default="op-vulhub-graph-view", help="Runtime operation id.")
    args = parser.parse_args()

    host, port, base_url = _vulhub_target()
    if not _is_tcp_open(host, port):
        raise SystemExit(f"Vulhub target {host}:{port} is not reachable")

    with TemporaryDirectory(prefix="aegra-vulhub-graph-view-") as tmp_dir:
        settings = AppSettings(runtime_store_backend="file", runtime_store_dir=Path(tmp_dir) / "runtime-store")
        orchestrator = AppOrchestrator(
            settings=settings,
            pipeline=_build_pipeline(
                host=host,
                port=port,
                base_url=base_url,
                mode=("http" if args.mode == "vuln" else args.mode),
            ),
        )
        orchestrator.create_operation(args.operation_id)
        state = orchestrator.get_operation_state(args.operation_id)
        state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
        orchestrator.runtime_store.save_state(state)

        result = orchestrator.run_operation_cycle(
            args.operation_id,
            graph_refs=_build_graph_refs(),
            planner_payload={"goal_refs": [], "planning_context": {"top_k": 1, "max_depth": 1}},
        )
        apply_results = list(result.apply_results)
        vuln_result = None
        if args.mode == "vuln":
            vuln_result, vuln_apply_result = _run_vulnerability_validation(
                state=result.runtime_state,
                host=host,
                port=port,
                base_url=base_url,
            )
            apply_results.append(vuln_apply_result)

    kg = _build_kg_from_apply_results(apply_results)

    print("=== Vulhub Graph View ===")
    print(f"target={base_url}")
    print(f"mode={args.mode}")
    print(f"kg_version={kg.version}")
    print(f"node_count={len(kg.list_nodes())}")
    print(f"edge_count={len(kg.list_edges())}")
    print()
    _print_cycle_summary(result)
    if vuln_result is not None:
        print("=== Vulnerability Validation ===")
        print(f"status={vuln_result.outcome_payload.get('status')}")
        print(f"confidence={vuln_result.outcome_payload.get('confidence')}")
        print(f"vulnerability_id={vuln_result.outcome_payload.get('vulnerability_id')}")
        print()
    _print_nodes(kg)
    _print_edges(kg)
    _print_runtime_metadata(result)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
