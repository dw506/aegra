from __future__ import annotations

import os
import socket
import sys
from urllib.parse import urlparse

import pytest

from src.core.agents.agent_protocol import GraphRef as ProtocolGraphRef, GraphScope
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import GraphRef
from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntime
from src.core.models.tg import TaskNode, TaskType
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.workers.recon_worker import ReconWorker


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _vulhub_target() -> tuple[str, int, str]:
    base_url = os.getenv("AEGRA_VULHUB_BASE_URL", "http://127.0.0.1:8080/").strip()
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.hostname or parsed.port is None:
        raise ValueError("AEGRA_VULHUB_BASE_URL must include scheme, host and port")
    return parsed.hostname, parsed.port, base_url


def _is_tcp_open(host: str, port: int, *, timeout_sec: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _build_runtime_state(host: str) -> RuntimeState:
    state = RuntimeState(
        operation_id="op-vulhub-smoke",
        execution=OperationRuntime(operation_id="op-vulhub-smoke"),
    )
    state.register_task(TaskRuntime(task_id="task-1", tg_node_id="task-1"))
    state.execution.metadata["runtime_policy"] = {"sensitive_task_types": []}
    state.execution.metadata["control_plane"] = {"audit_redaction_enabled": True}
    return state


def _build_task(host: str, port: int) -> TaskNode:
    return TaskNode(
        id="task-1",
        label="Validate Vulhub Service",
        task_type=TaskType.SERVICE_VALIDATION,
        source_action_id="action-1",
        input_bindings={"host_id": host, "port": port},
        target_refs=[GraphRef(graph="kg", ref_id=host, ref_type="Host", label=host)],
        resource_keys={f"host:{host}"},
    )


def _build_nmap_metadata(port: int) -> dict[str, object]:
    # 中文注释：
    # 显式要求 ReconWorker 使用 NmapAdapter 做真实端口探测。
    return {
        "probe_adapter": "nmap",
        "service_port": port,
        "port": port,
        "tool_timeout_sec": 30,
    }



def _clean_entity_delta(delta: dict) -> dict | None:
    if delta.get("delta_type") != "upsert_entity":
        return None

    patch = dict(delta.get("patch", {}))
    entity_type = str(patch.get("entity_type") or "").lower()
    attributes = dict(patch.get("attributes", {}))

    # 中文注释：
    # 这里只保留 KG 模型本身接受的最小字段，避免 result applier 产生的
    # evidence_chain / source_task_id / content_ref 等扩展字段触发 extra=forbid。
    if entity_type == "host":
        cleaned_patch = {
            "patch_id": patch.get("patch_id"),
            "entity_id": patch.get("entity_id"),
            "operation": patch.get("operation"),
            "entity_kind": patch.get("entity_kind"),
            "entity_type": patch.get("entity_type"),
            "label": attributes.get("label") or patch.get("entity_id"),
            "hostname": attributes.get("hostname"),
            "status": attributes.get("status"),
        }
    elif entity_type == "service":
        cleaned_patch = {
            "patch_id": patch.get("patch_id"),
            "entity_id": patch.get("entity_id"),
            "operation": patch.get("operation"),
            "entity_kind": patch.get("entity_kind"),
            "entity_type": patch.get("entity_type"),
            "label": attributes.get("label") or patch.get("entity_id"),
            "host_id": attributes.get("host_id"),
            "port": attributes.get("port"),
            "protocol": attributes.get("protocol"),
            "service_name": attributes.get("service_name"),
            "banner": attributes.get("banner"),
            "status": attributes.get("status"),
        }
    elif entity_type == "observation":
        cleaned_patch = {
            "patch_id": patch.get("patch_id"),
            "entity_id": patch.get("entity_id"),
            "operation": patch.get("operation"),
            "entity_kind": patch.get("entity_kind"),
            "entity_type": patch.get("entity_type"),
            "label": patch.get("label"),
            "observation_kind": attributes.get("observation_kind"),
            "summary": attributes.get("summary"),
        }
    elif entity_type == "evidence":
        cleaned_patch = {
            "patch_id": patch.get("patch_id"),
            "entity_id": patch.get("entity_id"),
            "operation": patch.get("operation"),
            "entity_kind": patch.get("entity_kind"),
            "entity_type": patch.get("entity_type"),
            "label": patch.get("label"),
            "evidence_kind": attributes.get("evidence_kind"),
            "summary": attributes.get("summary"),
        }
    else:
        return None

    return {
        **delta,
        "patch": {key: value for key, value in cleaned_patch.items() if value is not None},
    }


def _clean_relation_delta(delta: dict) -> dict | None:
    if delta.get("delta_type") != "upsert_relation":
        return None

    patch = dict(delta.get("patch", {}))
    relation_type = str(patch.get("relation_type") or "").upper()
    if relation_type not in {"HOSTS", "SUPPORTED_BY"}:
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
    }
    return {
        **delta,
        "patch": {key: value for key, value in cleaned_patch.items() if value is not None},
    }


def test_vulhub_http_service_builds_minimal_kg_chain() -> None:
    if not _env_flag("AEGRA_RUN_VULHUB_SMOKE"):
        pytest.skip("set AEGRA_RUN_VULHUB_SMOKE=1 to enable the Vulhub smoke test")

    host, port, base_url = _vulhub_target()
    if not _is_tcp_open(host, port):
        pytest.skip(f"Vulhub target {host}:{port} is not reachable")

    state = _build_runtime_state(host)
    task = _build_task(host, port)
    worker = ReconWorker()
    request = worker.build_request(
    task=task,
    operation_id=state.operation_id,
    metadata=_build_nmap_metadata(port),
)


    result = worker.execute_task(request)
    applied = PhaseTwoResultApplier().apply(
        result,
        state,
        kg_ref=ProtocolGraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
    )

    entity_deltas = [
        cleaned
        for delta in applied.kg_state_deltas
        if (cleaned := _clean_entity_delta(delta)) is not None
    ]
    relation_deltas = [
        cleaned
        for delta in applied.kg_state_deltas
        if (cleaned := _clean_relation_delta(delta)) is not None
    ]

    kg = KnowledgeGraph()
    kg.apply_patch_batch(
        {
            "patch_batch_id": "vulhub-entities",
            "base_kg_version": 0,
            "state_deltas": entity_deltas,
            "metadata": {},
        }
    )
    kg.apply_patch_batch(
        {
            "patch_batch_id": "vulhub-relations",
            "base_kg_version": kg.version,
            "state_deltas": relation_deltas,
            "metadata": {},
        }
    )

    service_id = f"{host}:{port}/tcp"

    host_nodes = [node.id for node in kg.list_nodes(type="Host")]
    service_nodes = [node.id for node in kg.list_nodes(type="Service")]
    observation_nodes = [node.id for node in kg.list_nodes(type="Observation")]
    evidence_nodes = [node.id for node in kg.list_nodes(type="Evidence")]
    hosts_edges = [edge.id for edge in kg.list_edges(type="HOSTS")]
    supported_by_edges = [edge.id for edge in kg.list_edges(type="SUPPORTED_BY")]

    assert result.status.value == "succeeded"
    assert entity_deltas
    assert relation_deltas
    assert host_nodes == [host]
    assert service_nodes == [service_id]
    assert hosts_edges == [f"hosts::{host}::{service_id}"]
    assert observation_nodes
    assert evidence_nodes
    assert supported_by_edges
