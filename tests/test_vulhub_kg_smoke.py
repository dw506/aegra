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


def _build_probe_command(*, host: str, port: int, base_url: str) -> list[str]:
    # 中文注释：
    # smoke test 直接请求 Vulhub 暴露的 HTTP 服务，再把真实响应收敛成
    # ReconWorker 当前能消费的统一 JSON 探测结果。
    script = (
        "import json\n"
        "import urllib.error\n"
        "import urllib.request\n"
        f"url = {base_url!r}\n"
        f"host = {host!r}\n"
        f"port = {port}\n"
        "body = ''\n"
        "status = None\n"
        "reachable = False\n"
        "failure_reason = None\n"
        "try:\n"
        "    response = urllib.request.urlopen(url, timeout=5)\n"
        "    status = response.getcode()\n"
        "    body = response.read(256).decode('utf-8', 'ignore')\n"
        "    reachable = True\n"
        "except urllib.error.HTTPError as exc:\n"
        "    status = exc.code\n"
        "    body = exc.read(256).decode('utf-8', 'ignore')\n"
        "    reachable = True\n"
        "    failure_reason = f'http_error:{exc.code}'\n"
        "except Exception as exc:\n"
        "    failure_reason = str(exc)\n"
        "payload = {\n"
        "    'summary': f'http probe {status if status is not None else \"unreachable\"}',\n"
        "    'reachable': reachable,\n"
        "    'confidence': 0.9,\n"
        "    'success': reachable,\n"
        "    'failure_reason': failure_reason,\n"
        "    'entities': [\n"
        "        {'id': host, 'type': 'Host', 'label': host, 'hostname': host, 'status': 'up'},\n"
        "        {\n"
        "            'id': f'{host}:{port}/tcp',\n"
        "            'type': 'Service',\n"
        "            'label': f'http-{port}',\n"
        "            'host_id': host,\n"
        "            'port': port,\n"
        "            'protocol': 'tcp',\n"
        "            'service_name': 'http',\n"
        "            'banner': body[:80] or 'http',\n"
        "            'status': 'open',\n"
        "        },\n"
        "    ],\n"
        "    'relations': [\n"
        "        {\n"
        "            'type': 'HOSTS',\n"
        "            'source': host,\n"
        "            'target': f'{host}:{port}/tcp',\n"
        "            'attributes': {'port': port, 'protocol': 'tcp', 'state': 'open'},\n"
        "        }\n"
        "    ],\n"
        "    'service': {\n"
        "        'id': f'{host}:{port}/tcp',\n"
        "        'port': port,\n"
        "        'protocol': 'tcp',\n"
        "        'banner': body[:80] or 'http',\n"
        "    },\n"
        "    'runtime_hints': {'reachable': reachable, 'http_status': status, 'target_url': url},\n"
        "}\n"
        "if not reachable:\n"
        "    payload['entities'] = []\n"
        "    payload['relations'] = []\n"
        "    payload['service'] = {}\n"
        "print(json.dumps(payload))\n"
    )
    return [sys.executable, "-c", script]


def _clean_entity_delta(delta: dict) -> dict | None:
    if delta.get("delta_type") != "upsert_entity":
        return None

    patch = dict(delta.get("patch", {}))
    entity_type = str(patch.get("entity_type") or "").lower()
    attributes = dict(patch.get("attributes", {}))

    # 中文注释：
    # 这里有意只保留 Host / Service 的最小 schema 字段，避免当前 smoke test
    # 被 Evidence / Observation / provenance 扩展字段的模型兼容问题干扰。
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
    if str(patch.get("relation_type") or "").upper() != "HOSTS":
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
        metadata={"tool_command": _build_probe_command(host=host, port=port, base_url=base_url)},
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

    assert result.status.value == "succeeded"
    assert entity_deltas
    assert relation_deltas
    assert [node.id for node in kg.list_nodes(type="Host")] == [host]
    assert [node.id for node in kg.list_nodes(type="Service")] == [service_id]
    assert [edge.id for edge in kg.list_edges(type="HOSTS")] == [f"hosts::{host}::{service_id}"]
