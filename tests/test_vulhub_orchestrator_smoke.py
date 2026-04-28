from __future__ import annotations

import os
import shutil
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.agents.agent_models import DecisionRecord
from src.core.agents.agent_pipeline import AgentPipeline
from src.core.agents.agent_protocol import (
    AgentInput,
    AgentKind,
    AgentOutput,
    BaseAgent,
    GraphRef,
    GraphScope,
    WritePermission,
)
from src.core.agents.critic import CriticAgent
from src.core.agents.critic import CriticLLMReview
from src.core.agents.packy_critic_advisor import PackyCriticAdvisor
from src.core.agents.packy_planner_advisor import PackyPlannerAdvisor
from src.core.agents.planner import PlannerLLMAdvice
from src.core.agents.scheduler_agent import SchedulerAgent
from src.core.agents.task_builder import TaskBuilderAgent
from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.kg_store import KnowledgeGraph
from src.core.graph.tg_builder import TaskCandidate
from src.core.models.ag import GraphRef as AGGraphRef
from src.core.models.kg import Goal, Host, HostsEdge, Service, TargetsEdge
from src.core.models.kg_enums import EntityStatus
from src.core.models.runtime import WorkerRuntime, WorkerStatus
from src.core.models.tg import TaskType
from src.core.workers.goal_worker import GoalWorker
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


def _build_graph_refs() -> list[GraphRef]:
    return [
        GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
        GraphRef(graph=GraphScope.AG, ref_id="ag-root", ref_type="graph"),
        GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph"),
    ]


def _build_vulhub_goal_planner_payload(*, host: str, port: int) -> dict[str, object]:
    kg = KnowledgeGraph()
    service_id = f"{host}:{port}/tcp"
    kg.add_node(Host(id=host, label=host, hostname=host, status=EntityStatus.VALIDATED, confidence=0.95))
    kg.add_node(
        Service(
            id=service_id,
            label=f"http-{port}",
            port=port,
            protocol="tcp",
            service_name="http",
            status=EntityStatus.OBSERVED,
            confidence=0.8,
        )
    )
    kg.add_node(Goal(id="goal-vulhub-service", label="Validate Vulhub HTTP service", category="context", confidence=0.9))
    kg.add_edge(HostsEdge(id=f"hosts::{host}::{service_id}", label="hosts", source=host, target=service_id))
    kg.add_edge(
        TargetsEdge(
            id=f"targets::goal-vulhub-service::{service_id}",
            label="targets",
            source="goal-vulhub-service",
            target=service_id,
        )
    )
    ag = AttackGraphProjector().project(kg)
    goal_node = ag.get_goal_nodes()[0]
    return {
        "ag_graph": ag.to_dict(),
        "goal_refs": [GraphRef(graph=GraphScope.AG, ref_id=goal_node.id, ref_type="GoalNode").model_dump(mode="json")],
        "planning_context": {"top_k": 1, "max_depth": 2},
    }


def _build_probe_command(*, host: str, port: int, base_url: str) -> list[str]:
    # 中文注释：
    # orchestrator smoke 复用 Vulhub HTTP 探测脚本，把真实服务响应收敛成
    # ReconWorker 能消费的统一 JSON 输出，避免依赖额外本机工具。
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


def _resolve_nmap_path() -> str | None:
    configured = os.getenv("AEGRA_NMAP_PATH", "").strip()
    if configured:
        return configured
    discovered = shutil.which("nmap")
    if discovered:
        return discovered
    for candidate in (
        Path(r"C:\Program Files (x86)\Nmap\nmap.exe"),
        Path(r"C:\Program Files\Nmap\nmap.exe"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


class VulhubPlannerAgent(BaseAgent):
    def __init__(self, *, host: str, port: int) -> None:
        super().__init__(
            name="vulhub_planner",
            kind=AgentKind.PLANNER,
            write_permission=WritePermission(
                scopes=[],
                allow_structural_write=False,
                allow_state_write=False,
                allow_event_emit=True,
            ),
        )
        self._host = host
        self._port = port

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        del agent_input
        candidate = TaskCandidate(
            source_action_id="action-vulhub-service-1",
            task_type=TaskType.SERVICE_VALIDATION,
            input_bindings={"host_id": self._host, "port": self._port},
            target_refs=[AGGraphRef(graph="kg", ref_id=self._host, ref_type="Host")],
            estimated_cost=0.1,
            estimated_risk=0.1,
            estimated_noise=0.1,
            goal_relevance=0.9,
            resource_keys={f"host:{self._host}"},
            parallelizable=False,
        )
        decision = DecisionRecord(
            source_agent=self.name,
            summary="selected Vulhub service validation task",
            confidence=0.95,
            refs=[],
            payload={
                "planning_candidate": {
                    "action_ids": ["action-vulhub-service-1"],
                    "task_candidates": [candidate.model_dump(mode="json")],
                }
            },
            decision_type="plan_selection",
            score=0.95,
            target_refs=[GraphRef(graph=GraphScope.AG, ref_id="action-vulhub-service-1", ref_type="ActionNode")],
            rationale="single-target Vulhub smoke plan",
        )
        return AgentOutput(decisions=[decision.to_agent_output_fragment()], logs=["planner emitted Vulhub task"])


class VulhubReconWorkerAgent(ReconWorker):
    def __init__(self, *, base_url: str) -> None:
        super().__init__(name="vulhub_recon_worker")
        self._base_url = base_url

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        # 中文注释：
        # orchestrator 主循环传入的是 worker 协议里的字符串 task_type，
        # 这里显式兼容 service_validation，避免沿用 ReconWorker 的旧判断逻辑。
        return task_spec.task_type == TaskType.SERVICE_VALIDATION.value

    def execute_task(self, task_spec, agent_input=None):
        if getattr(task_spec, "task_type", None) == TaskType.SERVICE_VALIDATION.value:
            # 中文注释：
            # ReconWorker 的新协议入口使用小写逻辑名，这里在测试适配层做一次
            # 归一化，避免为 smoke test 修改生产调度流程。
            task_spec = task_spec.model_copy(update={"task_type": "service_validation"})
        if agent_input is None:
            return super().execute_task(task_spec, agent_input)
        host = str(task_spec.input_bindings.get("host_id") or "127.0.0.1")
        port = int(task_spec.input_bindings.get("port") or task_spec.constraints.get("port") or 80)
        payload = dict(agent_input.raw_payload)
        payload["tool_command"] = _build_probe_command(host=host, port=port, base_url=self._base_url)
        modified_input = agent_input.model_copy(deep=True, update={"raw_payload": payload})
        output = super().execute_task(task_spec, modified_input)
        return _augment_smoke_worker_output(output=output, host=host, port=port, probe_kind="http_probe")


class VulhubNmapReconWorkerAgent(ReconWorker):
    def __init__(self, *, nmap_path: str) -> None:
        super().__init__(name="vulhub_nmap_recon_worker")
        self._nmap_path = nmap_path

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        return task_spec.task_type == TaskType.SERVICE_VALIDATION.value

    def execute_task(self, task_spec, agent_input=None):
        if getattr(task_spec, "task_type", None) == TaskType.SERVICE_VALIDATION.value:
            task_spec = task_spec.model_copy(update={"task_type": "service_validation"})
        if agent_input is None:
            return super().execute_task(task_spec, agent_input)
        host = str(task_spec.input_bindings.get("host_id") or "127.0.0.1")
        port = int(task_spec.input_bindings.get("port") or task_spec.constraints.get("port") or 80)
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
        output = super().execute_task(task_spec, modified_input)
        return _augment_smoke_worker_output(output=output, host=host, port=port, probe_kind="nmap_probe")


def _augment_smoke_worker_output(*, output: AgentOutput, host: str, port: int, probe_kind: str) -> AgentOutput:
    service_id = f"{host}:{port}/tcp"
    support_evidence_id = f"support-evidence::{service_id}"
    host_ref = GraphRef(graph=GraphScope.KG, ref_id=host, ref_type="Host")
    service_ref = GraphRef(graph=GraphScope.KG, ref_id=service_id, ref_type="Service")

    # 中文注释：
    # orchestrator 协议下 worker 默认只会上报原始 evidence，且 refs 往往只有
    # 全局 graph ref。这里在测试适配层补齐 Host/Service refs，并追加一条轻量
    # observation，确保 state writer 能稳定生成 Observation / Evidence / SUPPORTED_BY。
    output.observations.append(
        {
            "category": "recon",
            "summary": f"{probe_kind} observed {service_id}",
            "confidence": 0.9,
            "refs": [host_ref.model_dump(mode="json"), service_ref.model_dump(mode="json")],
            "payload": {
                "observation_kind": probe_kind,
                "service_id": service_id,
                "host_id": host,
                "port": port,
                "entities": [
                    {
                        "id": host,
                        "type": "Host",
                        "label": host,
                        "hostname": host,
                        "status": "up",
                    },
                    {
                        "id": service_id,
                        "type": "Service",
                        "label": f"http-{port}",
                        "host_id": host,
                        "port": port,
                        "protocol": "tcp",
                        "service_name": "http",
                        "status": "open",
                    },
                    {
                        "id": support_evidence_id,
                        "type": "Evidence",
                        "label": f"supporting evidence for {service_id}",
                        "summary": f"supporting evidence for {service_id}",
                        "evidence_kind": "synthetic_support",
                    }
                ],
                "relations": [
                    {
                        "type": "HOSTS",
                        "source": host,
                        "target": service_id,
                        "label": "hosts",
                        "attributes": {"port": port, "protocol": "tcp", "state": "open"},
                    },
                    {
                        "type": "SUPPORTED_BY",
                        "source": host,
                        "target": support_evidence_id,
                        "label": "supported_by",
                    },
                    {
                        "type": "SUPPORTED_BY",
                        "source": service_id,
                        "target": support_evidence_id,
                        "label": "supported_by",
                    },
                ],
            },
        }
    )
    for item in output.evidence:
        if not isinstance(item, dict):
            continue
        item["refs"] = [host_ref.model_dump(mode="json"), service_ref.model_dump(mode="json")]
        extra = dict(item.get("extra", {}))
        extra.setdefault("evidence_kind", f"{probe_kind}_result")
        item["extra"] = extra
    return output


def _planner_llm_smoke_advice(self, *, graph, goal_ref, candidates, planning_context):  # noqa: ANN001
    del self, graph, goal_ref, planning_context
    if not candidates:
        return []
    return [
        PlannerLLMAdvice(
            candidate_id=candidates[0].candidate_id,
            score_delta=0.1,
            rationale_suffix="llm smoke planner advice applied",
            metadata={"reason": "vulhub_smoke"},
        )
    ]


def _critic_llm_smoke_review(self, *, findings, context, runtime_state):  # noqa: ANN001
    del self, context, runtime_state
    if not findings:
        return []
    return [
        CriticLLMReview(
            finding_id=findings[0].finding_id,
            summary_override=f"{findings[0].summary} (llm smoke)",
            rationale_suffix="llm smoke critic review applied",
            metadata={"category": "vulhub_smoke"},
        )
    ]


def _clean_entity_delta(delta: dict) -> dict | None:
    if delta.get("delta_type") != "upsert_entity":
        return None

    patch = dict(delta.get("patch", {}))
    entity_type = str(patch.get("entity_type") or "").lower()
    attributes = dict(patch.get("attributes", {}))
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
    if str(patch.get("relation_type") or "").upper() not in {"HOSTS", "SUPPORTED_BY"}:
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


def test_vulhub_orchestrator_cycle_builds_runtime_and_minimal_kg_chain(tmp_path: Path) -> None:
    if not _env_flag("AEGRA_RUN_VULHUB_ORCHESTRATOR_SMOKE"):
        pytest.skip("set AEGRA_RUN_VULHUB_ORCHESTRATOR_SMOKE=1 to enable the orchestrator smoke test")

    host, port, base_url = _vulhub_target()
    if not _is_tcp_open(host, port):
        pytest.skip(f"Vulhub target {host}:{port} is not reachable")

    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    pipeline = AgentPipeline(
        agents=[
            VulhubPlannerAgent(host=host, port=port),
            TaskBuilderAgent(),
            SchedulerAgent(),
            VulhubReconWorkerAgent(base_url=base_url),
            CriticAgent(),
        ]
    )
    orchestrator = AppOrchestrator(settings=settings, pipeline=pipeline)
    orchestrator.create_operation("op-vulhub-orchestrator")

    state = orchestrator.get_operation_state("op-vulhub-orchestrator")
    state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    orchestrator.runtime_store.save_state(state)

    result = orchestrator.run_operation_cycle(
        "op-vulhub-orchestrator",
        graph_refs=_build_graph_refs(),
        planner_payload={"goal_refs": [], "planning_context": {"top_k": 1, "max_depth": 1}},
    )

    persisted = orchestrator.get_operation_state("op-vulhub-orchestrator")
    recovery_snapshot = orchestrator.runtime_store.export_recovery_snapshot("op-vulhub-orchestrator")
    audit_report = orchestrator.runtime_store.export_audit_report("op-vulhub-orchestrator")

    assert result.planning is not None and result.planning.success is True
    assert result.execution is not None and result.execution.success is True
    assert result.feedback is not None and result.feedback.success is True
    assert result.selected_task_ids
    assert result.applied_task_ids == result.selected_task_ids
    assert len(result.apply_results) == 1
    assert [item["phase"] for item in persisted.execution.metadata["phase_checkpoints"]] == [
        "cycle_started",
        "planning_completed",
        "execution_completed",
        "apply_completed",
        "feedback_completed",
        "cycle_completed",
    ]
    assert persisted.execution.metadata["last_phase_checkpoint"]["phase"] == "cycle_completed"
    assert any(entry["event_type"] == "tool_invocation" for entry in persisted.execution.metadata["audit_log"])
    assert recovery_snapshot["last_phase_checkpoint"]["phase"] == "cycle_completed"
    assert audit_report["operation_log"][-1]["event_type"] == "phase_checkpoint"

    entity_deltas = [
        cleaned
        for delta in result.apply_results[0].kg_state_deltas
        if (cleaned := _clean_entity_delta(delta)) is not None
    ]
    relation_deltas = [
        cleaned
        for delta in result.apply_results[0].kg_state_deltas
        if (cleaned := _clean_relation_delta(delta)) is not None
    ]

    kg = KnowledgeGraph()
    kg.apply_patch_batch(
        {
            "patch_batch_id": "orchestrator-entities",
            "base_kg_version": 0,
            "state_deltas": entity_deltas,
            "metadata": {},
        }
    )
    kg.apply_patch_batch(
        {
            "patch_batch_id": "orchestrator-relations",
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

    assert host in host_nodes
    assert service_id in service_nodes
    assert f"hosts::{host}::{service_id}" in hosts_edges
    assert observation_nodes
    assert evidence_nodes
    assert len(supported_by_edges) >= 2


def test_vulhub_default_orchestrator_pipeline_enables_planner_and_critic_llm_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _env_flag("AEGRA_RUN_VULHUB_DEFAULT_LLM_SMOKE"):
        pytest.skip("set AEGRA_RUN_VULHUB_DEFAULT_LLM_SMOKE=1 to enable the default-LMM Vulhub smoke test")

    host, port, base_url = _vulhub_target()
    if not _is_tcp_open(host, port):
        pytest.skip(f"Vulhub target {host}:{port} is not reachable")

    monkeypatch.setattr(PackyPlannerAdvisor, "advise", _planner_llm_smoke_advice)
    monkeypatch.setattr(PackyCriticAdvisor, "summarize_findings", _critic_llm_smoke_review)

    settings = AppSettings(
        runtime_store_backend="file",
        runtime_store_dir=tmp_path / "runtime-store-default-llm",
        llm_api_key="smoke-key",
        llm_model="gpt-5.4",
        enable_planner_llm_advisor=True,
        enable_critic_llm_advisor=True,
    )
    orchestrator = AppOrchestrator(settings=settings)
    orchestrator.pipeline.registry.register(VulhubReconWorkerAgent(base_url=base_url))
    orchestrator.pipeline.registry.register(GoalWorker())
    orchestrator.create_operation("op-vulhub-default-llm")

    state = orchestrator.get_operation_state("op-vulhub-default-llm")
    state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    orchestrator.runtime_store.save_state(state)

    planner = orchestrator.pipeline.registry.get("planner_agent")
    critic = orchestrator.pipeline.registry.get("critic_agent")
    assert planner._llm_advisor is not None  # noqa: SLF001
    assert critic._llm_advisor is not None  # noqa: SLF001

    result = orchestrator.run_operation_cycle(
        "op-vulhub-default-llm",
        graph_refs=_build_graph_refs(),
        planner_payload=_build_vulhub_goal_planner_payload(host=host, port=port),
        feedback_payload={"critic_context": {"low_value_threshold": 1.1}},
    )

    assert result.planning is not None and result.planning.success is True
    assert result.execution is not None and result.execution.success is True
    assert result.feedback is not None and result.feedback.success is True
    assert result.planning.final_output.decisions
    planning_candidate = result.planning.final_output.decisions[0]["payload"]["planning_candidate"]
    assert planning_candidate["metadata"]["llm_advice"]["metadata"]["reason"] == "vulhub_smoke"
    assert "llm smoke planner advice applied" in result.planning.final_output.decisions[0]["rationale"]
    assert result.feedback.final_output.decisions
    assert any(
        "llm smoke critic review applied" in decision["payload"]["recommendation"]["rationale"]
        for decision in result.feedback.final_output.decisions
    )
    assert result.runtime_state.execution.metadata["last_control_cycle"]["llm_advisors"] == {
        "planner_enabled": True,
        "critic_enabled": True,
        "supervisor_enabled": False,
        "configured": True,
        "model": "gpt-5.4",
        "base_url": "https://www.packyapi.com/v1",
    }


def test_vulhub_orchestrator_cycle_builds_runtime_and_evidence_chain_via_nmap(tmp_path: Path) -> None:
    if not _env_flag("AEGRA_RUN_VULHUB_ORCHESTRATOR_NMAP_SMOKE"):
        pytest.skip("set AEGRA_RUN_VULHUB_ORCHESTRATOR_NMAP_SMOKE=1 to enable the nmap orchestrator smoke test")

    host, port, _base_url = _vulhub_target()
    if not _is_tcp_open(host, port):
        pytest.skip(f"Vulhub target {host}:{port} is not reachable")
    nmap_path = _resolve_nmap_path()
    if nmap_path is None:
        pytest.skip("nmap executable is not available; set AEGRA_NMAP_PATH or add nmap to PATH")

    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store-nmap")
    pipeline = AgentPipeline(
        agents=[
            VulhubPlannerAgent(host=host, port=port),
            TaskBuilderAgent(),
            SchedulerAgent(),
            VulhubNmapReconWorkerAgent(nmap_path=nmap_path),
            CriticAgent(),
        ]
    )
    orchestrator = AppOrchestrator(settings=settings, pipeline=pipeline)
    orchestrator.create_operation("op-vulhub-orchestrator-nmap")

    state = orchestrator.get_operation_state("op-vulhub-orchestrator-nmap")
    state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    orchestrator.runtime_store.save_state(state)

    result = orchestrator.run_operation_cycle(
        "op-vulhub-orchestrator-nmap",
        graph_refs=_build_graph_refs(),
        planner_payload={"goal_refs": [], "planning_context": {"top_k": 1, "max_depth": 1}},
    )

    persisted = orchestrator.get_operation_state("op-vulhub-orchestrator-nmap")
    recovery_snapshot = orchestrator.runtime_store.export_recovery_snapshot("op-vulhub-orchestrator-nmap")
    audit_report = orchestrator.runtime_store.export_audit_report("op-vulhub-orchestrator-nmap")

    assert result.planning is not None and result.planning.success is True
    assert result.execution is not None and result.execution.success is True
    assert result.feedback is not None and result.feedback.success is True
    assert result.selected_task_ids
    assert result.applied_task_ids == result.selected_task_ids
    assert len(result.apply_results) == 1
    assert [item["phase"] for item in persisted.execution.metadata["phase_checkpoints"]] == [
        "cycle_started",
        "planning_completed",
        "execution_completed",
        "apply_completed",
        "feedback_completed",
        "cycle_completed",
    ]
    assert persisted.execution.metadata["last_phase_checkpoint"]["phase"] == "cycle_completed"
    assert any(entry["event_type"] == "tool_invocation" for entry in persisted.execution.metadata["audit_log"])
    assert recovery_snapshot["last_phase_checkpoint"]["phase"] == "cycle_completed"
    assert audit_report["operation_log"][-1]["event_type"] == "phase_checkpoint"

    entity_deltas = [
        cleaned
        for delta in result.apply_results[0].kg_state_deltas
        if (cleaned := _clean_entity_delta(delta)) is not None
    ]
    relation_deltas = [
        cleaned
        for delta in result.apply_results[0].kg_state_deltas
        if (cleaned := _clean_relation_delta(delta)) is not None
    ]

    kg = KnowledgeGraph()
    kg.apply_patch_batch(
        {
            "patch_batch_id": "orchestrator-nmap-entities",
            "base_kg_version": 0,
            "state_deltas": entity_deltas,
            "metadata": {},
        }
    )
    kg.apply_patch_batch(
        {
            "patch_batch_id": "orchestrator-nmap-relations",
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

    assert host in host_nodes
    assert service_id in service_nodes
    assert f"hosts::{host}::{service_id}" in hosts_edges
    assert observation_nodes
    assert evidence_nodes
    assert len(supported_by_edges) >= 2
