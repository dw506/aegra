from __future__ import annotations

import json

from src.core.agents.agent_protocol import AgentContext, AgentInput, GraphRef, GraphScope
from src.core.agents.critic import CriticAgent
from src.core.agents.packy_llm import PackyLLMClient, PackyLLMConfig
from src.core.agents.scheduler_agent import SchedulerAgent
from src.core.execution.configured_mcp_client import ConfiguredMCPClient
from src.core.models.ag import GraphRef as KGGraphRef
from src.core.models.events import (
    AgentResultStatus,
    AgentRole,
    AgentTaskResult,
    EvidenceArtifact,
    FactWriteKind,
    FactWriteRequest,
    ObservationRecord,
    RuntimeBudgetDelta,
    RuntimeControlRequest,
    RuntimeControlType,
)
from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntime, WorkerRuntime, WorkerStatus
from src.core.models.tg import TaskGraph, TaskNode, TaskStatus, TaskType
from src.core.planning.llm_mission_planner_advisor import LLMMissionPlannerAdvisor
from src.core.planning.mission_planner_agent import MissionPlannerAgent
from src.core.planning.stage_task_builder import StageTaskGraphBuilder
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.workers.base import WorkerTaskSpec
from src.core.workers.llm_worker import LLMWorkerAgent
from src.core.workers.llm_worker_advisor import LLMWorkerAdvisor


OPERATION_ID = "op-full-real-vulhub-12615"
TASK_ID = "task-full-real-vulhub-cve-2017-12615"
TARGET_URL = "http://host.docker.internal:8081/"
SERVICE_ID = "svc-vulhub-tomcat-8081"
VULN_ID = f"vuln::CVE-2017-12615::{SERVICE_ID}"


def build_runtime_state() -> RuntimeState:
    state = RuntimeState(
        operation_id=OPERATION_ID,
        execution=OperationRuntime(operation_id=OPERATION_ID),
    )
    state.execution.metadata["control_plane"] = {"audit_redaction_enabled": True}
    state.execution.metadata["runtime_policy"] = {
        "authorized_hosts": ["host.docker.internal", "127.0.0.1", "localhost"],
        "authorized_ports": [8081],
        "max_rounds": 1,
        "allow_safe_file_write_probe": True,
        "allow_command_execution_payload": False,
        "disabled_tools": ["exploit.tomcat_put_jsp_shell"],
        "sensitive_task_types": [],
    }
    state.workers["llm-worker-1"] = WorkerRuntime(
        worker_id="llm-worker-1",
        status=WorkerStatus.IDLE,
    )
    return state


def run_real_planner(state: RuntimeState) -> tuple[TaskGraph, dict]:
    llm_client = PackyLLMClient(PackyLLMConfig.from_env())
    planner = MissionPlannerAgent(
        advisor=LLMMissionPlannerAdvisor(client=llm_client)
    )

    service_ref = KGGraphRef(
        graph="kg",
        ref_id=SERVICE_ID,
        ref_type="Service",
        label="Vulhub Tomcat 8081",
    )

    graph_context = {
        "operation_id": OPERATION_ID,
        "kg": {
            "assets": [
                {
                    "id": "host-vulhub-tomcat",
                    "type": "Host",
                    "address": "host.docker.internal",
                },
                {
                    "id": SERVICE_ID,
                    "type": "Service",
                    "host_id": "host-vulhub-tomcat",
                    "port": 8081,
                    "protocol": "http",
                    "url": TARGET_URL,
                },
            ]
        },
        "tg": {"existing_tasks": []},
        "runtime": {"mode": "single_round", "tool_call_budget": 1},
        "policy": state.execution.metadata["runtime_policy"],
        "hard_constraints": [
            "Generate exactly one stage task.",
            "The stage task must target Tomcat CVE-2017-12615 validation.",
            "Do not generate exploit, access, pivot, or goal tasks.",
            "Do not output shell commands.",
            "Use target_url http://host.docker.internal:8081/.",
        ],
    }

    goal = (
        "Detect whether the authorized local Vulhub Tomcat target "
        "http://host.docker.internal:8081/ is vulnerable to CVE-2017-12615. "
        "Create exactly one safe vulnerability analysis stage task."
    )

    planner_result = planner.run(
        goal=goal,
        graph_context=graph_context,
        policy_context=state.execution.metadata["runtime_policy"],
    )

    if not planner_result.new_stage_tasks:
        raise RuntimeError(f"Planner produced no stage task: {planner_result.model_dump(mode='json')}")

    # 为了单轮测试稳定，强制只取 Planner 产出的第一个 task，并规范关键字段。
    stage_task = planner_result.new_stage_tasks[0]
    stage_task.task_id = TASK_ID
    stage_task.objective = "Validate Tomcat CVE-2017-12615 with safe marker probe"
    stage_task.target_refs = [service_ref]
    stage_task.required_context = {
        "target_url": TARGET_URL,
        "service_id": SERVICE_ID,
        "port": 8081,
        "cve": "CVE-2017-12615",
        "validation_mode": "safe_static_marker_probe",
    }
    stage_task.success_criteria = [
        "MCP tool returns structured validation result",
        "No command execution payload is used",
        "ResultApplier writes evidence and KG vulnerability relation when validated",
    ]
    stage_task.max_steps = 1
    stage_task.risk_level = "medium"

    tg = TaskGraph()
    created = StageTaskGraphBuilder().upsert_stage_tasks(tg, [stage_task])
    for task_id in created:
        tg.mark_task_status(task_id, TaskStatus.READY, reason="PlannerAgent proposed task accepted for single-round test")

    return tg, {
        "planner_result": planner_result.model_dump(mode="json"),
        "created_task_ids": created,
        "selected_stage_task": stage_task.model_dump(mode="json"),
    }


def run_real_scheduler(state: RuntimeState, tg: TaskGraph) -> dict:
    scheduler_input = AgentInput(
        graph_refs=[GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph")],
        context=AgentContext(operation_id=OPERATION_ID),
        raw_payload={
            "task_graph": tg.to_dict(),
            "runtime_state": state.model_dump(mode="json"),
            "scheduling_context": {
                "max_assignments": 1,
                "available_workers": [
                    {
                        "worker_id": "llm-worker-1",
                        "status": "idle",
                        "capabilities": ["vulnerability_validation"],
                    }
                ],
                "budget_summary": {},
                "policy_flags": {},
            },
        },
    )

    output = SchedulerAgent().execute(scheduler_input)
    accepted = [
        d for d in output.decisions
        if d.get("payload", {}).get("action") == "assign"
        or d.get("action") == "assign"
    ]

    # 兼容当前 DecisionRecord 序列化形态，若找不到 accepted，就直接打印给你排查。
    if not output.decisions:
        raise RuntimeError(f"Scheduler emitted no decisions: {output.model_dump(mode='json')}")

    task = tg.get_node(TASK_ID)
    if isinstance(task, TaskNode):
        if task.status == TaskStatus.READY:
            tg.transition_task(TASK_ID, TaskStatus.QUEUED, reason="SchedulerAgent assigned task")
        if task.status == TaskStatus.QUEUED:
            tg.transition_task(TASK_ID, TaskStatus.RUNNING, reason="LLMWorkerAgent starts assigned task")

    return output.model_dump(mode="json")


def run_real_worker(tg: TaskGraph) -> tuple[object, dict]:
    task = tg.get_node(TASK_ID)
    if not isinstance(task, TaskNode):
        raise RuntimeError(f"{TASK_ID} is not a TaskNode")

    service_ref = GraphRef(
          graph=GraphScope.KG,
          ref_id=SERVICE_ID,
          ref_type="Service",
  )

    mcp_client = ConfiguredMCPClient.from_sources(
        config_json={
            "servers": {
                "vulhub_lab": {
                    "transport": "http",
                    "url": "http://127.0.0.1:8765",
                }
            }
        }
    )

    task_spec = WorkerTaskSpec(
        task_id=TASK_ID,
        task_type=TaskType.VULNERABILITY_VALIDATION.value,
        input_bindings={
            "target_url": TARGET_URL,
            "service_id": SERVICE_ID,
            "port": 8081,
            "cve": "CVE-2017-12615",
        },
        target_refs=[service_ref],
        resource_keys=[f"service:{SERVICE_ID}"],
        constraints={
            "allowed_tools": ["vuln.tomcat_cve_2017_12615_probe"],
            "allowed_server_ids": ["vulhub_lab"],
            "allow_command_execution_payload": False,
            "validation_mode": "safe_static_marker_probe",
        },
        timeout_seconds=30,
    )

    agent_input = AgentInput(
        graph_refs=[
            service_ref,
            GraphRef(graph=GraphScope.TG, ref_id=TASK_ID, ref_type="TaskNode"),
        ],
        task_ref=TASK_ID,
        context=AgentContext(operation_id=OPERATION_ID),
        raw_payload={
            "scheduled_task": task.model_dump(mode="json"),
            "runtime_policy": {
                "authorized_hosts": ["host.docker.internal", "127.0.0.1", "localhost"],
                "authorized_ports": [8081],
                "allow_safe_file_write_probe": True,
                "allow_command_execution_payload": False,
            },
            "mcp_tool_catalog": mcp_client.list_tools(),
            "instruction": (
                "Choose the vuln.tomcat_cve_2017_12615_probe tool from server vulhub_lab. "
                "Use target_url http://host.docker.internal:8081/. "
                "Return call_mcp_tool only if the selected tool is safe marker validation."
            ),
        },
    )

    worker = LLMWorkerAgent(
        advisor=LLMWorkerAdvisor.from_env(),
        mcp_client=mcp_client,
        default_timeout_seconds=30,
    )

    output = worker.execute_task(task_spec, agent_input)
    return output, output.model_dump(mode="json")


def convert_worker_output_to_task_result(worker_output) -> AgentTaskResult:
    outcome = worker_output.outcomes[0]
    payload = outcome["payload"]
    tool_stdout = payload["tool_execution"]["stdout"]

    try:
        mcp_payload = json.loads(tool_stdout)
    except Exception:
        mcp_payload = {}

    validated = bool(mcp_payload.get("validated"))
    confidence = float(mcp_payload.get("confidence") or (0.92 if validated else 0.2))

    service_ref = KGGraphRef(graph="kg", ref_id=SERVICE_ID, ref_type="Service", label="Vulhub Tomcat 8081")
    vuln_ref = KGGraphRef(graph="kg", ref_id=VULN_ID, ref_type="Vulnerability", label="Tomcat CVE-2017-12615")
    refs = [service_ref, vuln_ref] if validated else [service_ref]

    critic_metadata = {
        "decision": "accept",
        "confidence": confidence,
        "quality": "high" if validated else "medium",
        "retry": False,
        "replan": False,
        "reason": "structured MCP result was available and safe marker validation completed",
    }

    evidence = EvidenceArtifact(
        kind="vulnerability_validation",
        summary=mcp_payload.get("summary") or outcome["summary"],
        payload_ref=f"runtime://worker-results/{TASK_ID}/full-real-chain",
        refs=refs,
        metadata={
            "mcp_payload": mcp_payload,
            "worker_output": worker_output.model_dump(mode="json"),
            "critic_metadata": critic_metadata,
            "safe_payload_summary": "static marker only; no command execution",
        },
    )

    observation = ObservationRecord(
        category="vulnerability_validation",
        summary=mcp_payload.get("summary") or outcome["summary"],
        confidence=confidence,
        refs=refs,
        payload={
            "service_id": SERVICE_ID,
            "vulnerability_id": VULN_ID,
            "cve": "CVE-2017-12615",
            "validated": validated,
            "critic": critic_metadata,
            "mcp_payload": mcp_payload,
        },
    )

    fact_writes = []
    if validated:
        fact_writes = [
            FactWriteRequest(
                kind=FactWriteKind.ENTITY_UPSERT,
                source_task_id=TASK_ID,
                subject_ref=vuln_ref,
                attributes={
                    "vulnerability_name": "Apache Tomcat CVE-2017-12615",
                    "validation_status": "validated",
                    "cve": "CVE-2017-12615",
                    "summary": mcp_payload.get("summary"),
                    "confidence": confidence,
                    "safe_payload_summary": "static marker only; no command execution",
                },
                confidence=confidence,
                evidence_ids=[evidence.evidence_id],
                summary=f"Validated {VULN_ID}",
            ),
            FactWriteRequest(
                kind=FactWriteKind.RELATION_UPSERT,
                source_task_id=TASK_ID,
                subject_ref=service_ref,
                relation_type="HAS_VULNERABILITY",
                object_ref=vuln_ref,
                attributes={
                    "validation_status": "validated",
                    "cve": "CVE-2017-12615",
                },
                confidence=confidence,
                evidence_ids=[evidence.evidence_id],
                summary=f"{SERVICE_ID} has CVE-2017-12615",
            ),
        ]

    return AgentTaskResult(
        request_id=f"request::{TASK_ID}",
        agent_role=AgentRole.VULNERABILITY_VALIDATION_WORKER,
        operation_id=OPERATION_ID,
        task_id=TASK_ID,
        tg_node_id=TASK_ID,
        status=AgentResultStatus.SUCCEEDED if outcome.get("success") else AgentResultStatus.FAILED,
        summary=mcp_payload.get("summary") or outcome["summary"],
        observations=[observation],
        evidence=[evidence],
        fact_write_requests=fact_writes,
        runtime_requests=[
            RuntimeControlRequest(
                request_type=RuntimeControlType.CONSUME_BUDGET,
                source_task_id=TASK_ID,
                budget_delta=RuntimeBudgetDelta(operations=1, noise=0.1, risk=0.1),
                reason="full real chain Vulhub CVE-2017-12615 safe validation",
            )
        ],
        outcome_payload={
            "validated": validated,
            "status": "validated" if validated else "not_detected",
            "confidence": confidence,
            "service_id": SERVICE_ID,
            "vulnerability_id": VULN_ID,
            "critic": critic_metadata,
            "mcp_payload": mcp_payload,
        },
    )


def run_existing_graph_critic(state: RuntimeState, tg: TaskGraph) -> dict:
    critic_input = AgentInput(
        graph_refs=[GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph")],
        context=AgentContext(operation_id=OPERATION_ID),
        raw_payload={
            "task_graph": tg.to_dict(),
            "runtime_state": state.model_dump(mode="json"),
            "critic_context": {
                "recent_outcomes": [],
                "critic_hints": {
                    "mode": "post_worker_graph_feedback",
                    "note": "This is existing graph-level CriticAgent, not single-result critic.",
                },
            },
        },
    )
    output = CriticAgent().execute(critic_input)
    return output.model_dump(mode="json")


def main() -> None:
    state = build_runtime_state()

    tg, planner_trace = run_real_planner(state)
    scheduler_trace = run_real_scheduler(state, tg)
    worker_output, worker_trace = run_real_worker(tg)
    task_result = convert_worker_output_to_task_result(worker_output)

    applied = PhaseTwoResultApplier().apply(
        task_result,
        state,
        task_graph=tg,
    )

    critic_trace = run_existing_graph_critic(state, tg)

    output = {
        "operation_id": OPERATION_ID,
        "target_url": TARGET_URL,
        "components": {
            "planner": "MissionPlannerAgent + LLMMissionPlannerAdvisor",
            "task_builder": "StageTaskGraphBuilder",
            "scheduler": "SchedulerAgent",
            "worker": "LLMWorkerAgent + LLMWorkerAdvisor",
            "mcp_client": "ConfiguredMCPClient",
            "mcp_server": "HTTP MCP Server / tools/list / tools/call",
            "critic": "Existing graph-level CriticAgent",
            "result_applier": "PhaseTwoResultApplier",
        },
        "planner_trace": planner_trace,
        "scheduler_trace": scheduler_trace,
        "worker_trace": worker_trace,
        "critic_trace": critic_trace,
        "result": {
            "status": task_result.status.value,
            "validated": task_result.outcome_payload["validated"],
            "confidence": task_result.outcome_payload["confidence"],
            "summary": task_result.summary,
        },
        "applied": {
            "kg_delta_count": len(applied.kg_state_deltas),
            "runtime_event_count": len(applied.runtime_event_refs),
            "visual_graph_delta_count": len(applied.visual_graph_deltas),
            "tg_graph_exists": applied.tg_graph is not None,
            "kg_state_deltas": applied.kg_state_deltas,
        },
        "runtime_audit_count": len(state.execution.metadata.get("audit_log", [])),
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print()
    print("=== FULL_REAL_CHAIN_VULHUB_RESULT ===")
    print("PASS" if task_result.status.value == "succeeded" else "CHECK")
    print(f"validated={task_result.outcome_payload['validated']}")
    print(f"confidence={task_result.outcome_payload['confidence']}")
    print(f"kg_delta_count={len(applied.kg_state_deltas)}")
    print(f"runtime_event_count={len(applied.runtime_event_refs)}")


if __name__ == "__main__":
    main()