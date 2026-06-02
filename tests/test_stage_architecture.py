from __future__ import annotations

from src.core.models.ag import GraphRef
from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntime
from src.core.models.tg import TaskGraph, TaskStatus, TaskType
from src.core.planning.stage_task_builder import StageTaskGraphBuilder
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.scheduling.stage_scheduler import schedule_ready_stage_tasks
from src.core.stage.adapters import StageResultAdapter
from src.core.stage.models import StageResult, StageTask, StageType, ToolTrace


def test_stage_task_builder_writes_ready_stage_chain() -> None:
    graph = TaskGraph()
    builder = StageTaskGraphBuilder()
    recon = StageTask(
        task_id="stage-recon-1",
        stage_type=StageType.RECON_STAGE,
        objective="Discover public web",
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
    )
    vuln = StageTask(
        task_id="stage-vuln-1",
        stage_type=StageType.VULN_ANALYSIS_STAGE,
        objective="Analyze vulnerabilities",
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
    )

    created = builder.upsert_stage_tasks(
        graph,
        [recon, vuln],
        dependencies=[{"source": "stage-recon-1", "target": "stage-vuln-1"}],
    )

    assert created == ["stage-recon-1", "stage-vuln-1"]
    assert graph.get_node("stage-recon-1").task_type == TaskType.RECON_STAGE
    assert graph.get_node("stage-recon-1").status == TaskStatus.READY
    assert graph.get_node("stage-vuln-1").status == TaskStatus.DRAFT
    assert [task.task_id for task in schedule_ready_stage_tasks(graph)] == ["stage-recon-1"]


def test_stage_result_adapter_and_applier_write_runtime_stage_effects() -> None:
    state = RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))
    state.register_task(TaskRuntime(task_id="stage-exploit-1", tg_node_id="stage-exploit-1"))
    graph = TaskGraph()
    graph.add_node(
        StageTaskGraphBuilder._task_node(
            StageTask(
                task_id="stage-exploit-1",
                stage_type=StageType.EXPLOIT_STAGE,
                objective="Establish access",
                target_refs=[GraphRef(graph="kg", ref_id="svc-1", ref_type="Service")],
            )
        )
    )
    graph.refresh_blocked_states()
    result = StageResult(
        operation_id="op-1",
        stage_task_id="stage-exploit-1",
        stage_type=StageType.EXPLOIT_STAGE,
        agent_name="exploit_validation_agent",
        status="succeeded",
        summary="Established controlled access",
        capabilities_gained=[
            {
                "capability_id": "cap-1",
                "capability_type": "session",
                "source_task_id": "stage-exploit-1",
                "host_id": "host-1",
                "runtime_ref": "session-1",
            }
        ],
        sessions=[
            {
                "session_id": "session-1",
                "bound_identity": "www-data",
                "bound_target": "host-1",
                "lease_seconds": 120,
                "reuse_policy": "shared",
            }
        ],
        pivot_routes=[
            {
                "route_id": "route-1",
                "destination_host": "internal-admin",
                "source_host": "host-1",
                "session_id": "session-1",
                "active": True,
            }
        ],
        next_stage_candidates=[
            {
                "stage_type": StageType.GOAL_STAGE.value,
                "objective": "Verify the mission goal from established access",
                "required_context": {"session_id": "session-1"},
                "success_criteria": ["goal condition verified"],
                "priority": 90,
            }
        ],
        tool_trace=[
            ToolTrace(
                step=0,
                server_id="pentest-tools",
                tool_name="controlled_exploit_attempt",
                success=True,
                summary="access established",
            )
        ],
    )

    task_result = StageResultAdapter.to_task_result(result)
    applied = PhaseTwoResultApplier().apply(task_result, state, task_graph=graph)

    assert state.sessions["session-1"].bound_identity == "www-data"
    assert state.pivot_routes["route-1"].is_usable()
    assert state.execution.metadata["capabilities"][0]["capability_id"] == "cap-1"
    assert graph.get_node("stage-exploit-1").status == TaskStatus.SUCCEEDED
    assert applied.tg_graph is not None
    updated_graph = TaskGraph.from_dict(applied.tg_graph)
    assert any(
        node.task_type == TaskType.GOAL_STAGE
        and node.input_bindings["objective"] == "Verify the mission goal from established access"
        for node in updated_graph.list_nodes()
        if hasattr(node, "task_type")
    )
    assert any(entry["event_type"] == "stage_tool_trace" for entry in state.execution.metadata["audit_log"])
