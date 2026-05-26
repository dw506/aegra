from __future__ import annotations

import sys

from src.core.execution import (
    ExecutionExecutor,
    NetnsShellAdapter,
    PivotExecutionContextResolver,
    ProxyShellAdapter,
    ToolPlan,
    TunnelAdapter,
    build_tool_plan,
)
from src.core.models.ag import GraphRef
from src.core.models.events import AgentResultStatus, AgentRole, AgentTaskResult, RuntimeControlRequest, RuntimeControlType
from src.core.models.runtime import OperationRuntime, PivotRouteRuntime, PivotRouteStatus, RuntimeState, SessionRuntime, SessionStatus
from src.core.models.tg import TaskNode, TaskStatus, TaskType
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.workers.base import WorkerTaskSpec
from src.core.workers.services.pivot_validation_service import PivotValidationRequest, PivotValidationService


def _state() -> RuntimeState:
    return RuntimeState(operation_id="op-pivot", execution=OperationRuntime(operation_id="op-pivot"))


def test_build_tool_plan_carries_selected_route_and_uses_route_adapter() -> None:
    task = TaskNode(
        id="task-1",
        label="pivoted probe",
        task_type=TaskType.SERVICE_VALIDATION,
        status=TaskStatus.READY,
        input_bindings={
            "tool_hint": "curl",
            "route_id": "route-1",
            "selected_route": {
                "route_id": "route-1",
                "session_id": "sess-1",
                "metadata": {"transport": {"adapter": "proxy_shell", "proxy_url": "socks5://127.0.0.1:1080"}},
            },
        },
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
    )

    plan = build_tool_plan(task)

    assert plan.adapter == "proxy_shell"
    assert plan.args["route_id"] == "route-1"
    assert plan.metadata["selected_route_id"] == "route-1"
    assert plan.metadata["selected_route"]["session_id"] == "sess-1"


def test_pivot_execution_context_resolves_route_session_and_proxy() -> None:
    state = _state()
    state.sessions["sess-1"] = SessionRuntime(
        session_id="sess-1",
        status=SessionStatus.ACTIVE,
        metadata={"execution_endpoint": {"adapter": "proxy_shell", "proxy_url": "socks5://127.0.0.1:1080"}},
    )
    state.pivot_routes["route-1"] = PivotRouteRuntime(
        route_id="route-1",
        destination_host="host-2",
        session_id="sess-1",
        status=PivotRouteStatus.ACTIVE,
        metadata={"transport": {"kind": "socks"}},
    )
    plan = ToolPlan(task_id="task-1", tool="echo", adapter="proxy_shell", command="echo ok", args={"route_id": "route-1"})

    context = PivotExecutionContextResolver().resolve(plan, state)

    assert context.route_id == "route-1"
    assert context.session_id == "sess-1"
    assert context.proxy_url == "socks5://127.0.0.1:1080"
    assert context.env["ALL_PROXY"] == "socks5://127.0.0.1:1080"


def test_proxy_shell_adapter_injects_proxy_environment_for_allowed_command() -> None:
    plan = ToolPlan(
        task_id="task-1",
        tool="python",
        adapter="proxy_shell",
        command=sys.executable,
        args={
            "proxy_url": "http://127.0.0.1:8080",
            "argv": [sys.executable, "-c", "import os; print(os.environ.get('HTTP_PROXY'))"],
        },
    )

    result = ProxyShellAdapter().execute(plan)

    assert result.success is True
    assert result.stdout.strip() == "http://127.0.0.1:8080"


def test_tunnel_adapter_returns_preopened_endpoint() -> None:
    plan = ToolPlan(
        task_id="task-1",
        tool="tunnel_probe",
        adapter="tcp_tunnel",
        command="tunnel_probe",
        args={"tunnel_endpoint": "127.0.0.1:15432"},
    )

    result = TunnelAdapter().execute(plan)

    assert result.success is True
    assert result.stdout == "127.0.0.1:15432"


def test_netns_adapter_reports_unsupported_on_windows() -> None:
    plan = ToolPlan(
        task_id="task-1",
        tool="echo",
        adapter="netns_shell",
        command="echo ok",
        args={"network_namespace": "aegra-test"},
    )

    result = NetnsShellAdapter().execute(plan)

    if sys.platform.startswith("win"):
        assert result.success is False
        assert result.exit_code == "unsupported_platform"


def test_result_applier_registers_tunnel_backed_pivot_route() -> None:
    state = _state()
    result = AgentTaskResult(
        request_id="req-1",
        agent_role=AgentRole.ACCESS_WORKER,
        operation_id=state.operation_id,
        task_id="task-1",
        tg_node_id="task-1",
        status=AgentResultStatus.SUCCEEDED,
        summary="registered route",
        runtime_requests=[
            RuntimeControlRequest(
                request_type=RuntimeControlType.REGISTER_PIVOT_ROUTE,
                source_task_id="task-1",
                metadata={
                    "route_id": "route-1",
                    "destination_host": "db-1",
                    "source_host": "web-1",
                    "active": True,
                    "transport": {"adapter": "proxy_shell", "proxy_url": "socks5://127.0.0.1:1080"},
                },
            ),
            RuntimeControlRequest(
                request_type=RuntimeControlType.OPEN_TUNNEL,
                source_task_id="task-1",
                metadata={"route_id": "route-1", "tunnel_endpoint": "127.0.0.1:15432"},
            ),
        ],
    )

    apply_result = PhaseTwoResultApplier().apply(result, state)

    assert len(apply_result.runtime_event_refs) == 2
    assert state.pivot_routes["route-1"].status == PivotRouteStatus.ACTIVE
    assert state.pivot_routes["route-1"].metadata["transport"]["tunnel_endpoint"] == "127.0.0.1:15432"


def test_pivot_validation_service_runs_probe_and_emits_verify_request() -> None:
    service = PivotValidationService(executor=ExecutionExecutor([ProxyShellAdapter()]))
    request = PivotValidationRequest(
        operation_id="op-pivot",
        task_id="task-1",
        task_type="pivot_route_validation",
        task_label="Pivot validation",
        input_bindings={
            "route_id": "route-1",
            "selected_route": {
                "route_id": "route-1",
                "destination_host": "db-1",
                "metadata": {"transport": {"proxy_url": "http://127.0.0.1:8080"}},
            },
            "execution_adapter": "proxy_shell",
            "probe_command": [sys.executable, "-c", "print('ok')"],
            "proxy_url": "http://127.0.0.1:8080",
        },
    )

    result = service.validate(request)

    assert result.success is True
    assert result.raw_payload["tool_execution"]["success"] is True
    assert result.runtime_requests[0]["request_type"] == "verify_pivot_route"


def test_pivot_validation_request_accepts_worker_task_route_constraint() -> None:
    task_spec = WorkerTaskSpec(
        task_id="task-1",
        task_type="pivot_route_validation",
        input_bindings={},
        constraints={"route_id": "route-1"},
    )

    request = PivotValidationRequest.from_task_spec(
        task_spec=task_spec,
        agent_input=type("Input", (), {"raw_payload": {}, "context": type("Context", (), {"operation_id": "op-1", "extra": {}})()})(),
    )

    assert request.metadata["selected_route"]["route_id"] == "route-1"
