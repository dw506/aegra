from __future__ import annotations

from src.core.models.ag import GraphRef
from src.core.agents.agent_protocol import AgentContext, AgentInput, GraphRef as AgentGraphRef, GraphScope
from src.core.agents.scheduler_agent import SchedulerAgent
from src.core.models.runtime import OperationRuntime, RuntimeState, WorkerRuntime, WorkerStatus
from src.core.models.scope import Asset, Engagement
from src.core.models.tg import TaskGraph, TaskNode, TaskStatus, TaskType
from src.core.runtime.policy_gate import PolicyGate, PolicyGateAction


def _state(policy: dict | None = None) -> RuntimeState:
    state = RuntimeState(operation_id="op-policy-gate", execution=OperationRuntime(operation_id="op-policy-gate"))
    if policy is not None:
        state.execution.metadata["runtime_policy"] = policy
    return state


def _task(*, host: str = "10.0.0.5", tool: str = "nmap", risk: float = 0.1) -> TaskNode:
    return TaskNode(
        id="task-policy-gate",
        label="Policy gate task",
        task_type=TaskType.SERVICE_VALIDATION,
        status=TaskStatus.READY,
        source_action_id="action-policy-gate",
        input_bindings={"host_id": host, "tool_hint": tool},
        target_refs=[GraphRef(graph="kg", ref_id=host, ref_type="Host")],
        estimated_risk=risk,
        estimated_noise=0.1,
    )


def test_policy_gate_denies_target_outside_scope() -> None:
    state = _state(
        {
            "engagement": Engagement(
                engagement_id="eng-1",
                assets=[Asset(kind="host", value="10.0.0.1")],
            ).model_dump(mode="json")
        }
    )

    decision = PolicyGate().evaluate(_task(host="10.0.0.5"), runtime_state=state)

    assert decision.action == PolicyGateAction.DENY
    assert decision.gate == "scope"


def test_policy_gate_denies_tool_outside_allowlist() -> None:
    state = _state({"command_allowlist": ["nmap"]})

    decision = PolicyGate().evaluate(_task(tool="curl"), runtime_state=state)

    assert decision.action == PolicyGateAction.DENY
    assert decision.gate == "tool"


def test_policy_gate_reports_approval_requirement() -> None:
    task = _task()
    task.approval_required = True

    decision = PolicyGate().evaluate(task, runtime_state=_state())

    assert decision.action == PolicyGateAction.NEED_APPROVAL
    assert decision.approval_id == f"task:{task.id}:approved"


def test_policy_gate_denies_when_budget_is_insufficient() -> None:
    state = _state()
    state.budgets.operation_budget_max = 0

    decision = PolicyGate().evaluate(_task(), runtime_state=state)

    assert decision.action == PolicyGateAction.DENY
    assert decision.gate == "budget"


def test_scheduler_agent_applies_policy_gate_before_assignment() -> None:
    state = _state({"command_allowlist": ["nmap"]})
    state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    graph = TaskGraph()
    graph.add_node(_task(tool="curl"))

    result = SchedulerAgent().run(
        AgentInput(
            graph_refs=[AgentGraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph")],
            context=AgentContext(operation_id="op-policy-gate"),
            raw_payload={"tg_graph": graph.to_dict(), "runtime_state": state.model_dump(mode="json")},
        )
    )

    assert result.success is True
    assert result.output.decisions[0]["action"] == "deny"
    assert not result.output.decisions[0]["accepted"]
    assert any(delta["delta_type"] == "task_status_update" for delta in result.output.state_deltas)
