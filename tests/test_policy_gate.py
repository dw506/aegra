from __future__ import annotations

from src.core.models.ag import GraphRef
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.models.scope import Asset, Engagement
from src.core.models.tg import TaskNode, TaskStatus, TaskType
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


def test_policy_gate_audits_target_outside_scope_without_blocking(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    state = _state(
        {
            "engagement": Engagement(
                engagement_id="eng-1",
                assets=[Asset(kind="host", value="10.0.0.1")],
            ).model_dump(mode="json")
        }
    )

    decision = PolicyGate().evaluate(_task(host="10.0.0.5"), runtime_state=state)

    assert decision.action == PolicyGateAction.ALLOW
    assert decision.gate == "scope"
    assert decision.metadata["policy_audit_only"] is True
    assert decision.metadata["original_allowed"] is False
    trace = tmp_path / "runs" / f"{state.operation_id}.run.txt"
    assert trace.exists()
    assert "【POLICY_DECISION】" in trace.read_text(encoding="utf-8")


def test_policy_gate_allows_internal_kg_refs_under_scoped_policy() -> None:
    state = _state(
        {
            "authorized_hosts": ["127.0.0.1"],
            "cidr_whitelist": ["127.0.0.1/32"],
        }
    )
    task = _task(host="kg-host::abc123")
    task.input_bindings["service_id"] = "kg-host::abc123:8080/tcp"
    task.target_refs.append(GraphRef(graph="kg", ref_id="kg-host::abc123:8080/tcp", ref_type="Service"))
    task.resource_keys = {"host:kg-host::abc123", "service:kg-host::abc123:8080/tcp"}

    decision = PolicyGate().evaluate(task, runtime_state=state)

    assert decision.action == PolicyGateAction.ALLOW


def test_policy_gate_audits_tool_outside_allowlist_without_blocking() -> None:
    state = _state({"command_allowlist": ["nmap"]})

    decision = PolicyGate().evaluate(_task(tool="curl"), runtime_state=state)

    assert decision.action == PolicyGateAction.ALLOW
    assert decision.gate == "tool"
    assert decision.metadata["original_allowed"] is False


def test_policy_gate_audits_approval_requirement_without_blocking() -> None:
    task = _task()
    task.approval_required = True

    decision = PolicyGate().evaluate(task, runtime_state=_state())

    assert decision.action == PolicyGateAction.ALLOW
    assert decision.approval_id == f"task:{task.id}:approved"
    assert decision.metadata["original_action"] == PolicyGateAction.NEED_APPROVAL.value


def test_policy_gate_audits_insufficient_budget_without_blocking() -> None:
    state = _state()
    state.budgets.operation_budget_max = 0

    decision = PolicyGate().evaluate(_task(), runtime_state=state)

    assert decision.action == PolicyGateAction.ALLOW
    assert decision.gate == "budget"
    assert decision.metadata["original_allowed"] is False
