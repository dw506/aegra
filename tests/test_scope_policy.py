from __future__ import annotations

from src.core.models.scope import Asset, DenylistRule, Engagement
from src.core.models.tg import TaskNode, TaskStatus, TaskType
from src.core.runtime.policy import RuntimePolicy
from src.core.runtime.policy_engine import PolicyEngine


def test_scope_policy_blocks_unauthorized_and_denylisted_targets() -> None:
    policy = RuntimePolicy(
        engagement=Engagement(
            engagement_id="eng-1",
            assets=[Asset(kind="host", value="10.0.0.5")],
            denylist=[DenylistRule(rule_id="deny-1", kind="host", value="10.0.0.9", reason="excluded asset")],
        )
    )
    engine = PolicyEngine(policy)

    allowed = engine.evaluate_target_scope("10.0.0.5")
    unauthorized = engine.evaluate_target_scope("10.0.0.8")
    denylisted = engine.evaluate_target_scope("10.0.0.9")

    assert allowed.decision == "allow"
    assert unauthorized.decision == "deny"
    assert unauthorized.reason == "target outside allowlist"
    assert denylisted.decision == "deny"
    assert denylisted.matched_rule_id == "deny-1"


def test_policy_engine_requires_approval_for_active_exploit_and_allows_fingerprint() -> None:
    engine = PolicyEngine(RuntimePolicy())
    task = TaskNode(
        id="task-exploit",
        label="Exploit",
        task_type=TaskType.VULNERABILITY_VALIDATION,
        status=TaskStatus.READY,
        source_action_id="action-exploit",
        input_bindings={"host_id": "127.0.0.1", "active_exploit": True},
        tags={"active_exploit"},
    )

    exploit_decision = engine.evaluate_task_policy(task)
    validator_decision = engine.evaluate_validator_policy("http-fingerprint", {"tags": ["fingerprint"]})
    tool_decision = engine.evaluate_tool_policy({"kind": "fingerprint", "operation": "HEAD"})
    command_decision = engine.evaluate_tool_policy({"kind": "command_execution"})

    assert exploit_decision.decision == "requires_approval"
    assert validator_decision.decision == "allow"
    assert tool_decision.decision == "allow"
    assert command_decision.decision == "deny"
