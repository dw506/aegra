from __future__ import annotations

from src.core.models.scope import Asset, DenylistRule, Engagement, RiskPolicy
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


def test_real_penetration_defaults_allow_exploit_and_command_execution() -> None:
    """With the real-penetration defaults the framework blocks no attack action."""

    engine = PolicyEngine(RuntimePolicy())
    task = {
        "id": "task-exploit",
        "input_bindings": {"host_id": "127.0.0.1", "active_exploit": True},
        "tags": ["active_exploit"],
    }

    exploit_decision = engine.evaluate_task_policy(task)
    validator_decision = engine.evaluate_validator_policy("http-fingerprint", {"tags": ["fingerprint"]})
    fingerprint_decision = engine.evaluate_tool_policy({"kind": "fingerprint", "operation": "HEAD"})
    command_decision = engine.evaluate_tool_policy({"kind": "command_execution"})
    file_write_decision = engine.evaluate_tool_policy({"kind": "file_write"})
    reverse_decision = engine.evaluate_tool_policy({"kind": "reverse_callback"})

    assert exploit_decision.decision == "allow"
    assert validator_decision.decision == "allow"
    assert fingerprint_decision.decision == "allow"
    assert command_decision.decision == "allow"
    assert file_write_decision.decision == "allow"
    assert reverse_decision.decision == "allow"


def test_risk_policy_block_flags_are_still_enforceable_when_profile_opts_in() -> None:
    """The blocking mechanism is retained: a profile may re-impose a restriction."""

    policy = RuntimePolicy(
        risk_policy=RiskPolicy(
            block_command_execution=True,
            block_active_exploit=True,
            require_approval_for_active_exploit=True,
        )
    )
    engine = PolicyEngine(policy)

    command_decision = engine.evaluate_tool_policy({"kind": "command_execution"})
    exploit_decision = engine.evaluate_tool_policy({"kind": "active_exploit"})

    assert command_decision.decision == "deny"
    assert exploit_decision.decision == "requires_approval"
