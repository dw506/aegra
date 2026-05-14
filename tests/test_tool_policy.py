from __future__ import annotations

import sys
from time import perf_counter

from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.runtime.policy import RuntimePolicy
from src.core.runtime.policy_engine import PolicyEngine
from src.core.workers.tool_runner import ToolExecutionSpec, ToolRunner


def build_state() -> RuntimeState:
    state = RuntimeState(operation_id="op-tool-policy", execution=OperationRuntime(operation_id="op-tool-policy"))
    state.execution.metadata["control_plane"] = {"audit_redaction_enabled": True}
    return state


def test_disabled_tool_cannot_execute_and_is_audited() -> None:
    state = build_state()
    runner = ToolRunner(policy_engine=PolicyEngine(RuntimePolicy(disabled_tools=["fixture-tool"])))

    result = runner.run(
        ToolExecutionSpec(
            command=[sys.executable, "-c", "print('should-not-run')"],
            policy_metadata={"kind": "fixture-tool", "name": "fixture-tool", "tags": ["fingerprint"]},
        ),
        runtime_state=state,
    )

    assert result.success is False
    assert result.category == "policy_denied"
    assert "disabled by policy" in str(result.error_message)
    assert result.stdout == ""
    assert state.execution.metadata["audit_log"][-1]["event_type"] == "tool_execution"
    assert state.execution.metadata["audit_log"][-1]["tool"]["category"] == "policy_denied"


def test_tool_runner_timeout_and_output_truncation_are_enforced() -> None:
    runner = ToolRunner()

    timeout_result = runner.run(
        ToolExecutionSpec(
            command=[sys.executable, "-c", "import time; time.sleep(2)"],
            timeout_sec=1,
        )
    )
    truncated_result = runner.run(
        ToolExecutionSpec(
            command=[sys.executable, "-c", "print('A' * 200)"],
            stdout_max_bytes=16,
        )
    )

    assert timeout_result.timed_out is True
    assert timeout_result.category == "timeout"
    assert truncated_result.success is True
    assert truncated_result.stdout_truncated is True
    assert "(truncated" in truncated_result.stdout


def test_command_allowlist_blocks_unapproved_binary() -> None:
    runner = ToolRunner()

    result = runner.run(
        ToolExecutionSpec(
            command=[sys.executable, "-c", "print('blocked')"],
            command_allowlist={"nmap"},
        )
    )

    assert result.success is False
    assert result.category == "policy_denied"
    assert "outside command allowlist" in str(result.error_message)


def test_tool_runner_applies_rate_limit_between_runs() -> None:
    runner = ToolRunner()
    spec = ToolExecutionSpec(
        command=[sys.executable, "-c", "print('ok')"],
        min_interval_sec=0.05,
        rate_limit_key="fixture-rate",
    )

    runner.run(spec)
    started = perf_counter()
    runner.run(spec)

    assert perf_counter() - started >= 0.04
