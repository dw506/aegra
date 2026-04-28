from __future__ import annotations

import pytest

from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.runtime.budgets import BudgetExceededError, RuntimeBudgetManager


def build_state() -> RuntimeState:
    return RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))


def test_remaining_budget_summary_reports_remaining_values() -> None:
    state = build_state()
    manager = RuntimeBudgetManager()
    state.budgets.time_budget_max_sec = 100
    state.budgets.time_budget_used_sec = 30
    state.budgets.token_budget_max = 1000
    state.budgets.token_budget_used = 200

    summary = manager.remaining_budget_summary(state)

    assert summary["time_budget_remaining_sec"] == 70
    assert summary["token_budget_remaining"] == 800


def test_set_and_get_approval() -> None:
    state = build_state()
    manager = RuntimeBudgetManager()

    manager.set_approval(state, "gate-1", True)

    assert manager.has_approval(state, "gate-1") is True


def test_set_policy_flag_updates_budget_state() -> None:
    state = build_state()
    manager = RuntimeBudgetManager()

    manager.set_policy_flag(state, "allow_shared_sessions", True)

    assert state.budgets.policy_flags["allow_shared_sessions"] is True


def test_to_readable_summary_contains_budget_dimensions() -> None:
    state = build_state()
    manager = RuntimeBudgetManager()

    summary = manager.to_readable_summary(state)

    assert "time=" in summary
    assert "tokens=" in summary


def test_consume_operations_raises_when_over_limit() -> None:
    state = build_state()
    manager = RuntimeBudgetManager()
    state.budgets.operation_budget_max = 2
    manager.consume_operations(state, 2)

    with pytest.raises(BudgetExceededError):
        manager.consume_operations(state, 1)

