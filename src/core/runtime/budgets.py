"""Runtime budget manager.

This module manages execution-time budget consumption only. The tracked budget
state belongs to Runtime State and must not be written into KG, AG or TG.
"""

from __future__ import annotations

from src.core.models.runtime import BudgetRuntime, RuntimeState, utc_now


class BudgetExceededError(RuntimeError):
    """Raised when a budget consumption request would exceed configured limits."""


class RuntimeBudgetManager:
    """Manage time, token, operation, noise and risk budgets for one run."""

    def consume_time(self, state: RuntimeState, seconds: float) -> BudgetRuntime:
        """Consume runtime wall-clock budget in seconds."""

        self._ensure_not_exceeded(state, time_sec=seconds)
        state.budgets.time_budget_used_sec += seconds
        state.last_updated = utc_now()
        return state.budgets

    def consume_tokens(self, state: RuntimeState, tokens: int) -> BudgetRuntime:
        """Consume runtime token budget."""

        self._ensure_not_exceeded(state, tokens=tokens)
        state.budgets.token_budget_used += tokens
        state.last_updated = utc_now()
        return state.budgets

    def consume_operations(self, state: RuntimeState, count: int = 1) -> BudgetRuntime:
        """Consume one or more runtime operation budget units."""

        self._ensure_not_exceeded(state, operations=count)
        state.budgets.operation_budget_used += count
        state.last_updated = utc_now()
        return state.budgets

    def consume_noise(self, state: RuntimeState, amount: float) -> BudgetRuntime:
        """Consume runtime noise budget."""

        self._ensure_not_exceeded(state, noise=amount)
        state.budgets.noise_budget_used += amount
        state.last_updated = utc_now()
        return state.budgets

    def consume_risk(self, state: RuntimeState, amount: float) -> BudgetRuntime:
        """Consume runtime risk budget."""

        self._ensure_not_exceeded(state, risk=amount)
        state.budgets.risk_budget_used += amount
        state.last_updated = utc_now()
        return state.budgets

    def set_approval(self, state: RuntimeState, gate_id: str, decision: bool) -> BudgetRuntime:
        """Record one runtime approval gate decision."""

        state.budgets.approval_cache[gate_id] = decision
        state.last_updated = utc_now()
        return state.budgets

    def has_approval(self, state: RuntimeState, gate_id: str) -> bool:
        """Return True when the runtime approval cache contains a positive decision."""

        return bool(state.budgets.approval_cache.get(gate_id, False))

    def remaining_budget_summary(self, state: RuntimeState) -> dict[str, int | float | None]:
        """Return remaining budget values for each tracked budget dimension."""

        budgets = state.budgets
        return {
            "time_budget_remaining_sec": self._remaining(budgets.time_budget_max_sec, budgets.time_budget_used_sec),
            "token_budget_remaining": self._remaining(budgets.token_budget_max, budgets.token_budget_used),
            "operation_budget_remaining": self._remaining(budgets.operation_budget_max, budgets.operation_budget_used),
            "noise_budget_remaining": self._remaining(budgets.noise_budget_max, budgets.noise_budget_used),
            "risk_budget_remaining": self._remaining(budgets.risk_budget_max, budgets.risk_budget_used),
        }

    def would_exceed_budget(
        self,
        state: RuntimeState,
        *,
        time_sec: float = 0.0,
        tokens: int = 0,
        operations: int = 0,
        noise: float = 0.0,
        risk: float = 0.0,
    ) -> bool:
        """Return True when a prospective consumption would exceed any configured limit."""

        budgets = state.budgets
        checks = (
            self._would_exceed(budgets.time_budget_max_sec, budgets.time_budget_used_sec, time_sec),
            self._would_exceed(budgets.token_budget_max, budgets.token_budget_used, tokens),
            self._would_exceed(budgets.operation_budget_max, budgets.operation_budget_used, operations),
            self._would_exceed(budgets.noise_budget_max, budgets.noise_budget_used, noise),
            self._would_exceed(budgets.risk_budget_max, budgets.risk_budget_used, risk),
        )
        return any(checks)

    def to_readable_summary(self, state: RuntimeState) -> str:
        """Render a compact human-readable budget summary for logs."""

        budgets = state.budgets
        remaining = self.remaining_budget_summary(state)
        return (
            "budgets("
            f"time={budgets.time_budget_used_sec}/{budgets.time_budget_max_sec}, "
            f"tokens={budgets.token_budget_used}/{budgets.token_budget_max}, "
            f"operations={budgets.operation_budget_used}/{budgets.operation_budget_max}, "
            f"noise={budgets.noise_budget_used}/{budgets.noise_budget_max}, "
            f"risk={budgets.risk_budget_used}/{budgets.risk_budget_max}, "
            f"remaining={remaining}"
            ")"
        )

    def set_policy_flag(
        self,
        state: RuntimeState,
        flag: str,
        value: bool | int | float | str,
    ) -> BudgetRuntime:
        """Record one runtime policy flag on the budget state."""

        state.budgets.policy_flags[flag] = value
        state.last_updated = utc_now()
        return state.budgets

    def _ensure_not_exceeded(
        self,
        state: RuntimeState,
        *,
        time_sec: float = 0.0,
        tokens: int = 0,
        operations: int = 0,
        noise: float = 0.0,
        risk: float = 0.0,
    ) -> None:
        """Raise a clear exception when a prospective consumption exceeds any limit."""

        if self.would_exceed_budget(
            state,
            time_sec=time_sec,
            tokens=tokens,
            operations=operations,
            noise=noise,
            risk=risk,
        ):
            raise BudgetExceededError(self.to_readable_summary(state))

    @staticmethod
    def _would_exceed(maximum: int | float | None, used: int | float, delta: int | float) -> bool:
        """Return True when `used + delta` exceeds a configured maximum."""

        if maximum is None:
            return False
        return used + delta > maximum

    @staticmethod
    def _remaining(maximum: int | float | None, used: int | float) -> int | float | None:
        """Return remaining budget or `None` when the budget is unbounded."""

        if maximum is None:
            return None
        return maximum - used


__all__ = ["BudgetExceededError", "RuntimeBudgetManager"]

