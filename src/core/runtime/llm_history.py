"""LLM decision history helpers for operation metadata."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.runtime import RuntimeState, utc_now

LLM_DECISION_HISTORY_KEY = "llm_decision_history"
DEFAULT_LLM_DECISION_HISTORY_LIMIT = 200


class LLMDecisionHistoryRecord(BaseModel):
    """Auditable, prompt-free summary of one LLM advisor decision."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    cycle_index: int = Field(ge=0)
    agent_kind: str = Field(min_length=1)
    advisor_type: str = Field(min_length=1)
    enabled: bool
    configured: bool
    decision_type: str = Field(min_length=1)
    accepted: bool
    rejected_reason: str | None = None
    model: str | None = None
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())


def ensure_llm_decision_history(state: RuntimeState) -> list[dict[str, Any]]:
    """Return the operation-level LLM decision history list."""

    history = state.execution.metadata.setdefault(LLM_DECISION_HISTORY_KEY, [])
    if not isinstance(history, list):
        history = []
        state.execution.metadata[LLM_DECISION_HISTORY_KEY] = history
    return history


def append_llm_decision_history(
    state: RuntimeState,
    records: list[LLMDecisionHistoryRecord],
    *,
    keep_last: int = DEFAULT_LLM_DECISION_HISTORY_LIMIT,
) -> list[dict[str, Any]]:
    """Append sanitized LLM decision history records to operation metadata."""

    history = ensure_llm_decision_history(state)
    history.extend(record.model_dump(mode="json") for record in records)
    if keep_last > 0 and len(history) > keep_last:
        del history[: len(history) - keep_last]
    state.last_updated = utc_now()
    return history


def recent_llm_decision_history(state: RuntimeState, *, limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent N LLM decision history entries."""

    history = ensure_llm_decision_history(state)
    parsed_limit = max(0, int(limit))
    if parsed_limit == 0:
        return []
    return [dict(item) for item in history[-parsed_limit:] if isinstance(item, dict)]


__all__ = [
    "DEFAULT_LLM_DECISION_HISTORY_LIMIT",
    "LLMDecisionHistoryRecord",
    "LLM_DECISION_HISTORY_KEY",
    "append_llm_decision_history",
    "ensure_llm_decision_history",
    "recent_llm_decision_history",
]
