"""Operation-level audit report helpers for LLM and control observability."""

from __future__ import annotations

import re
from typing import Any

from src.core.models.runtime import RuntimeState, utc_now

_REDACTED_VALUE = "[REDACTED]"
_DEFAULT_STRING_LIMIT = 512
_LONG_CONTEXT_LIMIT = 256
_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
)
_BLOCKED_KEY_MARKERS = (
    "prompt",
    "raw_response",
    "llm_response",
    "model_response",
)
_LONG_CONTEXT_KEY_MARKERS = (
    "context",
    "messages",
    "transcript",
)
_INLINE_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)([^\s,;]+)"),
    re.compile(r"(?i)(\b(?:token|password|secret|api[_-]?key)\b\s*[:=]\s*)([^\s,;]+)"),
)


def build_operation_audit_report(
    state: RuntimeState,
    *,
    limit: int = 100,
    agent_kind: str | None = None,
    accepted: bool | None = None,
) -> dict[str, Any]:
    """Build a sanitized operation audit/control observability report."""

    parsed_limit = _coerce_limit(limit)
    control_cycle_history = _metadata_list(state, "control_cycle_history")
    replan_requests = [item.model_dump(mode="json") for item in state.replan_requests]
    budget_summary = _budget_summary(state)
    report = {
        "operation_id": state.operation_id,
        "exported_at": utc_now().isoformat(),
        "operation_status": state.operation_status.value,
        "execution_status": state.execution.status.value,
        "pause_reason": state.execution.metadata.get("pause_reason"),
        "latest_supervisor_control_strategy": _metadata_mapping(
            state,
            "last_supervisor_control_strategy",
        ),
        "control_cycle_history": _recent_items(
            control_cycle_history,
            limit=parsed_limit,
        ),
        "replan_requests": _recent_items(
            replan_requests,
            limit=parsed_limit,
        ),
        "operation_log": _recent_items(
            _metadata_list(state, "operation_log"),
            limit=parsed_limit,
        ),
        "audit_log": _recent_items(
            _metadata_list(state, "audit_log"),
            limit=parsed_limit,
        ),
        "budget_summary": budget_summary,
        "derived": {
            "correlations": _build_correlations(
                state,
                control_cycle_history=control_cycle_history,
                replan_requests=replan_requests,
                budget_summary=budget_summary,
            ),
        },
        "filters": {
            "limit": parsed_limit,
            "agent_kind": agent_kind,
            "accepted": accepted,
        },
    }
    return _sanitize(report)


def _build_correlations(
    state: RuntimeState,
    *,
    control_cycle_history: list[dict[str, Any]],
    replan_requests: list[dict[str, Any]],
    budget_summary: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    strategy_records = _supervisor_strategy_records(state, control_cycle_history=control_cycle_history)
    correlations: dict[str, list[dict[str, Any]]] = {
        "accepted_request_replans": [],
        "accepted_pauses": [],
        "budget_guards": [],
        "deterministic_fallbacks": [],
    }
    for strategy in strategy_records:
        strategy_name = str(strategy.get("strategy", ""))
        cycle_index = _coerce_optional_int(strategy.get("cycle_index"))
        accepted = strategy.get("accepted") is True
        if accepted and strategy_name == "request_replan":
            correlations["accepted_request_replans"].append(
                {
                    "strategy": dict(strategy),
                    "replan_request": _matching_replan_request(replan_requests, cycle_index=cycle_index),
                }
            )
        elif accepted and strategy_name == "pause_for_review":
            correlations["accepted_pauses"].append(
                {
                    "strategy": dict(strategy),
                    "pause_reason": state.execution.metadata.get("pause_reason"),
                    "pause_cycle_index": state.execution.metadata.get("pause_cycle_index", cycle_index),
                }
            )
        elif accepted and strategy_name == "budget_guard":
            correlations["budget_guards"].append(
                {
                    "strategy": dict(strategy),
                    "guards": dict(budget_summary.get("guards", {})),
                    "requires_human_review": bool(budget_summary.get("requires_human_review", False)),
                }
            )
        elif strategy_name == "deterministic_fallback":
            correlations["deterministic_fallbacks"].append(
                {
                    "strategy": dict(strategy),
                }
            )
    return correlations


def _supervisor_strategy_records(
    state: RuntimeState,
    *,
    control_cycle_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for cycle in control_cycle_history:
        strategy = cycle.get("supervisor_control_strategy")
        if not isinstance(strategy, dict):
            continue
        item = dict(strategy)
        item.setdefault("cycle_index", cycle.get("cycle_index"))
        _append_unique_strategy(records, seen, item)
    latest = state.execution.metadata.get("last_supervisor_control_strategy")
    if isinstance(latest, dict):
        _append_unique_strategy(records, seen, dict(latest))
    return records


def _append_unique_strategy(
    records: list[dict[str, Any]],
    seen: set[tuple[Any, Any, Any]],
    item: dict[str, Any],
) -> None:
    key = (item.get("cycle_index"), item.get("strategy"), item.get("created_at"))
    if key in seen:
        return
    seen.add(key)
    records.append(item)


def _matching_replan_request(
    replan_requests: list[dict[str, Any]],
    *,
    cycle_index: int | None,
) -> dict[str, Any] | None:
    for request in reversed(replan_requests):
        metadata = request.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if metadata.get("source") != "supervisor":
            continue
        if cycle_index is None or _coerce_optional_int(metadata.get("cycle_index")) == cycle_index:
            return dict(request)
    return None


def _budget_summary(state: RuntimeState) -> dict[str, Any]:
    budgets = state.budgets
    guards = {
        "operation": _budget_exhausted(budgets.operation_budget_used, budgets.operation_budget_max),
        "time": _budget_exhausted(budgets.time_budget_used_sec, budgets.time_budget_max_sec),
        "token": _budget_exhausted(budgets.token_budget_used, budgets.token_budget_max),
        "noise": _budget_exhausted(budgets.noise_budget_used, budgets.noise_budget_max),
        "risk": _budget_exhausted(budgets.risk_budget_used, budgets.risk_budget_max),
    }
    return {
        "operation_budget_used": budgets.operation_budget_used,
        "operation_budget_max": budgets.operation_budget_max,
        "time_budget_used_sec": budgets.time_budget_used_sec,
        "time_budget_max_sec": budgets.time_budget_max_sec,
        "token_budget_used": budgets.token_budget_used,
        "token_budget_max": budgets.token_budget_max,
        "noise_budget_used": budgets.noise_budget_used,
        "noise_budget_max": budgets.noise_budget_max,
        "risk_budget_used": budgets.risk_budget_used,
        "risk_budget_max": budgets.risk_budget_max,
        "requires_human_review": any(guards.values()),
        "guards": guards,
    }


def _budget_exhausted(used: float | int, maximum: float | int | None) -> bool:
    return maximum is not None and used >= maximum


def _metadata_list(state: RuntimeState, key: str) -> list[dict[str, Any]]:
    value = state.execution.metadata.get(key, [])
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _metadata_mapping(state: RuntimeState, key: str) -> dict[str, Any]:
    value = state.execution.metadata.get(key, {})
    return dict(value) if isinstance(value, dict) else {}


def _recent_items(items: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return [dict(item) for item in items[-limit:]]


def _coerce_limit(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 100
    return max(0, parsed)


def _coerce_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sanitize(value: Any, *, key: str | None = None) -> Any:
    if key is not None:
        lowered = key.lower()
        if _is_sensitive_key(lowered) and isinstance(value, str):
            return _REDACTED_VALUE
        if any(marker in lowered for marker in _BLOCKED_KEY_MARKERS):
            return _REDACTED_VALUE

    if isinstance(value, dict):
        return {str(item_key): _sanitize(item_value, key=str(item_key)) for item_key, item_value in value.items()}

    if isinstance(value, list):
        return [_sanitize(item, key=key) for item in value]

    if isinstance(value, tuple):
        return [_sanitize(item, key=key) for item in value]

    if isinstance(value, str):
        redacted = _redact_inline_secrets(value)
        limit = _LONG_CONTEXT_LIMIT if key and any(marker in key.lower() for marker in _LONG_CONTEXT_KEY_MARKERS) else _DEFAULT_STRING_LIMIT
        if len(redacted) <= limit:
            return redacted
        return f"{redacted[:limit]}...(truncated {len(redacted) - limit} chars)"

    return value


def _is_sensitive_key(lowered_key: str) -> bool:
    if any(marker in lowered_key for marker in _SENSITIVE_KEY_MARKERS):
        return True
    return lowered_key == "token" or lowered_key.endswith("_token") or lowered_key.endswith("-token")


def _redact_inline_secrets(text: str) -> str:
    redacted = text
    for pattern in _INLINE_SECRET_PATTERNS:
        redacted = pattern.sub(r"\1[REDACTED]", redacted)
    return redacted


__all__ = ["build_operation_audit_report"]
