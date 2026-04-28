"""Lightweight runtime observability and recovery helpers."""

from __future__ import annotations

import re
from typing import Any

from src.core.models.runtime import (
    LockStatus,
    ReplayPlanRuntime,
    ReplayPlanStatus,
    RuntimeEventRef,
    RuntimeState,
    SessionStatus,
    TaskRuntimeStatus,
    WorkerStatus,
    utc_now,
)

_REDACTED_VALUE = "[REDACTED]"
_DEFAULT_AUDIT_MAX_ENTRIES = 200
_DEFAULT_OPERATION_LOG_MAX_ENTRIES = 200
_DEFAULT_AUDIT_STRING_LIMIT = 256
_DEFAULT_COMMAND_LIMIT = 512
_DEFAULT_COLLECTION_LIMIT = 20
_DEFAULT_MAPPING_LIMIT = 40
_SENSITIVE_FIELD_MARKERS = ("token", "password", "secret", "api_key", "authorization")
_INLINE_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)([^\s,;]+)"),
    re.compile(r"(?i)(\b(?:token|password|secret|api[_-]?key)\b\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r'(?i)(["\']?(?:token|password|secret|api[_-]?key|authorization)["\']?\s*[:=]\s*["\'])([^"\']+)(["\'])'),
)


def append_operation_log(state: RuntimeState, *, event_type: str, **payload: Any) -> dict[str, Any]:
    """Append one structured operation-level log entry to runtime metadata."""

    control_plane = _control_plane_settings(state)
    entries = state.execution.metadata.setdefault("operation_log", [])
    next_seq = _next_counter(state, key="_operation_log_seq")
    entry = {
        "seq": next_seq,
        "at": utc_now().isoformat(),
        "event_type": event_type,
        **_sanitize_log_payload(
            payload,
            redaction_enabled=control_plane["audit_redaction_enabled"],
        ),
    }
    entries.append(entry)
    _trim_log_entries(entries, max_entries=control_plane["operation_log_max_entries"])
    return entry


def append_audit_log(state: RuntimeState, entry: dict[str, Any]) -> dict[str, Any]:
    """Append one structured audit entry to runtime metadata."""

    control_plane = _control_plane_settings(state)
    audit_log = state.execution.metadata.setdefault("audit_log", [])
    item = {
        "audit_id": f"audit-{_next_counter(state, key='_audit_log_seq')}",
        "at": utc_now().isoformat(),
        **_sanitize_log_payload(
            entry,
            redaction_enabled=control_plane["audit_redaction_enabled"],
        ),
    }
    audit_log.append(item)
    _trim_log_entries(audit_log, max_entries=control_plane["audit_max_entries"])
    return item


def build_audit_report(state: RuntimeState) -> dict[str, Any]:
    """Build one exportable audit document from a runtime snapshot."""

    control_plane = _control_plane_settings(state)
    report = {
        "operation_id": state.operation_id,
        "exported_at": utc_now().isoformat(),
        "operation_status": state.operation_status.value,
        "event_cursor": state.event_cursor,
        "operation_log": list(state.execution.metadata.get("operation_log", [])),
        "audit_log": list(state.execution.metadata.get("audit_log", [])),
        "recent_outcomes": [item.model_dump(mode="json") for item in state.recent_outcomes],
        "pending_events": [item.model_dump(mode="json") for item in state.pending_events],
        "control_cycle_history": list(state.execution.metadata.get("control_cycle_history", [])),
        "phase_checkpoints": list(state.execution.metadata.get("phase_checkpoints", [])),
    }
    # 中文注释：
    # audit 导出需要对历史遗留的原始 metadata 再做一次统一净化，保证文件落盘和 API 导出
    # 拿到的永远是稳定、脱敏后的结果，而不是依赖写入方是否已经升级。
    return _sanitize_log_payload(report, redaction_enabled=control_plane["audit_redaction_enabled"])


def build_recovery_snapshot(state: RuntimeState) -> dict[str, Any]:
    """Build one small recovery-oriented view for persistence and resume."""

    recovery = _recovery_metadata(state)
    inflight_tasks = [
        task.task_id
        for task in state.execution.tasks.values()
        if task.status in {TaskRuntimeStatus.CLAIMED, TaskRuntimeStatus.RUNNING}
    ]
    return {
        "operation_id": state.operation_id,
        "captured_at": utc_now().isoformat(),
        "operation_status": state.operation_status.value,
        "event_cursor": state.event_cursor,
        "last_updated": state.last_updated.isoformat(),
        "unclean_shutdown": bool(recovery.get("unclean_shutdown", False)),
        "inflight_task_ids": inflight_tasks,
        "recovery_metadata": dict(recovery),
        "last_phase_checkpoint": dict(recovery.get("last_phase_checkpoint", {})),
        "replay_plan": dict(state.execution.metadata.get("replay_plan", {})),
        "pending_event_count": len(state.pending_events),
        "replan_request_count": len(state.replan_requests),
    }


def record_phase_checkpoint(
    state: RuntimeState,
    *,
    cycle_index: int,
    phase: str,
    status: str,
    selected_task_ids: list[str] | None = None,
    applied_task_ids: list[str] | None = None,
    runtime_event_count: int | None = None,
    step_count: int | None = None,
    success: bool | None = None,
    stopped: bool | None = None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    """Record one lightweight phase checkpoint into runtime metadata."""

    now = utc_now().isoformat()
    checkpoint = {
        "cycle_index": cycle_index,
        "phase": phase,
        "status": status,
        "at": now,
        "selected_task_ids": list(selected_task_ids or []),
        "applied_task_ids": list(applied_task_ids or []),
        "runtime_event_count": int(runtime_event_count or 0),
    }
    if step_count is not None:
        checkpoint["step_count"] = step_count
    if success is not None:
        checkpoint["success"] = success
    if stopped is not None:
        checkpoint["stopped"] = stopped
    if stop_reason is not None:
        checkpoint["stop_reason"] = stop_reason

    # 中文注释：
    # phase checkpoint 是“轻量边界快照”，既写 operation_log 供审计，也写 recovery metadata
    # 供 crash 后判断主循环上次停在哪个阶段。
    append_operation_log(
        state,
        event_type="phase_checkpoint",
        checkpoint_status=status,
        **checkpoint,
    )
    checkpoints = state.execution.metadata.setdefault("phase_checkpoints", [])
    checkpoints.append(dict(checkpoint))
    state.execution.metadata["last_phase_checkpoint"] = dict(checkpoint)

    recovery = _recovery_metadata(state)
    recovery["last_phase"] = phase
    recovery["last_phase_status"] = status
    recovery["last_phase_at"] = now
    recovery["last_phase_checkpoint"] = dict(checkpoint)
    recovery["phase_checkpoint_count"] = len(checkpoints)
    state.last_updated = utc_now()
    return checkpoint


def prepare_state_for_resume(
    state: RuntimeState,
    *,
    reason: str,
    event_refs: list[RuntimeEventRef] | None = None,
) -> dict[str, Any]:
    """Normalize in-flight runtime fields so the operation can safely resume."""

    now = utc_now()
    resumed_task_ids: list[str] = []
    resumed_worker_ids: list[str] = []
    released_lock_ids: list[str] = []
    expired_session_ids: list[str] = []

    # 中文注释：
    # crash 后 running/claimed task 的执行上下文不可信，统一退回 pending，
    # 避免直接沿用半写入的 started/deadline/worker 绑定继续执行。
    for task in state.execution.tasks.values():
        if task.status not in {TaskRuntimeStatus.CLAIMED, TaskRuntimeStatus.RUNNING}:
            continue
        task.status = TaskRuntimeStatus.PENDING
        task.assigned_worker = None
        task.started_at = None
        task.deadline = None
        task.metadata["resume_reason"] = reason
        task.metadata["resumed_at"] = now.isoformat()
        resumed_task_ids.append(task.task_id)

    # 中文注释：
    # 活动锁在 crash 后已失去 owner 连通性，保守策略是直接释放，
    # 避免下一轮 scheduler 因旧锁永久阻塞。
    for lock in state.locks.values():
        if lock.status != LockStatus.ACTIVE:
            continue
        lock.status = LockStatus.RELEASED
        lock.metadata["released_by_recovery"] = True
        lock.metadata["recovery_reason"] = reason
        released_lock_ids.append(lock.lock_key)

    # 中文注释：
    # 当前阶段不做事件重放，但先把未消费事件标记成“待恢复/可回放”结构，
    # 后续 replay 只需要消费这些 metadata，不必再倒推 crash 来源。
    for event in state.pending_events:
        recovery_meta = event.metadata.setdefault("recovery", {})
        recovery_meta["resume_reason"] = reason
        recovery_meta["recovered_at"] = now.isoformat()
        recovery_meta["replay_ready"] = True
        recovery_meta["requires_replay"] = True

    recovered_event_count = len(state.pending_events)
    replay_plan = plan_runtime_event_replay(state, reason=reason, event_refs=event_refs)

    # 中文注释：
    # 运行中的 session 句柄在 crash 后无法确认远端是否仍然可用，
    # 因此把 opening/active/draining 统一转成 expired，阻止脏 session 被复用。
    for session in state.sessions.values():
        if session.status in {SessionStatus.CLOSED, SessionStatus.FAILED, SessionStatus.EXPIRED}:
            continue
        session.status = SessionStatus.EXPIRED
        session.lease_expiry = now
        session.heartbeat_at = now
        session.failure_count += 1
        session.metadata["expiry_reason"] = reason
        session.metadata["expired_by_recovery"] = True
        session.metadata["bound_task_ids"] = []
        expired_session_ids.append(session.session_id)

    # 中文注释：
    # lease 视图要和 session/task 恢复结果保持一致：所有仍然活动的 lease 都在恢复点收口释放，
    # 否则 scheduler 会误以为 session 仍被旧 task 持有。
    for lease in state.session_leases.values():
        if "released_at" in lease.metadata:
            continue
        lease.lease_expiry = now
        lease.metadata.setdefault("released_at", now.isoformat())
        lease.metadata["release_reason"] = reason
        lease.metadata["released_by_recovery"] = True

    for worker in state.workers.values():
        if worker.status in {WorkerStatus.LOST, WorkerStatus.UNAVAILABLE}:
            continue
        if worker.current_task_id is not None or worker.status == WorkerStatus.BUSY:
            worker.current_task_id = None
            worker.status = WorkerStatus.IDLE
            worker.current_load = 0
            worker.metadata["resume_reason"] = reason
            worker.metadata["resumed_at"] = now.isoformat()
            resumed_worker_ids.append(worker.worker_id)
    metadata = _recovery_metadata(state)
    metadata["last_resume_at"] = now.isoformat()
    metadata["last_resume_reason"] = reason
    metadata["unclean_shutdown"] = False
    metadata["resumed_task_ids"] = resumed_task_ids
    metadata["resumed_worker_ids"] = resumed_worker_ids
    metadata["released_lock_ids"] = released_lock_ids
    metadata["expired_session_ids"] = expired_session_ids
    metadata["recovered_event_count"] = recovered_event_count
    metadata["pending_event_ids"] = [event.event_id for event in state.pending_events]
    metadata["replay_cursor"] = replay_plan["start_cursor"]
    metadata["last_replayed_cursor"] = replay_plan["last_replayed_cursor"]
    metadata["replay_required"] = replay_plan["replay_required"]
    metadata["replay_reason"] = replay_plan["replay_reason"]
    metadata["replay_status"] = replay_plan["replay_status"]
    metadata["replay_candidate_event_ids"] = list(replay_plan["replay_candidate_event_ids"])
    state.last_updated = now
    return {
        "resumed_task_ids": resumed_task_ids,
        "resumed_worker_ids": resumed_worker_ids,
        "released_lock_ids": released_lock_ids,
        "expired_session_ids": expired_session_ids,
        "recovered_event_count": recovered_event_count,
        "last_resume_reason": reason,
        "replay_status": replay_plan["replay_status"],
        "replay_candidate_event_ids": list(replay_plan["replay_candidate_event_ids"]),
    }


def mark_unclean_shutdown(state: RuntimeState, *, cycle_index: int) -> None:
    """Mark the operation as having entered a cycle that must be recovered if interrupted."""

    metadata = _recovery_metadata(state)
    metadata["unclean_shutdown"] = True
    metadata["last_cycle_index"] = cycle_index
    metadata["last_cycle_started_at"] = utc_now().isoformat()


def mark_clean_shutdown(state: RuntimeState, *, cycle_index: int) -> None:
    """Clear the pending recovery marker once the cycle snapshot is persisted."""

    metadata = _recovery_metadata(state)
    metadata["unclean_shutdown"] = False
    metadata["last_cycle_index"] = cycle_index
    metadata["last_cycle_completed_at"] = utc_now().isoformat()


def _recovery_metadata(state: RuntimeState) -> dict[str, Any]:
    value = state.execution.metadata.setdefault("recovery", {})
    return value if isinstance(value, dict) else {}


def plan_runtime_event_replay(
    state: RuntimeState,
    *,
    reason: str,
    event_refs: list[RuntimeEventRef] | None = None,
) -> dict[str, Any]:
    """Build and persist one lightweight replay plan without replaying side effects."""

    now = utc_now()
    recovery = _recovery_metadata(state)
    last_replayed_cursor = _coerce_positive_int(recovery.get("last_replayed_cursor"), default=0)
    ordered_pending_events = sorted(
        state.pending_events,
        key=lambda item: (item.cursor, item.created_at, item.event_id),
    )
    replay_window = sorted(
        [
            item.model_copy(deep=True)
            for item in (event_refs or state.pending_events)
            if item.cursor >= last_replayed_cursor
        ],
        key=lambda item: (item.cursor, item.created_at, item.event_id),
    )
    candidate_ids = [item.event_id for item in ordered_pending_events]
    candidate_cursors = [item.cursor for item in ordered_pending_events if item.cursor >= 0]
    window_cursors = [item.cursor for item in replay_window if item.cursor >= 0]
    start_cursor = max(
        last_replayed_cursor,
        min(candidate_cursors + window_cursors, default=last_replayed_cursor),
    )
    end_cursor = max(window_cursors + [state.event_cursor, last_replayed_cursor], default=state.event_cursor)
    status = ReplayPlanStatus.PLANNED if candidate_ids else ReplayPlanStatus.NOT_REQUIRED
    plan = ReplayPlanRuntime(
        plan_id=f"replay-plan-{_next_counter(state, key='_replay_plan_seq')}",
        replay_status=status,
        replay_reason=reason if candidate_ids else None,
        last_replayed_cursor=last_replayed_cursor,
        start_cursor=start_cursor,
        end_cursor=end_cursor,
        replay_candidate_event_ids=list(candidate_ids),
        pending_event_count=len(state.pending_events),
        metadata={
            "planned_at": now.isoformat(),
            "event_window_count": len(replay_window),
        },
    )

    # 中文注释：
    # 这里先把 replay planning 固化成独立结构，后续真正引入副作用重放时，
    # 只需要消费这份 plan/status，而不必重新设计 recovery snapshot。
    plan_payload = plan.model_dump(mode="json")
    state.execution.metadata["replay_plan"] = plan_payload

    candidate_rank_by_id = {
        event_id: index + 1
        for index, event_id in enumerate(candidate_ids)
    }
    for event in ordered_pending_events:
        replay_meta = event.metadata.setdefault("replay", {})
        replay_meta["plan_id"] = plan.plan_id
        replay_meta["replay_status"] = status.value
        replay_meta["replay_reason"] = reason
        replay_meta["planned_at"] = now.isoformat()
        replay_meta["candidate_rank"] = candidate_rank_by_id[event.event_id]
        replay_meta["last_replayed_cursor"] = last_replayed_cursor
        replay_meta["start_cursor"] = start_cursor

    recovery["last_replayed_cursor"] = last_replayed_cursor
    recovery["replay_required"] = bool(candidate_ids)
    recovery["replay_reason"] = reason if candidate_ids else None
    recovery["replay_status"] = status.value
    recovery["replay_candidate_event_ids"] = list(candidate_ids)
    recovery["replay_cursor"] = start_cursor
    recovery["replay_plan_id"] = plan.plan_id
    recovery["replay_planned_at"] = now.isoformat()
    return {
        "plan_id": plan.plan_id,
        "last_replayed_cursor": last_replayed_cursor,
        "start_cursor": start_cursor,
        "end_cursor": end_cursor,
        "replay_required": bool(candidate_ids),
        "replay_reason": reason if candidate_ids else None,
        "replay_status": status.value,
        "replay_candidate_event_ids": list(candidate_ids),
    }


def build_event_log_replay_annotations(
    state: RuntimeState,
    event_refs: list[RuntimeEventRef],
) -> dict[str, dict[str, Any]]:
    """Build lightweight replay annotations for persisted event-log rows."""

    recovery = _recovery_metadata(state)
    plan = state.execution.metadata.get("replay_plan", {})
    if not isinstance(plan, dict):
        plan = {}
    last_replayed_cursor = _coerce_positive_int(recovery.get("last_replayed_cursor"), default=0)
    candidate_ids = set(
        item
        for item in recovery.get("replay_candidate_event_ids", [])
        if isinstance(item, str) and item
    )
    if not event_refs or not plan:
        return {}

    annotations: dict[str, dict[str, Any]] = {}
    candidate_rank_by_id = {
        event_id: index + 1
        for index, event_id in enumerate(plan.get("replay_candidate_event_ids", []))
        if isinstance(event_id, str) and event_id
    }
    for event in event_refs:
        if event.cursor < last_replayed_cursor and event.event_id not in candidate_ids:
            continue
        annotations[event.event_id] = {
            "plan_id": plan.get("plan_id"),
            "last_replayed_cursor": last_replayed_cursor,
            "start_cursor": plan.get("start_cursor", last_replayed_cursor),
            "end_cursor": plan.get("end_cursor", state.event_cursor),
            "replay_reason": recovery.get("replay_reason"),
            "replay_status": "candidate" if event.event_id in candidate_ids else "replay_window",
            "candidate_rank": candidate_rank_by_id.get(event.event_id),
        }
    return annotations


def _control_plane_settings(state: RuntimeState) -> dict[str, Any]:
    value = state.execution.metadata.get("control_plane", {})
    control_plane = value if isinstance(value, dict) else {}
    return {
        "audit_max_entries": _coerce_positive_int(
            control_plane.get("audit_max_entries"),
            default=_DEFAULT_AUDIT_MAX_ENTRIES,
        ),
        "operation_log_max_entries": _coerce_positive_int(
            control_plane.get("operation_log_max_entries"),
            default=_DEFAULT_OPERATION_LOG_MAX_ENTRIES,
        ),
        "audit_redaction_enabled": bool(control_plane.get("audit_redaction_enabled", True)),
    }


def _coerce_positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _next_counter(state: RuntimeState, *, key: str) -> int:
    current = _coerce_positive_int(state.execution.metadata.get(key), default=0)
    next_value = current + 1
    state.execution.metadata[key] = next_value
    return next_value


def _trim_log_entries(entries: list[dict[str, Any]], *, max_entries: int) -> None:
    overflow = len(entries) - max_entries
    if overflow > 0:
        del entries[:overflow]


def _sanitize_log_payload(payload: Any, *, redaction_enabled: bool, field_name: str | None = None) -> Any:
    if field_name is not None and redaction_enabled and _is_sensitive_field(field_name):
        return _REDACTED_VALUE

    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        items = list(payload.items())
        existing_truncated_field_count = 0
        if "_truncated_field_count" in payload:
            existing_value = payload.get("_truncated_field_count")
            if isinstance(existing_value, int) and existing_value >= 0:
                existing_truncated_field_count = existing_value
            items = [(key, value) for key, value in items if key != "_truncated_field_count"]
        for index, (key, value) in enumerate(items):
            if index >= _DEFAULT_MAPPING_LIMIT:
                sanitized["_truncated_field_count"] = (len(items) - _DEFAULT_MAPPING_LIMIT) + existing_truncated_field_count
                break
            sanitized[key] = _sanitize_log_payload(
                value,
                redaction_enabled=redaction_enabled,
                field_name=str(key),
            )
        if existing_truncated_field_count and "_truncated_field_count" not in sanitized:
            sanitized["_truncated_field_count"] = existing_truncated_field_count
        return sanitized

    if isinstance(payload, list):
        limit = _collection_limit_for(field_name)
        items = list(payload)
        existing_truncated_item_count = 0
        if items and isinstance(items[-1], dict) and set(items[-1].keys()) == {"truncated_item_count"}:
            existing_value = items[-1].get("truncated_item_count")
            if isinstance(existing_value, int) and existing_value >= 0:
                existing_truncated_item_count = existing_value
            items = items[:-1]
        sanitized_items = [
            _sanitize_log_payload(item, redaction_enabled=redaction_enabled, field_name=field_name)
            for item in items[:limit]
        ]
        overflow = max(0, len(items) - limit)
        total_truncated = overflow + existing_truncated_item_count
        if total_truncated > 0:
            sanitized_items.append({"truncated_item_count": total_truncated})
        return sanitized_items

    if isinstance(payload, tuple):
        return _sanitize_log_payload(list(payload), redaction_enabled=redaction_enabled, field_name=field_name)

    if isinstance(payload, str):
        text = _redact_sensitive_text(payload) if redaction_enabled else payload
        return _truncate_text(text, limit=_string_limit_for(field_name))

    return payload


def _is_sensitive_field(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _SENSITIVE_FIELD_MARKERS)


def _redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern in _INLINE_SECRET_PATTERNS:
        redacted = pattern.sub(_replace_secret_match, redacted)
    return redacted


def _replace_secret_match(match: re.Match[str]) -> str:
    if match.lastindex == 3:
        return f"{match.group(1)}{_REDACTED_VALUE}{match.group(3)}"
    return f"{match.group(1)}{_REDACTED_VALUE}"


def _truncate_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(truncated {len(text) - limit} chars)"


def _string_limit_for(field_name: str | None) -> int:
    if field_name == "command":
        return _DEFAULT_COMMAND_LIMIT
    return _DEFAULT_AUDIT_STRING_LIMIT


def _collection_limit_for(field_name: str | None) -> int:
    if field_name in {"operation_log", "audit_log"}:
        return _DEFAULT_AUDIT_MAX_ENTRIES
    return _DEFAULT_COLLECTION_LIMIT
