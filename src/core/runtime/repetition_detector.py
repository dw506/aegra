"""Deterministic repetition detection for task proposals.

The detector intentionally avoids embeddings. It builds a stable task
signature from bounded structured fields, then compares exact hashes and
token-level similarity against prior terminal task outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.graph.tg_builder import TaskCandidate
from src.core.models.ag import GraphRef
from src.core.models.tg import BaseTaskNode, TaskStatus


class RepetitionAction(str, Enum):
    """Planner-facing action returned by the detector."""

    ALLOW = "allow"
    WARN = "warn"
    REJECT = "reject"


class RepetitionDecision(BaseModel):
    """Decision produced for one proposed task."""

    model_config = ConfigDict(extra="forbid")

    action: RepetitionAction = RepetitionAction.ALLOW
    reason: str = "no repetition found"
    matched_task_ids: list[str] = Field(default_factory=list)
    signature_hash: str | None = None
    similarity: float = 0.0

    @property
    def allow(self) -> bool:
        return self.action != RepetitionAction.REJECT

    @property
    def warn(self) -> bool:
        return self.action == RepetitionAction.WARN

    @property
    def reject(self) -> bool:
        return self.action == RepetitionAction.REJECT


class TaskSignature(BaseModel):
    """Normalized task identity used for exact and fuzzy comparison."""

    model_config = ConfigDict(extra="forbid")

    task_type: str
    target_refs: list[str] = Field(default_factory=list)
    tool_hint: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    expected_evidence: list[str] = Field(default_factory=list)

    @property
    def hash(self) -> str:
        payload = self.model_dump(mode="json")
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @property
    def tokens(self) -> set[str]:
        return _tokens(self.model_dump(mode="json"))


class RepetitionRecord(BaseModel):
    """Terminal historical task signature."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: str
    signature: TaskSignature
    signature_hash: str

    @property
    def is_failed_like(self) -> bool:
        return _status(self.status) in {"failed", "blocked"}

    @property
    def is_success_like(self) -> bool:
        return _status(self.status) in {"completed", "succeeded", "success"}


class RepetitionDetector:
    """Track terminal task signatures and reject repeated failures."""

    def __init__(
        self,
        records: Iterable[RepetitionRecord | dict[str, Any]] | None = None,
        *,
        similarity_threshold: float = 0.95,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self._records: list[RepetitionRecord] = []
        self._by_hash: dict[str, list[RepetitionRecord]] = {}
        for record in records or []:
            self.add_record(record)

    @property
    def records(self) -> list[RepetitionRecord]:
        return list(self._records)

    def add_record(self, record: RepetitionRecord | dict[str, Any]) -> RepetitionRecord:
        resolved = self._coerce_record(record)
        self._records.append(resolved)
        self._by_hash.setdefault(resolved.signature_hash, []).append(resolved)
        return resolved

    def record_task(
        self,
        task: TaskCandidate | BaseTaskNode | dict[str, Any],
        *,
        task_id: str | None = None,
        status: str | TaskStatus,
    ) -> RepetitionRecord:
        signature = self.signature_for(task)
        return self.add_record(
            RepetitionRecord(
                task_id=task_id or _task_id(task),
                status=_status(status),
                signature=signature,
                signature_hash=signature.hash,
            )
        )

    def decide(self, task: TaskCandidate | BaseTaskNode | dict[str, Any]) -> RepetitionDecision:
        signature = self.signature_for(task)
        exact_matches = self._by_hash.get(signature.hash, [])
        failed_exact = [record for record in exact_matches if record.is_failed_like]
        if failed_exact:
            return RepetitionDecision(
                action=RepetitionAction.REJECT,
                reason="exact repeat of failed or blocked task",
                matched_task_ids=[record.task_id for record in failed_exact],
                signature_hash=signature.hash,
                similarity=1.0,
            )

        success_exact = [record for record in exact_matches if record.is_success_like]
        if success_exact:
            return RepetitionDecision(
                action=RepetitionAction.WARN,
                reason="exact repeat of completed task",
                matched_task_ids=[record.task_id for record in success_exact],
                signature_hash=signature.hash,
                similarity=1.0,
            )

        best_similarity = 0.0
        best_records: list[RepetitionRecord] = []
        tokens = signature.tokens
        for record in self._records:
            if not record.is_failed_like:
                continue
            similarity = _jaccard(tokens, record.signature.tokens)
            if similarity > best_similarity:
                best_similarity = similarity
                best_records = [record]
            elif similarity == best_similarity and similarity > 0:
                best_records.append(record)

        if best_similarity >= self.similarity_threshold and best_records:
            return RepetitionDecision(
                action=RepetitionAction.REJECT,
                reason="similar to failed or blocked task",
                matched_task_ids=[record.task_id for record in best_records],
                signature_hash=signature.hash,
                similarity=round(best_similarity, 6),
            )

        return RepetitionDecision(
            action=RepetitionAction.ALLOW,
            reason="no repetition found",
            signature_hash=signature.hash,
            similarity=round(best_similarity, 6),
        )

    @classmethod
    def signature_for(cls, task: TaskCandidate | BaseTaskNode | dict[str, Any]) -> TaskSignature:
        payload = _task_payload(task)
        input_bindings = _mapping(payload.get("input_bindings"))
        params = _normalized_params(input_bindings)
        expected_evidence = _expected_evidence(payload, input_bindings)
        return TaskSignature(
            task_type=str(payload.get("task_type") or "").lower(),
            target_refs=_ref_keys(payload.get("target_refs")),
            tool_hint=_tool_hint(input_bindings),
            params=params,
            expected_evidence=expected_evidence,
        )

    def _coerce_record(self, record: RepetitionRecord | dict[str, Any]) -> RepetitionRecord:
        if isinstance(record, RepetitionRecord):
            return record
        data = dict(record)
        if isinstance(data.get("signature"), TaskSignature):
            signature = data["signature"]
        elif isinstance(data.get("signature"), dict):
            signature = TaskSignature.model_validate(data["signature"])
        elif "task" in data:
            signature = self.signature_for(data["task"])
        else:
            signature = self.signature_for(data)
        return RepetitionRecord(
            task_id=str(data.get("task_id") or data.get("id") or _task_id(data)),
            status=_status(data.get("status") or data.get("outcome") or "failed"),
            signature=signature,
            signature_hash=str(data.get("signature_hash") or signature.hash),
        )


def repetition_summary(events: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count repetition detector decisions in audit-like event payloads."""

    counts = {"allow": 0, "warn": 0, "reject": 0}
    for event in events:
        decision = _find_repetition_action(event)
        if decision in counts:
            counts[decision] += 1
    counts["rejection_count"] = counts["reject"]
    return counts


def _find_repetition_action(value: Any) -> str | None:
    if isinstance(value, dict):
        if value.get("event_type") == "repetition_detected":
            action = value.get("action") or value.get("decision")
            return str(action).lower() if action is not None else None
        candidate = value.get("repetition_decision") or value.get("repetition")
        if isinstance(candidate, dict):
            action = candidate.get("action") or candidate.get("decision")
            if action is not None:
                return str(action).lower()
        for nested in value.values():
            found = _find_repetition_action(nested)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_repetition_action(item)
            if found:
                return found
    return None


def _task_payload(task: TaskCandidate | BaseTaskNode | dict[str, Any]) -> dict[str, Any]:
    if isinstance(task, (TaskCandidate, BaseTaskNode)):
        return task.model_dump(mode="json")
    return dict(task)


def _task_id(task: TaskCandidate | BaseTaskNode | dict[str, Any]) -> str:
    payload = _task_payload(task)
    return str(payload.get("id") or payload.get("task_id") or payload.get("source_action_id") or "unknown-task")


def _status(status: str | TaskStatus | Any) -> str:
    raw = status.value if hasattr(status, "value") else status
    normalized = str(raw or "").strip().lower()
    if normalized == "blocked":
        return "blocked"
    if normalized in {"completed", "complete", "succeeded", "success"}:
        return "succeeded"
    if normalized in {"failed", "failure", "timed_out", "timeout"}:
        return "failed"
    return normalized


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _tool_hint(input_bindings: dict[str, Any]) -> str | None:
    for key in ("tool_hint", "tool", "tool_name", "recipe_id", "command_name"):
        value = input_bindings.get(key)
        if value is not None and str(value).strip():
            return _normalize_scalar(value)
    return None


def _expected_evidence(payload: dict[str, Any], input_bindings: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for ref in _ref_keys(payload.get("expected_output_refs")):
        values.append(ref)
    for key in ("expected_evidence", "expected_output", "evidence_hint"):
        value = input_bindings.get(key)
        if isinstance(value, list):
            values.extend(_normalize_scalar(item) for item in value if item is not None)
        elif value is not None:
            values.append(_normalize_scalar(value))
    return sorted({item for item in values if item})


def _normalized_params(input_bindings: dict[str, Any]) -> dict[str, Any]:
    ignored = {
        "created_at",
        "updated_at",
        "timestamp",
        "nonce",
        "attempt",
        "attempt_count",
        "retry",
        "request_id",
        "trace_id",
        "graph_llm_proposal_id",
        "graph_llm_task_proposal_id",
        "expected_evidence",
        "expected_output",
        "evidence_hint",
        "tool",
        "tool_hint",
        "tool_name",
        "recipe_id",
        "command_name",
    }
    return {
        str(key).lower(): _normalize_value(value)
        for key, value in sorted(input_bindings.items(), key=lambda item: str(item[0]))
        if str(key).lower() not in ignored and value is not None
    }


def _normalize_value(value: Any) -> Any:
    if isinstance(value, GraphRef):
        return value.key()
    if isinstance(value, dict):
        if {"graph", "ref_id"}.issubset(value.keys()):
            return _ref_key(value)
        return {
            str(key).lower(): _normalize_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if item is not None
        }
    if isinstance(value, (list, tuple, set)):
        return sorted(_normalize_value(item) for item in value if item is not None)
    if isinstance(value, str):
        return _normalize_scalar(value)
    return value


def _normalize_scalar(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _ref_keys(value: Any) -> list[str]:
    if value is None:
        return []
    refs = value if isinstance(value, list) else [value]
    return sorted({_ref_key(ref) for ref in refs if _ref_key(ref)})


def _ref_key(ref: Any) -> str:
    if isinstance(ref, GraphRef):
        return ref.key()
    if isinstance(ref, dict):
        graph = str(ref.get("graph") or "").lower()
        ref_id = str(ref.get("ref_id") or "")
        ref_type = str(ref.get("ref_type") or "")
        if graph and ref_id:
            return ":".join(item for item in (graph, ref_type.lower(), ref_id.lower()) if item)
    return _normalize_scalar(ref) if ref is not None else ""


def _tokens(value: Any) -> set[str]:
    text = json.dumps(value, sort_keys=True, default=str) if not isinstance(value, str) else value
    return {token for token in re.findall(r"[a-z0-9_.:/-]+", text.lower()) if token}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


__all__ = [
    "RepetitionAction",
    "RepetitionDecision",
    "RepetitionDetector",
    "RepetitionRecord",
    "TaskSignature",
    "repetition_summary",
]
