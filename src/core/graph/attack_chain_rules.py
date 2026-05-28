"""Deterministic follow-up task rules for the runtime attack-chain loop."""

from __future__ import annotations

from typing import Any, Iterable

from src.core.graph.tg_builder import TaskCandidate
from src.core.models.ag import GraphRef
from src.core.models.events import AgentTaskResult
from src.core.models.tg import TaskType


def generate_followup_task_candidates_from_result(result: AgentTaskResult) -> list[TaskCandidate]:
    """Build bounded TG candidates from canonical worker results."""

    outcome_type = _string(result.outcome_payload.get("outcome_type")) or _string(result.outcome_payload.get("kind"))
    if outcome_type is None:
        return []
    candidates: list[TaskCandidate] = []
    if outcome_type == "vulnerability_validation" and bool(result.outcome_payload.get("validated")):
        candidates.append(_session_context_candidate(result))
    if outcome_type in {"credential_validation", "credential_reuse_validation"} and _authenticated(result):
        candidates.append(_privilege_context_candidate(result))
        candidates.append(_lateral_reachability_candidate(result))
    if outcome_type in {"pivot_route_validation", "lateral_reachability_validation"} and _reachable(result):
        candidates.append(_internal_service_candidate(result))
    return [candidate for candidate in candidates if candidate is not None]


def _session_context_candidate(result: AgentTaskResult) -> TaskCandidate | None:
    service_id = _string(result.outcome_payload.get("service_id")) or _ref_id(result, "Service")
    if service_id is None:
        return None
    return TaskCandidate(
        source_action_id=f"chain::{result.task_id}::session-context::{service_id}",
        task_type=TaskType.IDENTITY_CONTEXT_CONFIRMATION,
        input_bindings={"service_id": service_id, "vulnerability_id": result.outcome_payload.get("vulnerability_id")},
        target_refs=_target_refs(result),
        resource_keys={f"service:{service_id}"},
        estimated_risk=0.1,
        estimated_noise=0.1,
        goal_relevance=0.6,
        tags={"attack_chain", "session_context"},
    )


def _privilege_context_candidate(result: AgentTaskResult) -> TaskCandidate | None:
    target_id = _credential_target(result)
    if target_id is None:
        return None
    return TaskCandidate(
        source_action_id=f"chain::{result.task_id}::privilege-context::{target_id}",
        task_type=TaskType.PRIVILEGE_CONFIGURATION_VALIDATION,
        input_bindings={"target_id": target_id, "credential_id": _credential_id(result)},
        target_refs=_target_refs(result),
        resource_keys={f"target:{target_id}"},
        estimated_risk=0.1,
        estimated_noise=0.05,
        goal_relevance=0.7,
        tags={"attack_chain", "privilege_context"},
    )


def _lateral_reachability_candidate(result: AgentTaskResult) -> TaskCandidate | None:
    target_id = _credential_target(result)
    if target_id is None:
        return None
    return TaskCandidate(
        source_action_id=f"chain::{result.task_id}::lateral-reachability::{target_id}",
        task_type=TaskType.LATERAL_REACHABILITY_VALIDATION,
        input_bindings={"source_host": target_id, "credential_id": _credential_id(result)},
        target_refs=_target_refs(result),
        resource_keys={f"target:{target_id}"},
        estimated_risk=0.1,
        estimated_noise=0.1,
        goal_relevance=0.65,
        tags={"attack_chain", "lateral_reachability"},
    )


def _internal_service_candidate(result: AgentTaskResult) -> TaskCandidate | None:
    destination_host = _string(result.outcome_payload.get("destination_host"))
    reachability = _dict(result.outcome_payload.get("reachability"))
    selected_route = _dict(result.outcome_payload.get("selected_route"))
    destination_host = destination_host or _string(reachability.get("target_id")) or _string(selected_route.get("destination_host"))
    if destination_host is None:
        return None
    return TaskCandidate(
        source_action_id=f"chain::{result.task_id}::internal-service::{destination_host}",
        task_type=TaskType.INTERNAL_SERVICE_FINGERPRINT,
        input_bindings={
            "host_id": destination_host,
            "route_id": selected_route.get("route_id") or reachability.get("route_id"),
            "port": _first(selected_route.get("allowed_ports") or reachability.get("allowed_ports") or reachability.get("port")),
        },
        target_refs=_target_refs(result),
        resource_keys={f"host:{destination_host}"},
        estimated_risk=0.05,
        estimated_noise=0.1,
        goal_relevance=0.6,
        tags={"attack_chain", "internal_service_discovery"},
    )


def _authenticated(result: AgentTaskResult) -> bool:
    credential = _dict(result.outcome_payload.get("credential_validation"))
    if credential:
        return bool(credential.get("authenticated") or credential.get("validated") or credential.get("status") == "valid")
    payload = _dict(result.outcome_payload.get("payload"))
    credential = _dict(payload.get("credential_validation"))
    return bool(credential.get("authenticated") or credential.get("validated") or credential.get("status") == "valid")


def _reachable(result: AgentTaskResult) -> bool:
    reachability = _dict(result.outcome_payload.get("reachability"))
    if "reachable" in reachability:
        return bool(reachability.get("reachable"))
    payload = _dict(result.outcome_payload.get("payload"))
    reachability = _dict(payload.get("reachability"))
    if "reachable" in reachability:
        return bool(reachability.get("reachable"))
    return bool(result.outcome_payload.get("reachable"))


def _credential_target(result: AgentTaskResult) -> str | None:
    credential = _dict(result.outcome_payload.get("credential_validation"))
    payload = _dict(result.outcome_payload.get("payload"))
    if not credential:
        credential = _dict(payload.get("credential_validation"))
    return (
        _string(credential.get("target_id"))
        or _string(credential.get("target_service_id"))
        or _string(credential.get("bound_target"))
        or _string(result.outcome_payload.get("target_id"))
        or _ref_id(result, "Service")
        or _ref_id(result, "Host")
    )


def _credential_id(result: AgentTaskResult) -> str | None:
    credential = _dict(result.outcome_payload.get("credential_validation"))
    payload = _dict(result.outcome_payload.get("payload"))
    if not credential:
        credential = _dict(payload.get("credential_validation"))
    return _string(credential.get("credential_id")) or _string(result.outcome_payload.get("credential_id"))


def _target_refs(result: AgentTaskResult) -> list[GraphRef]:
    refs: list[GraphRef] = []
    for collection in (result.observations, result.evidence):
        for item in collection:
            refs.extend(ref for ref in getattr(item, "refs", []) if isinstance(ref, GraphRef))
    return list({ref.key(): ref for ref in refs}.values())


def _ref_id(result: AgentTaskResult, ref_type: str) -> str | None:
    expected = ref_type.lower()
    for ref in _iter_refs(result):
        if str(ref.ref_type or "").lower() == expected:
            return ref.ref_id
    return None


def _iter_refs(result: AgentTaskResult) -> Iterable[GraphRef]:
    for collection in (result.observations, result.evidence):
        for item in collection:
            for ref in getattr(item, "refs", []):
                if isinstance(ref, GraphRef):
                    yield ref


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        return next(iter(value), None)
    return value


def _string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["generate_followup_task_candidates_from_result"]
