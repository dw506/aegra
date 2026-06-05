"""Adapters between StageResult and the canonical AgentTaskResult protocol."""

from __future__ import annotations

from typing import Any

from src.core.models.ag import GraphRef
from src.core.models.events import (
    AgentResultStatus,
    AgentRole,
    AgentTaskResult,
    EvidenceArtifact,
    FactWriteKind,
    FactWriteRequest,
    ObservationRecord,
    ReplanHint,
    ReplanScope,
    RuntimeControlRequest,
    RuntimeControlType,
    new_protocol_id,
)
from src.core.stage.models import StageResult, normalize_stage_name


class StageResultAdapter:
    """Convert StageResult into the existing ResultApplier protocol."""

    ROLE_BY_STAGE = {
        "RECON_STAGE": AgentRole.RECON_STAGE_AGENT,
        "VULN_ANALYSIS_STAGE": AgentRole.VULN_ANALYSIS_STAGE_AGENT,
        "EXPLOIT_STAGE": AgentRole.EXPLOIT_STAGE_AGENT,
        "ACCESS_PIVOT_STAGE": AgentRole.ACCESS_PIVOT_STAGE_AGENT,
        "GOAL_STAGE": AgentRole.GOAL_STAGE_AGENT,
    }

    STATUS_MAP = {
        "success": AgentResultStatus.SUCCEEDED,
        "succeeded": AgentResultStatus.SUCCEEDED,
        "failed": AgentResultStatus.FAILED,
        "partial": AgentResultStatus.SUCCEEDED,
        "blocked": AgentResultStatus.NEEDS_REPLAN,
        "need_more_info": AgentResultStatus.NEEDS_REPLAN,
        "needs_replan": AgentResultStatus.NEEDS_REPLAN,
    }

    @classmethod
    def to_task_result(cls, stage_result: StageResult) -> AgentTaskResult:
        observations = cls._observations(stage_result)
        evidence = cls._evidence(stage_result)
        fact_writes = cls._fact_writes(stage_result)
        runtime_requests = cls._runtime_requests(stage_result)
        replan_hints = []
        if stage_result.status in {"needs_replan", "blocked", "need_more_info"}:
            replan_hints.append(
                ReplanHint(
                    source_task_id=stage_result.stage_task_id,
                    scope=ReplanScope.LOCAL,
                    reason=stage_result.replan_recommendation or stage_result.summary,
                    task_ids=[stage_result.stage_task_id],
                    metadata={"stage_type": stage_result.stage_type},
                )
            )
        return AgentTaskResult(
            request_id=f"stage-{stage_result.stage_task_id}",
            agent_role=cls.ROLE_BY_STAGE[normalize_stage_name(stage_result.stage_type)],
            operation_id=stage_result.operation_id,
            task_id=stage_result.stage_task_id,
            execution_node_id=stage_result.stage_task_id,
            status=cls.STATUS_MAP[stage_result.status],
            summary=stage_result.summary,
            observations=observations,
            evidence=evidence,
            fact_write_requests=fact_writes,
            runtime_requests=runtime_requests,
            replan_hints=replan_hints,
            outcome_payload={
                "outcome_type": stage_result.stage_type,
                "stage_result": stage_result.model_dump(mode="json"),
                "runtime_hints": dict(stage_result.runtime_hints),
                "writeback_hints": dict(stage_result.writeback_hints),
                "tool_trace": [item.model_dump(mode="json") for item in stage_result.tool_trace],
                "graph_update_intents": [item.model_dump(mode="json") for item in stage_result.graph_update_intents],
                "policy_notes": list(stage_result.policy_notes),
                "retry_recommendation": stage_result.retry_recommendation,
                "replan_recommendation": stage_result.replan_recommendation,
                "next_stage_suggestion": stage_result.next_stage_suggestion,
                "next_stage_candidates": list(stage_result.next_stage_candidates),
            },
            metadata={
                "adapted_from": "stage_result",
                "stage_type": stage_result.stage_type,
                "capabilities_gained": list(stage_result.capabilities_gained),
                "failed_hypotheses": list(stage_result.failed_hypotheses),
            },
        )

    @classmethod
    def _observations(cls, stage_result: StageResult) -> list[ObservationRecord]:
        observations = [
            ObservationRecord(
                category=str(item.get("category") or stage_result.stage_type.lower()),
                summary=str(item.get("summary") or item.get("description") or stage_result.summary),
                confidence=cls._confidence(item),
                refs=cls._refs(item.get("refs")),
                payload={key: value for key, value in item.items() if key not in {"category", "summary", "refs"}},
            )
            for item in stage_result.observations
            if isinstance(item, dict)
        ]
        for hint in cls._runtime_hint_records(stage_result):
            observations.append(
                ObservationRecord(
                    category="stage_runtime_hint",
                    summary=f"Runtime hint from {stage_result.agent_name}",
                    confidence=cls._confidence(hint),
                    payload={"runtime_hints": hint},
                )
            )
        if stage_result.tool_trace:
            observations.append(
                ObservationRecord(
                    category="stage_tool_trace",
                    summary=f"{len(stage_result.tool_trace)} tool call(s) executed by {stage_result.agent_name}",
                    confidence=0.8,
                    payload={"tool_trace": [item.model_dump(mode="json") for item in stage_result.tool_trace]},
                )
            )
        return observations

    @classmethod
    def _evidence(cls, stage_result: StageResult) -> list[EvidenceArtifact]:
        evidence = [
            EvidenceArtifact(
                evidence_id=str(item.get("evidence_id") or new_protocol_id("evidence")),
                kind=str(item.get("kind") or stage_result.stage_type.lower()),
                summary=str(item.get("summary") or stage_result.summary),
                payload_ref=str(item.get("payload_ref") or f"runtime://stage-results/{stage_result.stage_task_id}"),
                tool_output_ref=item.get("tool_output_ref"),
                refs=cls._refs(item.get("refs")),
                metadata={key: value for key, value in item.items() if key not in {"evidence_id", "kind", "summary", "payload_ref", "tool_output_ref", "refs"}},
            )
            for item in stage_result.evidence
            if isinstance(item, dict)
        ]
        for trace in stage_result.tool_trace:
            evidence.append(
                EvidenceArtifact(
                    evidence_id=new_protocol_id("evidence"),
                    kind="stage_tool_trace",
                    summary=trace.summary or f"{trace.server_id}.{trace.tool_name}",
                    payload_ref=f"runtime://stage-results/{stage_result.stage_task_id}/tool-trace/{trace.step}",
                    tool_output_ref=f"runtime://stage-results/{stage_result.stage_task_id}/tool-trace/{trace.step}",
                    metadata=trace.model_dump(mode="json"),
                )
            )
        return evidence

    @classmethod
    def _fact_writes(cls, stage_result: StageResult) -> list[FactWriteRequest]:
        writes: list[FactWriteRequest] = []
        for item in stage_result.discovered_entities + stage_result.capabilities_gained:
            if not isinstance(item, dict):
                continue
            ref = cls._ref_from_entity(item)
            if ref is None:
                continue
            writes.append(
                FactWriteRequest(
                    kind=FactWriteKind.ENTITY_UPSERT,
                    source_task_id=stage_result.stage_task_id,
                    subject_ref=ref,
                    attributes={key: value for key, value in item.items() if key not in {"id", "entity_id", "ref_id", "type", "ref_type"}},
                    confidence=cls._confidence(item),
                    summary=str(item.get("summary") or f"stage discovered {ref.ref_type or ref.ref_id}"),
                )
            )
        for item in stage_result.discovered_relations:
            if not isinstance(item, dict):
                continue
            subject = cls._ref(item.get("subject_ref") or item.get("source_ref"))
            obj = cls._ref(item.get("object_ref") or item.get("target_ref"))
            relation_type = item.get("relation_type") or item.get("type")
            if subject is None or obj is None or relation_type is None:
                continue
            writes.append(
                FactWriteRequest(
                    kind=FactWriteKind.RELATION_UPSERT,
                    source_task_id=stage_result.stage_task_id,
                    subject_ref=subject,
                    object_ref=obj,
                    relation_type=str(relation_type),
                    attributes={key: value for key, value in item.items() if key not in {"subject_ref", "source_ref", "object_ref", "target_ref", "relation_type", "type"}},
                    confidence=cls._confidence(item),
                    summary=str(item.get("summary") or f"stage discovered relation {relation_type}"),
                )
            )
        return writes

    @classmethod
    @classmethod
    def _runtime_requests(cls, stage_result: StageResult) -> list[RuntimeControlRequest]:
        requests: list[RuntimeControlRequest] = []
        for item in stage_result.sessions:
            if not isinstance(item, dict):
                continue
            requests.append(
                RuntimeControlRequest(
                    request_type=RuntimeControlType.OPEN_SESSION,
                    source_task_id=stage_result.stage_task_id,
                    session_id=cls._string(item.get("session_id")),
                    lease_seconds=int(item.get("lease_seconds") or 300),
                    reuse_policy=item.get("reuse_policy") or "shared",
                    metadata={
                        "bound_identity": item.get("bound_identity") or item.get("identity"),
                        "bound_target": item.get("bound_target") or item.get("target_id") or item.get("host_id"),
                        **dict(item.get("metadata") or {}),
                    },
                )
            )
        for item in stage_result.pivot_routes:
            if not isinstance(item, dict):
                continue
            requests.append(
                RuntimeControlRequest(
                    request_type=RuntimeControlType.REGISTER_PIVOT_ROUTE,
                    source_task_id=stage_result.stage_task_id,
                    session_id=cls._string(item.get("session_id")),
                    metadata=dict(item),
                )
            )
        return requests

    @classmethod
    def _runtime_hint_records(cls, stage_result: StageResult) -> list[dict[str, Any]]:
        hints: list[dict[str, Any]] = []
        if stage_result.runtime_hints:
            hints.append(dict(stage_result.runtime_hints))
        for item in stage_result.credentials:
            if isinstance(item, dict):
                hints.append(
                    {
                        "credential_id": item.get("credential_id") or item.get("id"),
                        "principal": item.get("principal") or item.get("username") or "unknown-principal",
                        "kind": item.get("kind") or item.get("credential_kind") or "password",
                        "secret_ref": item.get("secret_ref"),
                        "status": item.get("status") or "valid",
                        "target_id": item.get("target_id") or item.get("bound_target") or item.get("service_id"),
                    }
                )
        for item in stage_result.sessions:
            if isinstance(item, dict):
                hints.append(
                    {
                        "open_session": True,
                        "session_id": item.get("session_id"),
                        "bound_identity": item.get("bound_identity") or item.get("identity"),
                        "bound_target": item.get("bound_target") or item.get("target_id") or item.get("host_id"),
                        "lease_seconds": item.get("lease_seconds") or 300,
                        "reuse_policy": item.get("reuse_policy") or "shared",
                    }
                )
        for item in stage_result.pivot_routes:
            if isinstance(item, dict):
                hints.append({"register_pivot_route": True, **item})
        return [{key: value for key, value in hint.items() if value is not None} for hint in hints]

    @staticmethod
    @classmethod
    def _refs(cls, value: Any) -> list[GraphRef]:
        if not isinstance(value, list):
            return []
        refs: list[GraphRef] = []
        for item in value:
            ref = cls._ref(item)
            if ref is not None:
                refs.append(ref)
        return refs

    @staticmethod
    def _ref(value: Any) -> GraphRef | None:
        if isinstance(value, GraphRef):
            return value
        if isinstance(value, dict):
            try:
                return GraphRef.model_validate(value)
            except Exception:
                return None
        return None

    @classmethod
    def _ref_from_entity(cls, item: dict[str, Any]) -> GraphRef | None:
        ref = cls._ref(item.get("ref"))
        if ref is not None:
            return ref
        ref_id = item.get("ref_id") or item.get("entity_id") or item.get("id") or item.get("capability_id")
        if ref_id is None:
            return None
        return GraphRef(
            graph="kg",
            ref_id=str(ref_id),
            ref_type=cls._string(item.get("ref_type") or item.get("type") or item.get("capability_type")),
            label=cls._string(item.get("label") or item.get("summary")),
        )

    @staticmethod
    def _confidence(item: dict[str, Any]) -> float:
        try:
            value = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            value = 0.5
        return max(0.0, min(1.0, value))

    @staticmethod
    def _string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = ["StageResultAdapter"]
