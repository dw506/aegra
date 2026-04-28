"""Perception agent that converts worker execution outputs into KG-ready records.

The perception layer is intentionally non-authoritative. It interprets worker
outcomes and raw execution payloads into structured observations and evidence
without performing graph writes or planning decisions.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_models import EvidenceRecord, ObservationRecord, OutcomeRecord
from src.core.agents.agent_protocol import (
    AgentInput,
    AgentKind,
    AgentOutput,
    BaseAgent,
    GraphRef,
    GraphScope,
    WritePermission,
)


class PerceptionNormalizationAdvice(BaseModel):
    """可选的 LLM 归一化结果。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    normalized_result: dict[str, Any] = Field(default_factory=dict)
    summary_suffix: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PerceptionLLMNormalizer(Protocol):
    """Perception 可选 LLM 接口。

    中文注释：
    LLM 在这里仅用于原始工具输出归一化，不能决定工具参数，也不能直接触发执行动作。
    """

    def normalize(
        self,
        *,
        outcome: OutcomeRecord,
        raw_result: dict[str, Any],
        refs: list[GraphRef],
    ) -> PerceptionNormalizationAdvice | None:
        """Return optional normalized worker output."""


class PerceptionAgent(BaseAgent):
    """Translate worker outcomes into structured observations and evidence."""

    def __init__(self, name: str = "perception_agent", llm_normalizer: PerceptionLLMNormalizer | None = None) -> None:
        self._llm_normalizer = llm_normalizer
        super().__init__(
            name=name,
            kind=AgentKind.PERCEPTION,
            write_permission=WritePermission(
                scopes=[],
                allow_structural_write=False,
                allow_state_write=False,
                allow_event_emit=True,
            ),
        )

    def validate_input(self, agent_input: AgentInput) -> None:
        """Validate that the payload contains one worker outcome to interpret."""

        super().validate_input(agent_input)
        outcome_payload = agent_input.raw_payload.get("outcome")
        if outcome_payload is None:
            raise ValueError("perception input requires raw_payload.outcome")

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Convert one worker outcome plus raw result into observation/evidence records."""

        outcome = self._parse_outcome(agent_input.raw_payload["outcome"])
        raw_result = self._coerce_mapping(agent_input.raw_payload.get("raw_result"))
        refs = self._collect_refs(agent_input, outcome, raw_result)
        raw_result, normalization_log = self._normalize_with_llm(outcome=outcome, raw_result=raw_result, refs=refs)

        observations = self._extract_observations(
            outcome=outcome,
            raw_result=raw_result,
            refs=refs,
        )
        evidence = self._extract_evidence(
            outcome=outcome,
            raw_result=raw_result,
            refs=refs,
            observations=observations,
        )

        logs = [
            f"processed outcome {outcome.id} for task {outcome.task_id}",
            f"interpreted outcome_type={outcome.outcome_type} success={outcome.success}",
            f"emitted {len(observations)} observation(s) and {len(evidence)} evidence record(s)",
        ]
        if not raw_result:
            logs.append("raw_result missing or empty; perception used outcome payload only")
        if normalization_log is not None:
            logs.append(normalization_log)

        return AgentOutput(
            observations=[record.to_agent_output_fragment() for record in observations],
            evidence=[record.to_agent_output_fragment() for record in evidence],
            logs=logs,
        )

    # 中文注释：
    # Perception 的 LLM 只能把杂乱原始输出整理成稳定字段，不碰执行边界。
    def _normalize_with_llm(
        self,
        *,
        outcome: OutcomeRecord,
        raw_result: dict[str, Any],
        refs: list[GraphRef],
    ) -> tuple[dict[str, Any], str | None]:
        if self._llm_normalizer is None:
            return raw_result, None
        advice = self._llm_normalizer.normalize(outcome=outcome, raw_result=raw_result, refs=refs)
        if advice is None:
            return raw_result, None
        normalized = dict(raw_result)
        normalized.update(dict(advice.normalized_result))
        if advice.metadata:
            normalized["llm_normalization"] = dict(advice.metadata)
        log = "perception llm normalizer refined raw worker output"
        if advice.summary_suffix:
            normalized["summary"] = self._coalesce_string(
                normalized.get("summary"),
                outcome.summary,
            )
            normalized["summary"] = f"{normalized['summary']}; {advice.summary_suffix}"
        return normalized, log

    def _extract_observations(
        self,
        *,
        outcome: OutcomeRecord,
        raw_result: dict[str, Any],
        refs: list[GraphRef],
    ) -> list[ObservationRecord]:
        """Extract KG-ready observation records from one outcome/result pair."""

        confidence = self._infer_confidence(outcome=outcome, raw_result=raw_result)
        outcome_type = outcome.outcome_type.strip().lower()
        branch = outcome_type or "generic"
        summary = self._coalesce_string(
            raw_result.get("observation_summary"),
            raw_result.get("summary"),
            outcome.summary,
            f"Observed worker outcome for task {outcome.task_id}",
        )

        payload: dict[str, Any] = {
            "task_id": outcome.task_id,
            "outcome_id": outcome.id,
            "outcome_type": outcome.outcome_type,
            "success": outcome.success,
            "branch": branch,
            "kg_ready": True,
        }

        observed_entities = self._coerce_list(
            raw_result.get("observed_entities")
            or raw_result.get("artifacts")
            or outcome.payload.get("observed_entities")
        )
        key_findings = self._coerce_list(
            raw_result.get("findings")
            or raw_result.get("observations")
            or outcome.payload.get("findings")
        )

        if branch == "execution_result":
            payload["execution_state"] = self._coalesce_string(
                raw_result.get("status"),
                "succeeded" if outcome.success else "failed",
            )
        elif branch == "validation_result":
            payload["validation_status"] = self._coalesce_string(
                raw_result.get("validation_status"),
                "validated" if outcome.success else "rejected",
            )
        elif branch == "collection_result":
            payload["collection_scope"] = self._coalesce_string(
                raw_result.get("collection_scope"),
                outcome.payload.get("collection_scope"),
                "unspecified",
            )
        else:
            payload["interpreted_as"] = branch

        if observed_entities:
            payload["observed_entities"] = observed_entities
        if key_findings:
            payload["findings"] = key_findings
        if outcome.raw_result_ref:
            payload["raw_result_ref"] = outcome.raw_result_ref
        if outcome.payload:
            payload["outcome_payload"] = dict(outcome.payload)
        if raw_result:
            payload["raw_fields"] = {
                key: value
                for key, value in raw_result.items()
                if key not in {"summary", "observation_summary", "evidence_summary"}
            }

        return [
            ObservationRecord(
                source_agent=self.name,
                summary=summary,
                confidence=confidence,
                refs=refs,
                payload=payload,
            )
        ]

    def _extract_evidence(
        self,
        *,
        outcome: OutcomeRecord,
        raw_result: dict[str, Any],
        refs: list[GraphRef],
        observations: list[ObservationRecord],
    ) -> list[EvidenceRecord]:
        """Extract evidence references and summaries from one outcome/result pair."""

        confidence = self._infer_confidence(outcome=outcome, raw_result=raw_result)
        evidence_summary = self._coalesce_string(
            raw_result.get("evidence_summary"),
            raw_result.get("summary"),
            f"Evidence derived from {outcome.outcome_type} for task {outcome.task_id}",
        )

        payload = {
            "task_id": outcome.task_id,
            "outcome_id": outcome.id,
            "outcome_type": outcome.outcome_type,
            "success": outcome.success,
            "observation_ids": [record.id for record in observations],
            "evidence_kind": self._coalesce_string(
                raw_result.get("evidence_kind"),
                f"{outcome.outcome_type}_evidence",
            ),
        }

        artifacts = self._coerce_list(
            raw_result.get("artifacts")
            or raw_result.get("evidence_items")
            or outcome.payload.get("artifacts")
        )
        if artifacts:
            payload["artifacts"] = artifacts
        if raw_result:
            payload["raw_result_excerpt"] = {
                key: value
                for key, value in raw_result.items()
                if key
                not in {
                    "summary",
                    "observation_summary",
                    "evidence_summary",
                }
            }

        payload_ref = self._coalesce_optional_string(
            raw_result.get("payload_ref"),
            raw_result.get("raw_result_ref"),
            outcome.raw_result_ref,
        )

        return [
            EvidenceRecord(
                source_agent=self.name,
                summary=evidence_summary,
                confidence=confidence,
                refs=refs,
                payload=payload,
                payload_ref=payload_ref,
            )
        ]

    def _infer_confidence(
        self,
        *,
        outcome: OutcomeRecord,
        raw_result: dict[str, Any],
    ) -> float:
        """Infer a stable confidence score from outcome/result content."""

        candidates = [
            raw_result.get("confidence"),
            raw_result.get("score"),
            outcome.payload.get("confidence"),
            outcome.confidence,
        ]
        for candidate in candidates:
            try:
                if candidate is None:
                    continue
                value = float(candidate)
                return max(0.0, min(1.0, value))
            except (TypeError, ValueError):
                continue

        if outcome.success and raw_result:
            return 0.7
        if outcome.success:
            return 0.6
        if raw_result:
            return 0.4
        return 0.3

    def _parse_outcome(self, value: Any) -> OutcomeRecord:
        """Normalize the provided outcome payload into an `OutcomeRecord`."""

        if isinstance(value, OutcomeRecord):
            return value
        if not isinstance(value, dict):
            raise TypeError("raw_payload.outcome must be an OutcomeRecord or mapping")
        return OutcomeRecord.model_validate(value)

    def _collect_refs(
        self,
        agent_input: AgentInput,
        outcome: OutcomeRecord,
        raw_result: dict[str, Any],
    ) -> list[GraphRef]:
        """Collect task and graph references from all available input sources."""

        refs: list[GraphRef] = []
        seen: set[tuple[str, str, str | None]] = set()

        def add(ref: GraphRef) -> None:
            key = (ref.graph.value, ref.ref_id, ref.ref_type)
            if key in seen:
                return
            seen.add(key)
            refs.append(ref)

        for ref in agent_input.graph_refs:
            add(ref)
        for ref in outcome.refs:
            add(ref)

        for raw_ref in self._coerce_list(raw_result.get("refs")):
            if isinstance(raw_ref, GraphRef):
                add(raw_ref)
            elif isinstance(raw_ref, dict):
                try:
                    add(GraphRef.model_validate(raw_ref))
                except Exception:
                    continue

        task_ref = agent_input.task_ref or outcome.task_id
        if task_ref and not any(ref.graph == GraphScope.TG and ref.ref_id == task_ref for ref in refs):
            add(GraphRef(graph=GraphScope.TG, ref_id=task_ref, ref_type="task"))

        return refs

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any]:
        """Return a shallow mapping copy or an empty mapping."""

        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def _coerce_list(value: Any) -> list[Any]:
        """Return a list representation that is stable for absent fields."""

        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _coalesce_string(*values: Any) -> str:
        """Return the first non-empty string-like value."""

        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return "unspecified"

    @staticmethod
    def _coalesce_optional_string(*values: Any) -> str | None:
        """Return the first non-empty string-like value or None."""

        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None


__all__ = [
    "PerceptionAgent",
    "PerceptionLLMNormalizer",
    "PerceptionNormalizationAdvice",
]
