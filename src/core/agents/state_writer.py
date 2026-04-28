"""State writer agent that normalizes perception records into KG deltas.

The State Writer is the only formal KG writer in the agent layer. This module
does not mutate a concrete KG store directly. Instead, it converts structured
observation and evidence records into serializable KG patch/delta payloads that
an external store owner can apply later.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_models import (
    EvidenceRecord,
    ObservationRecord,
    StateDeltaRecord,
    new_record_id,
)
from src.core.agents.agent_protocol import (
    AgentInput,
    AgentKind,
    AgentOutput,
    BaseAgent,
    GraphRef,
    GraphScope,
    WritePermission,
)
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.kg_enums import EdgeType, NodeType


class KGEntityPatch(BaseModel):
    """Serializable node patch for later KG application."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    patch_id: str = Field(default_factory=lambda: new_record_id("kg-entity-patch"))
    entity_id: str = Field(min_length=1)
    operation: str = Field(default="upsert", min_length=1)
    entity_kind: str = Field(default="node", min_length=1)
    entity_type: str = Field(min_length=1)
    label: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class KGRelationPatch(BaseModel):
    """Serializable edge patch for later KG application."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    patch_id: str = Field(default_factory=lambda: new_record_id("kg-relation-patch"))
    relation_id: str = Field(min_length=1)
    operation: str = Field(default="upsert", min_length=1)
    entity_kind: str = Field(default="edge", min_length=1)
    relation_type: str = Field(min_length=1)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    label: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class KGDeltaEvent(BaseModel):
    """Event emitted after one State Writer normalization pass."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    event_id: str = Field(default_factory=lambda: new_record_id("kg-event"))
    event_type: str = Field(default="kg.delta", min_length=1)
    producer: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    kg_ref: dict[str, Any] = Field(default_factory=dict)
    delta_ids: list[str] = Field(default_factory=list)
    entity_patch_ids: list[str] = Field(default_factory=list)
    relation_patch_ids: list[str] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    task_ref: str | None = None
    decision_ref: str | None = None
    runtime_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KGPatchApplyRequest(BaseModel):
    """Serializable handoff object for an external KG store."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    patch_batch_id: str = Field(default_factory=lambda: new_record_id("kg-patch-batch"))
    kg_ref: dict[str, Any]
    operation_id: str = Field(min_length=1)
    task_ref: str | None = None
    decision_ref: str | None = None
    runtime_ref: str | None = None
    base_kg_version: int | None = Field(default=None, ge=0)
    resulting_kg_version: int | None = Field(default=None, ge=0)
    state_deltas: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StateWriterAgent(BaseAgent):
    """Normalize perception-layer records into KG-only patch deltas."""

    def __init__(self, name: str = "state_writer_agent") -> None:
        super().__init__(
            name=name,
            kind=AgentKind.STATE_WRITER,
            write_permission=WritePermission(
                scopes=[GraphScope.KG],
                allow_structural_write=True,
                allow_state_write=True,
                allow_event_emit=True,
            ),
        )

    def validate_input(self, agent_input: AgentInput) -> None:
        """Ensure the invocation contains KG context and writable records."""

        super().validate_input(agent_input)
        self._resolve_kg_ref(agent_input)

        observations = self._parse_records(
            agent_input.raw_payload.get("observation") or agent_input.raw_payload.get("observations"),
            ObservationRecord,
        )
        evidence = self._parse_records(
            agent_input.raw_payload.get("evidence") or agent_input.raw_payload.get("evidences"),
            EvidenceRecord,
        )
        if not observations and not evidence:
            raise ValueError(
                "state writer input requires raw_payload.observation(s) or raw_payload.evidence"
            )

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Convert observations and evidence into KG patch deltas and events."""

        kg_ref = self._resolve_kg_ref(agent_input)
        observations = self._parse_records(
            agent_input.raw_payload.get("observation") or agent_input.raw_payload.get("observations"),
            ObservationRecord,
        )
        evidence = self._parse_records(
            agent_input.raw_payload.get("evidence") or agent_input.raw_payload.get("evidences"),
            EvidenceRecord,
        )
        context_refs = self._build_context_refs(agent_input=agent_input, kg_ref=kg_ref)

        state_deltas: list[dict[str, Any]] = []
        entity_patch_ids: list[str] = []
        relation_patch_ids: list[str] = []

        for observation in observations:
            entity_patch = self._normalize_observation_to_entity_patch(
                observation=observation,
                agent_input=agent_input,
                context_refs=context_refs,
            )
            state_deltas.append(self._build_entity_state_delta(entity_patch))
            entity_patch_ids.append(entity_patch.patch_id)

            for relation_patch in self._build_observation_relation_patches(
                observation=observation,
                agent_input=agent_input,
                context_refs=context_refs,
            ):
                state_deltas.append(self._build_relation_state_delta(relation_patch))
                relation_patch_ids.append(relation_patch.patch_id)

            structured_entities, structured_relations = self._build_structured_patches(
                payload=observation.payload,
                agent_input=agent_input,
                context_refs=context_refs,
                record_id=observation.id,
                source_agent=observation.source_agent,
            )
            for patch in structured_entities:
                state_deltas.append(self._build_entity_state_delta(patch))
                entity_patch_ids.append(patch.patch_id)
            for patch in structured_relations:
                state_deltas.append(self._build_relation_state_delta(patch))
                relation_patch_ids.append(patch.patch_id)

        for record in evidence:
            entity_patch = self._normalize_evidence_to_entity_patch(
                evidence=record,
                agent_input=agent_input,
                context_refs=context_refs,
            )
            state_deltas.append(self._build_entity_state_delta(entity_patch))
            entity_patch_ids.append(entity_patch.patch_id)

            for relation_patch in self._normalize_evidence_to_relation_patch(
                evidence=record,
                agent_input=agent_input,
                context_refs=context_refs,
            ):
                state_deltas.append(self._build_relation_state_delta(relation_patch))
                relation_patch_ids.append(relation_patch.patch_id)

            structured_entities, structured_relations = self._build_structured_patches(
                payload=record.payload,
                agent_input=agent_input,
                context_refs=context_refs,
                record_id=record.id,
                source_agent=record.source_agent,
            )
            for patch in structured_entities:
                state_deltas.append(self._build_entity_state_delta(patch))
                entity_patch_ids.append(patch.patch_id)
            for patch in structured_relations:
                state_deltas.append(self._build_relation_state_delta(patch))
                relation_patch_ids.append(patch.patch_id)

        emitted_events = self._build_kg_delta_events(
            state_deltas=state_deltas,
            kg_ref=kg_ref,
            observations=observations,
            evidence=evidence,
            entity_patch_ids=entity_patch_ids,
            relation_patch_ids=relation_patch_ids,
            agent_input=agent_input,
        )

        logs = [
            f"resolved kg_ref={kg_ref.ref_id}",
            f"normalized {len(observations)} observation(s) into KG patch deltas",
            f"normalized {len(evidence)} evidence record(s) into KG patch deltas",
            f"prepared {len(state_deltas)} KG state delta(s) and {len(emitted_events)} event(s)",
            "no KG store mutation was executed; apply patches through an external store owner",
        ]
        if agent_input.task_ref:
            logs.append(f"associated task_ref={agent_input.task_ref}")
        if agent_input.decision_ref:
            logs.append(f"associated decision_ref={agent_input.decision_ref}")
        if agent_input.context.runtime_state_ref:
            logs.append(f"associated runtime_ref={agent_input.context.runtime_state_ref}")

        return AgentOutput(
            state_deltas=state_deltas,
            emitted_events=[event.model_dump(mode="json") for event in emitted_events],
            logs=logs,
        )

    def build_store_apply_request(
        self,
        *,
        kg_ref: GraphRef,
        state_deltas: Sequence[dict[str, Any]],
        agent_input: AgentInput,
        base_kg_version: int | None = None,
    ) -> KGPatchApplyRequest:
        """Build the handoff payload that an external KG store may consume."""

        patch_batch_id = new_record_id("kg-patch-batch")
        return KGPatchApplyRequest(
            patch_batch_id=patch_batch_id,
            kg_ref=kg_ref.model_dump(mode="json"),
            operation_id=agent_input.context.operation_id,
            task_ref=agent_input.task_ref,
            decision_ref=agent_input.decision_ref,
            runtime_ref=agent_input.context.runtime_state_ref,
            base_kg_version=base_kg_version,
            state_deltas=[dict(delta) for delta in state_deltas],
            metadata={
                "producer": self.name,
                "scope": GraphScope.KG.value,
                "patch_batch_id": patch_batch_id,
            },
        )

    def apply_to_store(
        self,
        *,
        store: KnowledgeGraph,
        apply_request: KGPatchApplyRequest,
    ) -> dict[str, Any]:
        """通过 StateWriter 的正式入口把 patch batch 落到 KG store。

        这样可以保持“StateWriter 是 KG owner”的边界：
        - 外部编排器调用 StateWriter 生成 patch batch
        - 仍由 StateWriter 发起 store.apply_patch_batch(...)
        - KG store 只执行结构变更和版本推进
        """

        result = store.apply_patch_batch(apply_request.model_dump(mode="json"))
        apply_request.resulting_kg_version = int(result["kg_version"])
        return result

    def _normalize_observation_to_entity_patch(
        self,
        *,
        observation: ObservationRecord,
        agent_input: AgentInput,
        context_refs: list[GraphRef],
    ) -> KGEntityPatch:
        """Normalize one observation record into an observation node patch."""

        attributes = {
            "summary": observation.summary,
            "confidence": observation.confidence,
            "observation_kind": self._coalesce_string(
                observation.payload.get("branch"),
                observation.payload.get("observation_kind"),
                "agent_observation",
            ),
            "properties": dict(observation.payload),
        }
        return KGEntityPatch(
            entity_id=observation.id,
            entity_type=NodeType.OBSERVATION.value,
            label=observation.summary,
            attributes=attributes,
            source_refs=self._graph_refs_to_payload([*context_refs, *observation.refs]),
            provenance=self._build_provenance(
                record_id=observation.id,
                source_agent=observation.source_agent,
                agent_input=agent_input,
            ),
        )

    def _normalize_evidence_to_entity_patch(
        self,
        *,
        evidence: EvidenceRecord,
        agent_input: AgentInput,
        context_refs: list[GraphRef],
    ) -> KGEntityPatch:
        """Normalize one evidence record into an evidence node patch."""

        attributes = {
            "summary": evidence.summary,
            "confidence": evidence.confidence,
            "evidence_kind": self._coalesce_string(
                evidence.payload.get("evidence_kind"),
                evidence.payload.get("kind"),
                "agent_evidence",
            ),
            "content_ref": evidence.payload_ref,
            "properties": dict(evidence.payload),
        }
        return KGEntityPatch(
            entity_id=evidence.id,
            entity_type=NodeType.EVIDENCE.value,
            label=evidence.summary,
            attributes=attributes,
            source_refs=self._graph_refs_to_payload([*context_refs, *evidence.refs]),
            provenance=self._build_provenance(
                record_id=evidence.id,
                source_agent=evidence.source_agent,
                agent_input=agent_input,
            ),
        )

    def _build_observation_relation_patches(
        self,
        *,
        observation: ObservationRecord,
        agent_input: AgentInput,
        context_refs: list[GraphRef],
    ) -> list[KGRelationPatch]:
        """Create observation-to-KG relation patches for referenced entities."""

        patches: list[KGRelationPatch] = []
        for target_ref in self._iter_kg_entity_refs([*agent_input.graph_refs, *observation.refs]):
            if target_ref.ref_id == observation.id:
                continue
            patches.append(
                KGRelationPatch(
                    relation_id=f"{EdgeType.OBSERVED_ON.value.lower()}::{observation.id}::{target_ref.ref_id}",
                    relation_type=EdgeType.OBSERVED_ON.value,
                    source=observation.id,
                    target=target_ref.ref_id,
                    label=EdgeType.OBSERVED_ON.value.lower(),
                    attributes={"confidence": observation.confidence},
                    source_refs=self._graph_refs_to_payload([*context_refs, *observation.refs]),
                    provenance=self._build_provenance(
                        record_id=observation.id,
                        source_agent=observation.source_agent,
                        agent_input=agent_input,
                    ),
                )
            )
        return patches

    def _normalize_evidence_to_relation_patch(
        self,
        *,
        evidence: EvidenceRecord,
        agent_input: AgentInput,
        context_refs: list[GraphRef],
    ) -> list[KGRelationPatch]:
        """Normalize one evidence record into support and derivation relations."""

        patches: list[KGRelationPatch] = []
        shared_refs = self._graph_refs_to_payload([*context_refs, *evidence.refs])
        provenance = self._build_provenance(
            record_id=evidence.id,
            source_agent=evidence.source_agent,
            agent_input=agent_input,
        )

        for target_ref in self._iter_kg_entity_refs([*agent_input.graph_refs, *evidence.refs]):
            if target_ref.ref_id == evidence.id:
                continue
            patches.append(
                KGRelationPatch(
                    relation_id=f"{EdgeType.SUPPORTED_BY.value.lower()}::{target_ref.ref_id}::{evidence.id}",
                    relation_type=EdgeType.SUPPORTED_BY.value,
                    source=target_ref.ref_id,
                    target=evidence.id,
                    label=EdgeType.SUPPORTED_BY.value.lower(),
                    attributes={"confidence": evidence.confidence},
                    source_refs=shared_refs,
                    provenance=provenance,
                )
            )

        for observation_id in self._extract_observation_ids(evidence):
            if observation_id == evidence.id:
                continue
            patches.append(
                KGRelationPatch(
                    relation_id=f"{EdgeType.DERIVED_FROM.value.lower()}::{evidence.id}::{observation_id}",
                    relation_type=EdgeType.DERIVED_FROM.value,
                    source=evidence.id,
                    target=observation_id,
                    label=EdgeType.DERIVED_FROM.value.lower(),
                    attributes={"confidence": evidence.confidence},
                    source_refs=shared_refs,
                    provenance=provenance,
                )
            )

        return self._dedupe_relation_patches(patches)

    def _build_kg_delta_events(
        self,
        *,
        state_deltas: Sequence[dict[str, Any]],
        kg_ref: GraphRef,
        observations: Sequence[ObservationRecord],
        evidence: Sequence[EvidenceRecord],
        entity_patch_ids: Sequence[str],
        relation_patch_ids: Sequence[str],
        agent_input: AgentInput,
    ) -> list[KGDeltaEvent]:
        """Build event payloads that describe the proposed KG delta set."""

        if not state_deltas:
            return []
        store_apply_request = self.build_store_apply_request(
            kg_ref=kg_ref,
            state_deltas=state_deltas,
            agent_input=agent_input,
            base_kg_version=self._extract_kg_version(agent_input),
        )
        return [
            KGDeltaEvent(
                producer=self.name,
                operation_id=agent_input.context.operation_id,
                kg_ref=kg_ref.model_dump(mode="json"),
                delta_ids=[str(delta.get("id")) for delta in state_deltas if delta.get("id")],
                entity_patch_ids=list(entity_patch_ids),
                relation_patch_ids=list(relation_patch_ids),
                observation_ids=[record.id for record in observations],
                evidence_ids=[record.id for record in evidence],
                task_ref=agent_input.task_ref,
                decision_ref=agent_input.decision_ref,
                runtime_ref=agent_input.context.runtime_state_ref,
                metadata={
                    "delta_count": len(state_deltas),
                    "scope": GraphScope.KG.value,
                    "base_kg_version": store_apply_request.base_kg_version,
                    "patch_batch_id": store_apply_request.patch_batch_id,
                    "store_apply_request": store_apply_request.model_dump(mode="json"),
                },
            )
        ]

    def _build_structured_patches(
        self,
        *,
        payload: dict[str, Any],
        agent_input: AgentInput,
        context_refs: list[GraphRef],
        record_id: str,
        source_agent: str,
    ) -> tuple[list[KGEntityPatch], list[KGRelationPatch]]:
        """Extract stable KG node/edge patches from structured tool payloads."""

        entities_raw = payload.get("entities") or payload.get("properties", {}).get("entities") or []
        relations_raw = payload.get("relations") or payload.get("properties", {}).get("relations") or []
        provenance = self._build_provenance(
            record_id=record_id,
            source_agent=source_agent,
            agent_input=agent_input,
        )
        refs = self._graph_refs_to_payload(context_refs)

        entity_patches: list[KGEntityPatch] = []
        for raw in entities_raw:
            if not isinstance(raw, dict):
                continue
            entity_id = self._coalesce_optional_string(raw.get("id"), raw.get("entity_id"))
            entity_type = self._normalize_enum_value(raw.get("type"), NodeType)
            if entity_id is None or entity_type is None:
                continue
            label = self._coalesce_string(raw.get("label"), raw.get("name"), entity_id)
            attributes = {
                key: value
                for key, value in raw.items()
                if key not in {"id", "entity_id", "type", "label", "name"}
            }
            entity_patches.append(
                KGEntityPatch(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    label=label,
                    attributes=attributes,
                    source_refs=refs,
                    provenance=provenance,
                )
            )

        relation_patches: list[KGRelationPatch] = []
        for raw in relations_raw:
            if not isinstance(raw, dict):
                continue
            source = self._coalesce_optional_string(raw.get("source"), raw.get("subject"), raw.get("subject_id"))
            target = self._coalesce_optional_string(raw.get("target"), raw.get("object"), raw.get("object_id"))
            relation_type = self._normalize_enum_value(raw.get("type"), EdgeType)
            if source is None or target is None or relation_type is None:
                continue
            relation_id = self._coalesce_string(
                raw.get("id"),
                f"{relation_type.lower()}::{source}::{target}",
            )
            label = self._coalesce_string(raw.get("label"), relation_type.lower())
            attributes = {
                key: value
                for key, value in raw.items()
                if key
                not in {
                    "id",
                    "type",
                    "label",
                    "source",
                    "target",
                    "subject",
                    "subject_id",
                    "object",
                    "object_id",
                }
            }
            relation_patches.append(
                KGRelationPatch(
                    relation_id=relation_id,
                    relation_type=relation_type,
                    source=source,
                    target=target,
                    label=label,
                    attributes=attributes,
                    source_refs=refs,
                    provenance=provenance,
                )
            )

        return self._dedupe_entity_patches(entity_patches), self._dedupe_relation_patches(relation_patches)

    def _build_entity_state_delta(self, patch: KGEntityPatch) -> dict[str, Any]:
        """Wrap one entity patch as a formal KG state delta fragment."""

        return StateDeltaRecord(
            source_agent=self.name,
            summary=f"Upsert KG entity {patch.entity_id}",
            graph_scope=GraphScope.KG,
            delta_type="upsert_entity",
            target_ref=GraphRef(
                graph=GraphScope.KG,
                ref_id=patch.entity_id,
                ref_type=patch.entity_type,
            ),
            patch=patch.model_dump(mode="json"),
            payload={"patch_kind": "entity"},
        ).to_agent_output_fragment() | {"write_type": "structural"}

    def _build_relation_state_delta(self, patch: KGRelationPatch) -> dict[str, Any]:
        """Wrap one relation patch as a formal KG state delta fragment."""

        return StateDeltaRecord(
            source_agent=self.name,
            summary=f"Upsert KG relation {patch.relation_id}",
            graph_scope=GraphScope.KG,
            delta_type="upsert_relation",
            target_ref=GraphRef(
                graph=GraphScope.KG,
                ref_id=patch.relation_id,
                ref_type=patch.relation_type,
            ),
            patch=patch.model_dump(mode="json"),
            payload={"patch_kind": "relation"},
        ).to_agent_output_fragment() | {"write_type": "structural"}

    def _resolve_kg_ref(self, agent_input: AgentInput) -> GraphRef:
        """Resolve the current KG reference from graph refs or raw payload."""

        for ref in agent_input.graph_refs:
            if ref.graph == GraphScope.KG:
                return ref

        raw_ref = agent_input.raw_payload.get("kg_ref")
        if isinstance(raw_ref, GraphRef):
            if raw_ref.graph != GraphScope.KG:
                raise ValueError("raw_payload.kg_ref must reference the KG scope")
            return raw_ref
        if isinstance(raw_ref, dict):
            candidate = GraphRef.model_validate(raw_ref)
            if candidate.graph != GraphScope.KG:
                raise ValueError("raw_payload.kg_ref must reference the KG scope")
            return candidate
        raise ValueError("state writer input requires a current KG GraphRef")

    @staticmethod
    def _extract_kg_version(agent_input: AgentInput) -> int | None:
        """从输入上下文中提取当前 KG 版本，作为 patch batch 的基线版本。"""

        raw_version = agent_input.raw_payload.get("kg_version")
        if raw_version is None:
            return None
        try:
            return max(int(raw_version), 0)
        except (TypeError, ValueError):
            return None

    def _build_context_refs(self, *, agent_input: AgentInput, kg_ref: GraphRef) -> list[GraphRef]:
        """Collect invocation-level refs for provenance and patch metadata."""

        refs = [kg_ref]
        if agent_input.task_ref:
            refs.append(GraphRef(graph=GraphScope.TG, ref_id=agent_input.task_ref, ref_type="task"))
        if agent_input.context.runtime_state_ref:
            refs.append(
                GraphRef(
                    graph=GraphScope.RUNTIME,
                    ref_id=agent_input.context.runtime_state_ref,
                    ref_type="runtime_state",
                )
            )
        return refs

    def _build_provenance(
        self,
        *,
        record_id: str,
        source_agent: str,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Build stable provenance metadata for one generated patch."""

        return {
            "record_id": record_id,
            "source_agent": source_agent,
            "writer_agent": self.name,
            "operation_id": agent_input.context.operation_id,
            "task_ref": agent_input.task_ref,
            "decision_ref": agent_input.decision_ref,
            "runtime_ref": agent_input.context.runtime_state_ref,
        }

    def _extract_observation_ids(self, evidence: EvidenceRecord) -> list[str]:
        """Extract referenced observation IDs from one evidence payload."""

        raw_ids = evidence.payload.get("observation_ids")
        if isinstance(raw_ids, list):
            return [str(item) for item in raw_ids if str(item).strip()]
        if raw_ids is None:
            return []
        text = str(raw_ids).strip()
        return [text] if text else []

    def _iter_kg_entity_refs(self, refs: Iterable[GraphRef]) -> list[GraphRef]:
        """Return KG refs that point at actual KG entities instead of the graph root."""

        result: list[GraphRef] = []
        seen: set[tuple[str, str, str | None]] = set()
        for ref in refs:
            if ref.graph != GraphScope.KG:
                continue
            if ref.ref_type == "graph":
                continue
            key = (ref.graph.value, ref.ref_id, ref.ref_type)
            if key in seen:
                continue
            seen.add(key)
            result.append(ref)
        return result

    def _dedupe_relation_patches(self, patches: Sequence[KGRelationPatch]) -> list[KGRelationPatch]:
        """Drop duplicate relation patches while keeping stable order."""

        result: list[KGRelationPatch] = []
        seen: set[str] = set()
        for patch in patches:
            if patch.relation_id in seen:
                continue
            seen.add(patch.relation_id)
            result.append(patch)
        return result

    def _dedupe_entity_patches(self, patches: Sequence[KGEntityPatch]) -> list[KGEntityPatch]:
        """Drop duplicate entity patches while keeping stable order."""

        result: list[KGEntityPatch] = []
        seen: set[str] = set()
        for patch in patches:
            if patch.entity_id in seen:
                continue
            seen.add(patch.entity_id)
            result.append(patch)
        return result

    def _parse_records(
        self,
        value: Any,
        model_type: type[ObservationRecord] | type[EvidenceRecord],
    ) -> list[ObservationRecord] | list[EvidenceRecord]:
        """Normalize one record or list of records into typed models."""

        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [
            item
            if isinstance(item, model_type)
            else model_type.model_validate(item)
            for item in items
        ]

    @staticmethod
    def _graph_refs_to_payload(refs: Iterable[GraphRef]) -> list[dict[str, Any]]:
        """Serialize refs while dropping duplicates."""

        payload: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str | None]] = set()
        for ref in refs:
            key = (ref.graph.value, ref.ref_id, ref.ref_type)
            if key in seen:
                continue
            seen.add(key)
            payload.append(ref.model_dump(mode="json"))
        return payload

    @staticmethod
    def _coalesce_string(*values: Any) -> str:
        """Return the first non-empty string representation."""

        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return "unspecified"

    @staticmethod
    def _coalesce_optional_string(*values: Any) -> str | None:
        """Return the first non-empty string representation or None."""

        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    @staticmethod
    def _normalize_enum_value(value: Any, enum_type: type[NodeType] | type[EdgeType]) -> str | None:
        """Normalize one enum candidate against the KG enum value set."""

        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        for member in enum_type:
            if text == member.value or text.upper() == member.value.upper() or text.upper() == member.name:
                return member.value
        return None


__all__ = [
    "KGDeltaEvent",
    "KGEntityPatch",
    "KGRelationPatch",
    "KGPatchApplyRequest",
    "StateWriterAgent",
]
