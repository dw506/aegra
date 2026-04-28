"""Graph Projection agent that maps KG delta events into AG deltas.

This agent is the AG-side owner in the agent layer. It consumes KG delta
events, projects only the affected slices into Attack Graph state/action
patches, and emits AG-scoped deltas plus downstream projection events. It does
not write TG and does not rebuild AG wholesale by default.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_models import StateDeltaRecord, new_record_id
from src.core.agents.agent_protocol import (
    AgentInput,
    AgentKind,
    AgentOutput,
    BaseAgent,
    GraphRef,
    GraphScope,
    WritePermission,
)
from src.core.agents.kg_events import KGDeltaEvent, KGDeltaEventType, KGEventBatch
from src.core.graph.ag_projector import AttackGraphProjector
from src.core.models.ag import (
    ActivationStatus,
    ActionNodeType,
    GraphRef as AGGraphRef,
    ProjectionTrace,
    StateNodeType,
    TruthStatus,
    stable_node_id,
)


class AGProjectionEvent(BaseModel):
    """Structured AG projection event emitted after delta generation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    event_id: str = Field(default_factory=lambda: new_record_id("ag-event"))
    event_type: str = Field(default="ag.delta", min_length=1)
    source_agent: str = Field(min_length=1)
    affected_ag_refs: list[dict[str, Any]] = Field(default_factory=list)
    source_kg_event_ids: list[str] = Field(default_factory=list)
    source_kg_version: int | None = Field(default=None, ge=0)
    ag_version: int | None = Field(default=None, ge=0)
    projection_batch_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphProjectionAgent(BaseAgent):
    """Project incremental KG changes into AG states and candidate actions."""

    def __init__(self, name: str = "graph_projection_agent") -> None:
        super().__init__(
            name=name,
            kind=AgentKind.GRAPH_PROJECTION,
            write_permission=WritePermission(
                scopes=[GraphScope.AG],
                allow_structural_write=True,
                allow_state_write=True,
                allow_event_emit=True,
            ),
        )

    def validate_input(self, agent_input: AgentInput) -> None:
        """Require KG refs and at least one KG delta event or event batch."""

        super().validate_input(agent_input)
        if not any(ref.graph == GraphScope.KG for ref in agent_input.graph_refs):
            raise ValueError("graph projection input requires at least one KG ref")
        if not self._parse_event_batch(agent_input):
            raise ValueError("graph projection input requires raw_payload.kg_event(s) or raw_payload.kg_event_batch")

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Project KG delta events into AG state and action deltas."""

        kg_refs = [ref for ref in agent_input.graph_refs if ref.graph == GraphScope.KG]
        batch = self._parse_event_batch(agent_input)
        goal_context = self._coerce_mapping(agent_input.raw_payload.get("goal_context"))
        policy_context = self._coerce_mapping(agent_input.raw_payload.get("policy_context"))
        source_kg_version = self._source_kg_version(batch)
        ag_version = self._derived_ag_version(batch, state_deltas_count_hint=len(batch.events))

        state_deltas: list[dict[str, Any]] = []
        affected_ag_refs: list[GraphRef] = []
        seen_ag_ref_keys: set[tuple[str, str, str | None]] = set()

        for event in batch.events:
            state_nodes = (
                self._project_entity_to_state_nodes(
                    event=event,
                    goal_context=goal_context,
                    policy_context=policy_context,
                )
                if self._is_entity_event(event)
                else self._project_relation_to_state_nodes(
                    event=event,
                    goal_context=goal_context,
                    policy_context=policy_context,
                )
            )
            action_candidates = self._instantiate_action_candidates(
                event=event,
                state_nodes=state_nodes,
                goal_context=goal_context,
                policy_context=policy_context,
            )
            for patch in [*state_nodes, *action_candidates]:
                delta = self._build_ag_delta(patch=patch, event=event)
                state_deltas.append(delta)
                target_ref = GraphRef.model_validate(delta["target_ref"])
                key = (target_ref.graph.value, target_ref.ref_id, target_ref.ref_type)
                if key not in seen_ag_ref_keys:
                    seen_ag_ref_keys.add(key)
                    affected_ag_refs.append(target_ref)

        emitted_events = self._build_projection_events(
            batch=batch,
            affected_ag_refs=affected_ag_refs,
            goal_context=goal_context,
            policy_context=policy_context,
            state_deltas=state_deltas,
            source_kg_version=source_kg_version,
            ag_version=max(ag_version, len(state_deltas)),
        )

        logs = [
            f"consumed {len(batch.events)} KG delta event(s)",
            f"scoped projection to {len(kg_refs)} KG ref(s)",
            f"anchored AG projection to KG version {source_kg_version if source_kg_version is not None else 'unknown'}",
            f"prepared {len(state_deltas)} AG delta(s) without full AG rebuild",
            "projection only writes AG-scoped deltas and emits AG events",
            "no TG writes or replanning requests were produced",
        ]
        if goal_context:
            logs.append("goal_context applied to AG projection scoring")
        if policy_context:
            logs.append("policy_context applied to AG projection gating")

        return AgentOutput(
            state_deltas=state_deltas,
            emitted_events=[event.model_dump(mode="json") for event in emitted_events],
            logs=logs,
        )

    def project_with_projector(
        self,
        *,
        projector: AttackGraphProjector | None = None,
        changed_refs: Sequence[str] | None = None,
        goal_context: dict[str, Any] | None = None,
        policy_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a compatibility-oriented handoff for `AttackGraphProjector`.

        This agent does not call into a store or reconstruct AG directly, but
        the returned payload is shaped so an orchestrator can hand it to the
        existing `AttackGraphProjector.project_incremental(...)` path.
        """

        resolved = projector or AttackGraphProjector()
        return {
            "projector": resolved.__class__.__name__,
            "changed_refs": list(changed_refs or []),
            "goal_context": dict(goal_context or {}),
            "policy_context": dict(policy_context or {}),
            "source_kg_version": goal_context.get("source_kg_version") if isinstance(goal_context, dict) else None,
        }

    def _project_entity_to_state_nodes(
        self,
        *,
        event: KGDeltaEvent,
        goal_context: dict[str, Any],
        policy_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Project one KG entity delta into AG state node patch candidates."""

        target_type = str(event.target_ref.ref_type or "").lower()
        patch = dict(event.patch)
        attributes = self._coerce_mapping(patch.get("attributes"))
        properties = self._coerce_mapping(attributes.get("properties"))
        confidence = self._extract_confidence(attributes, patch)
        truth_status = self._truth_for_event(event)
        goal_relevance = self._goal_relevance(event=event, goal_context=goal_context)
        subject_refs = [self._to_ag_subject_ref(event.target_ref)]

        state_type = self._entity_state_type_from_ref_type(target_type, confidence)
        if state_type is None:
            return []

        state_ref = self._build_ag_node_ref(
            node_id=stable_node_id(
                "ag-state",
                {
                    "state_type": state_type,
                    "subject_refs": [ref.ref_id for ref in subject_refs],
                    "origin_event": event.event_id,
                },
            ),
            ref_type="StateNode",
        )
        return [
            {
                "node_kind": "state",
                "node_id": state_ref.ref_id,
                "node_type": state_type,
                "label": self._build_state_label(state_type=state_type, event=event),
                "subject_refs": [ref.model_dump(mode="json") for ref in subject_refs],
                "created_from": [self._to_ag_subject_ref(event.target_ref).model_dump(mode="json")],
                "source_refs": [self._to_ag_subject_ref(event.target_ref).model_dump(mode="json")],
                "truth_status": truth_status,
                "confidence": confidence,
                "goal_relevance": goal_relevance,
                "properties": {
                    "projection_source": "kg_entity_delta",
                    "kg_event_type": event.event_type.value,
                    "policy_hints": dict(policy_context),
                    **properties,
                },
                "projection_traces": [
                    ProjectionTrace(
                        rule=f"entity:{target_type or 'unknown'}",
                        input_refs=[self._to_trace_ref(event.target_ref)],
                        metadata={"kg_event_id": event.event_id},
                    ).model_dump(mode="json")
                ],
            }
        ]

    def _project_relation_to_state_nodes(
        self,
        *,
        event: KGDeltaEvent,
        goal_context: dict[str, Any],
        policy_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Project one KG relation delta into AG state node patch candidates."""

        patch = dict(event.patch)
        relation_type = str(
            patch.get("relation_type")
            or patch.get("entity_type")
            or event.target_ref.ref_type
            or ""
        ).upper()
        source_id = str(patch.get("source") or "")
        target_id = str(patch.get("target") or "")
        confidence = self._extract_confidence(self._coerce_mapping(patch.get("attributes")), patch)
        goal_relevance = self._goal_relevance(event=event, goal_context=goal_context)
        truth_status = self._truth_for_event(event)

        state_type = self._relation_state_type(relation_type, confidence)
        if state_type is None:
            return []

        subject_refs = [
            GraphRef(graph=GraphScope.KG, ref_id=source_id, ref_type="relation_source"),
            GraphRef(graph=GraphScope.KG, ref_id=target_id, ref_type="relation_target"),
        ]
        node_id = stable_node_id(
            "ag-state",
            {
                "state_type": state_type,
                "subject_refs": [ref.ref_id for ref in subject_refs],
                "relation_type": relation_type,
            },
        )
        return [
            {
                "node_kind": "state",
                "node_id": node_id,
                "node_type": state_type,
                "label": self._build_relation_state_label(state_type=state_type, relation_type=relation_type),
                "subject_refs": [ref.model_dump(mode="json") for ref in subject_refs],
                "created_from": [self._to_ag_subject_ref(event.target_ref).model_dump(mode="json")],
                "source_refs": [self._to_ag_subject_ref(event.target_ref).model_dump(mode="json")],
                "truth_status": truth_status,
                "confidence": confidence,
                "goal_relevance": goal_relevance,
                "properties": {
                    "projection_source": "kg_relation_delta",
                    "kg_event_type": event.event_type.value,
                    "relation_type": relation_type,
                    "source_id": source_id,
                    "target_id": target_id,
                    "policy_hints": dict(policy_context),
                },
                "projection_traces": [
                    ProjectionTrace(
                        rule=f"relation:{relation_type.lower()}",
                        input_refs=[self._to_trace_ref(event.target_ref)],
                        metadata={"kg_event_id": event.event_id},
                    ).model_dump(mode="json")
                ],
            }
        ]

    def _instantiate_action_candidates(
        self,
        *,
        event: KGDeltaEvent,
        state_nodes: Sequence[dict[str, Any]],
        goal_context: dict[str, Any],
        policy_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Instantiate AG action candidates implied by the projected state nodes."""

        actions: list[dict[str, Any]] = []
        approval_required = bool(self._coerce_mapping(policy_context).get("approval_required", False))
        for state_patch in state_nodes:
            state_type = str(state_patch.get("node_type", ""))
            action_type = self._action_type_for_state(state_type)
            if action_type is None:
                continue

            node_id = stable_node_id(
                "ag-action",
                {
                    "action_type": action_type,
                    "state_id": state_patch["node_id"],
                    "event_id": event.event_id,
                },
            )
            source_refs = [self._to_ag_subject_ref(event.target_ref).model_dump(mode="json")]
            actions.append(
                {
                    "node_kind": "action",
                    "node_id": node_id,
                    "node_type": action_type,
                    "label": self._build_action_label(action_type=action_type, state_patch=state_patch),
                    "bound_args": self._action_bound_args(state_patch),
                    "required_inputs": sorted(self._action_bound_args(state_patch).keys()),
                    "precondition_schema": {"state_types": [state_type]},
                    "postcondition_schema": {"derived_from_state": state_type},
                    "required_capabilities": ["planner-managed"],
                    "cost": self._action_cost(action_type),
                    "risk": self._action_risk(action_type),
                    "noise": self._action_noise(action_type),
                    "expected_value": max(float(state_patch.get("goal_relevance", 0.0)), 0.4),
                    "success_probability_prior": max(float(state_patch.get("confidence", 0.5)), 0.35),
                    "goal_relevance": self._goal_relevance(event=event, goal_context=goal_context),
                    "parallelizable": action_type != ActionNodeType.ESTABLISH_MANAGED_SESSION.value,
                    "approval_required": approval_required
                    or action_type == ActionNodeType.ESTABLISH_MANAGED_SESSION.value,
                    "resource_keys": self._resource_keys(state_patch),
                    "activation_status": self._activation_status_for_event(event),
                    "activation_conditions": [
                        {
                            "key": "required_states",
                            "required_refs": [
                                self._build_ag_node_ref(
                                    node_id=state_patch["node_id"],
                                    ref_type="StateNode",
                                ).model_dump(mode="json")
                            ],
                            "expression": {"kg_event_type": event.event_type.value},
                            "status": self._activation_status_for_event(event),
                        }
                    ],
                    "source_refs": source_refs,
                    "projection_traces": [
                        ProjectionTrace(
                            rule=f"action:{action_type.lower()}",
                        input_refs=[self._to_trace_ref(event.target_ref)],
                        metadata={"kg_event_id": event.event_id},
                    ).model_dump(mode="json")
                ],
                    "properties": {
                        "projection_source": "incremental_graph_projection",
                        "depends_on_state_id": state_patch["node_id"],
                    },
                }
            )
        return actions

    def _build_ag_delta(
        self,
        *,
        patch: dict[str, Any],
        event: KGDeltaEvent,
    ) -> dict[str, Any]:
        """Wrap one projected AG patch as a formal AG state delta."""

        node_kind = str(patch["node_kind"])
        delta_type = f"upsert_{node_kind}"
        ref_type = "StateNode" if node_kind == "state" else "ActionNode"
        return StateDeltaRecord(
            source_agent=self.name,
            summary=f"Project {node_kind} from KG event {event.event_id}",
            graph_scope=GraphScope.AG,
            delta_type=delta_type,
            target_ref=self._build_ag_node_ref(node_id=str(patch["node_id"]), ref_type=ref_type),
            patch={
                "projection_source_event": event.model_dump(mode="json"),
                "ag_patch": patch,
            },
            payload={"patch_kind": node_kind},
        ).to_agent_output_fragment() | {"write_type": "structural"}

    def _build_projection_events(
        self,
        *,
        batch: KGEventBatch,
        affected_ag_refs: Sequence[GraphRef],
        goal_context: dict[str, Any],
        policy_context: dict[str, Any],
        state_deltas: Sequence[dict[str, Any]],
        source_kg_version: int | None,
        ag_version: int,
    ) -> list[AGProjectionEvent]:
        """Build AG-facing projection events for downstream consumers."""

        if not state_deltas:
            return []
        return [
            AGProjectionEvent(
                source_agent=self.name,
                affected_ag_refs=[ref.model_dump(mode="json") for ref in affected_ag_refs],
                source_kg_event_ids=[event.event_id for event in batch.events],
                source_kg_version=source_kg_version,
                ag_version=ag_version,
                projection_batch_id=batch.batch_id,
                metadata={
                    "batch_id": batch.batch_id,
                    "delta_count": len(state_deltas),
                    "source_kg_version": source_kg_version,
                    "ag_version": ag_version,
                    "goal_context": dict(goal_context),
                    "policy_context": dict(policy_context),
                    "projector_handoff": self.project_with_projector(
                        changed_refs=[event.target_ref.ref_id for event in batch.events],
                        goal_context={**goal_context, "source_kg_version": source_kg_version},
                        policy_context=policy_context,
                    ),
                },
            )
        ]

    @staticmethod
    def _source_kg_version(batch: KGEventBatch) -> int | None:
        """从 KG event batch 中提取最新的来源 KG 版本。"""

        versions = [
            event.metadata.get("resulting_kg_version")
            or event.metadata.get("base_kg_version")
            or (event.metadata.get("store_apply_request") or {}).get("resulting_kg_version")
            or (event.metadata.get("store_apply_request") or {}).get("base_kg_version")
            for event in batch.events
            if isinstance(event.metadata, dict)
        ]
        normalized = []
        for version in versions:
            try:
                if version is not None:
                    normalized.append(int(version))
            except (TypeError, ValueError):
                continue
        return max(normalized) if normalized else None

    @staticmethod
    def _derived_ag_version(batch: KGEventBatch, *, state_deltas_count_hint: int) -> int:
        """给增量投影计算一个轻量 AG 版本号。

        当前阶段不引入 AG store，因此这里用“批次大小 + 来源 batch”生成可追踪版本。
        后续若接入正式 AG store，可以直接替换为 store 返回的版本号。
        """

        return max(len(batch.events), state_deltas_count_hint, 1)

    def _parse_event_batch(self, agent_input: AgentInput) -> KGEventBatch:
        """Normalize one event or batch payload into `KGEventBatch`."""

        raw_batch = agent_input.raw_payload.get("kg_event_batch")
        if isinstance(raw_batch, KGEventBatch):
            return raw_batch
        if isinstance(raw_batch, dict):
            return KGEventBatch.model_validate(raw_batch)

        raw_events = agent_input.raw_payload.get("kg_events")
        if raw_events is None:
            single_event = agent_input.raw_payload.get("kg_event")
            if single_event is None:
                return KGEventBatch()
            raw_events = [single_event]

        events = [
            item if isinstance(item, KGDeltaEvent) else KGDeltaEvent.model_validate(item)
            for item in (raw_events if isinstance(raw_events, list) else [raw_events])
        ]
        return KGEventBatch.from_events(events)

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any]:
        """Return a shallow mapping copy or an empty mapping."""

        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def _is_entity_event(event: KGDeltaEvent) -> bool:
        """Return True when the event targets a KG entity-like node."""

        return event.event_type in {
            KGDeltaEventType.ENTITY_ADDED,
            KGDeltaEventType.ENTITY_UPDATED,
            KGDeltaEventType.CONFIDENCE_CHANGED,
            KGDeltaEventType.STATE_INVALIDATED,
        }

    @staticmethod
    def _extract_confidence(attributes: dict[str, Any], patch: dict[str, Any]) -> float:
        """Extract and clamp confidence from patch attributes."""

        for candidate in (
            attributes.get("confidence"),
            patch.get("confidence"),
            attributes.get("properties", {}).get("confidence") if isinstance(attributes.get("properties"), dict) else None,
        ):
            try:
                if candidate is None:
                    continue
                return max(0.0, min(1.0, float(candidate)))
            except (TypeError, ValueError):
                continue
        return 0.5

    @staticmethod
    def _truth_for_event(event: KGDeltaEvent) -> str:
        """Map KG event types onto AG truth status strings."""

        if event.event_type == KGDeltaEventType.STATE_INVALIDATED:
            return TruthStatus.STALE.value
        if event.event_type in {KGDeltaEventType.ENTITY_UPDATED, KGDeltaEventType.RELATION_UPDATED}:
            return TruthStatus.ACTIVE.value
        if event.event_type == KGDeltaEventType.CONFIDENCE_CHANGED:
            return TruthStatus.ACTIVE.value
        return TruthStatus.CANDIDATE.value

    def _goal_relevance(self, *, event: KGDeltaEvent, goal_context: dict[str, Any]) -> float:
        """Infer a coarse goal relevance score for one KG event."""

        target_ids = {str(item) for item in goal_context.get("target_ref_ids", [])}
        goal_ids = {str(item) for item in goal_context.get("goal_ids", [])}
        if event.target_ref.ref_id in target_ids or event.target_ref.ref_id in goal_ids:
            return 1.0
        return 0.25 if goal_context else 0.1

    @staticmethod
    def _to_ag_subject_ref(ref: GraphRef) -> GraphRef:
        """Keep the source KG ref in agent-protocol `GraphRef` form."""

        return GraphRef(graph=GraphScope.KG, ref_id=ref.ref_id, ref_type=ref.ref_type, metadata=dict(ref.metadata))

    @staticmethod
    def _to_trace_ref(ref: GraphRef) -> AGGraphRef:
        """Convert protocol refs into AG-model refs for projection traces."""

        return AGGraphRef(graph="kg", ref_id=ref.ref_id, ref_type=ref.ref_type)

    @staticmethod
    def _build_ag_node_ref(*, node_id: str, ref_type: str) -> GraphRef:
        """Build an AG target ref for one projected node."""

        return GraphRef(graph=GraphScope.AG, ref_id=node_id, ref_type=ref_type)

    @staticmethod
    def _entity_state_type_from_ref_type(ref_type: str, confidence: float) -> str | None:
        """Map KG node kinds onto AG planner state types."""

        mapping = {
            "host": StateNodeType.HOST_VALIDATED.value if confidence >= 0.85 else StateNodeType.HOST_KNOWN.value,
            "service": StateNodeType.SERVICE_CONFIRMED.value if confidence >= 0.85 else StateNodeType.SERVICE_KNOWN.value,
            "identity": StateNodeType.IDENTITY_KNOWN.value,
            "credential": StateNodeType.CREDENTIAL_USABLE.value,
            "session": StateNodeType.MANAGED_SESSION_AVAILABLE.value,
            "privilege_state": StateNodeType.PRIVILEGE_VALIDATED.value,
            "data_asset": StateNodeType.DATA_ASSET_KNOWN.value,
            "goal": StateNodeType.GOAL_STATE_SATISFIED.value,
            "observation": StateNodeType.HOST_KNOWN.value,
            "evidence": StateNodeType.SERVICE_KNOWN.value,
        }
        return mapping.get(ref_type)

    @staticmethod
    def _relation_state_type(relation_type: str, confidence: float) -> str | None:
        """Map KG relation types onto AG planner state types."""

        if relation_type == "CAN_REACH":
            return (
                StateNodeType.REACHABILITY_VALIDATED.value
                if confidence >= 0.85
                else StateNodeType.PATH_CANDIDATE.value
            )
        if relation_type == "AUTHENTICATES_AS":
            return StateNodeType.CREDENTIAL_USABLE.value
        if relation_type in {"HOSTS", "OBSERVED_ON", "SUPPORTED_BY", "DERIVED_FROM"}:
            return StateNodeType.SERVICE_KNOWN.value
        return None

    @staticmethod
    def _build_state_label(*, state_type: str, event: KGDeltaEvent) -> str:
        """Build one readable AG state label."""

        return f"{state_type.replace('_', ' ').title()} from {event.target_ref.ref_id}"

    @staticmethod
    def _build_relation_state_label(*, state_type: str, relation_type: str) -> str:
        """Build one readable AG state label for relation-derived states."""

        return f"{state_type.replace('_', ' ').title()} via {relation_type}"

    @staticmethod
    def _action_type_for_state(state_type: str) -> str | None:
        """Map projected AG state types onto candidate AG action types."""

        mapping = {
            StateNodeType.HOST_KNOWN.value: ActionNodeType.ENUMERATE_HOST.value,
            StateNodeType.SERVICE_KNOWN.value: ActionNodeType.VALIDATE_SERVICE.value,
            StateNodeType.PATH_CANDIDATE.value: ActionNodeType.VALIDATE_REACHABILITY.value,
            StateNodeType.IDENTITY_KNOWN.value: ActionNodeType.ENUMERATE_IDENTITY_CONTEXT.value,
            StateNodeType.HOST_VALIDATED.value: ActionNodeType.ESTABLISH_MANAGED_SESSION.value,
            StateNodeType.MANAGED_SESSION_AVAILABLE.value: ActionNodeType.VALIDATE_PRIVILEGE_STATE.value,
            StateNodeType.DATA_ASSET_KNOWN.value: ActionNodeType.LOCATE_GOAL_RELEVANT_DATA.value,
            StateNodeType.GOAL_RELEVANT_DATA_LOCATED.value: ActionNodeType.VALIDATE_GOAL_CONDITION.value,
            StateNodeType.GOAL_STATE_SATISFIED.value: ActionNodeType.VALIDATE_GOAL_CONDITION.value,
        }
        return mapping.get(state_type)

    @staticmethod
    def _build_action_label(*, action_type: str, state_patch: dict[str, Any]) -> str:
        """Build a readable action label."""

        return f"{action_type.replace('_', ' ').title()} for {state_patch['node_id']}"

    @staticmethod
    def _action_bound_args(state_patch: dict[str, Any]) -> dict[str, Any]:
        """Build bound args from one state patch."""

        properties = state_patch.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        return {
            "state_id": state_patch["node_id"],
            **{key: value for key, value in properties.items() if key.endswith("_id")},
        }

    @staticmethod
    def _action_cost(action_type: str) -> float:
        """Return a small default cost prior for one action type."""

        return {
            ActionNodeType.ENUMERATE_HOST.value: 0.15,
            ActionNodeType.VALIDATE_SERVICE.value: 0.2,
            ActionNodeType.VALIDATE_REACHABILITY.value: 0.18,
            ActionNodeType.ENUMERATE_IDENTITY_CONTEXT.value: 0.12,
            ActionNodeType.ESTABLISH_MANAGED_SESSION.value: 0.4,
            ActionNodeType.VALIDATE_PRIVILEGE_STATE.value: 0.35,
            ActionNodeType.LOCATE_GOAL_RELEVANT_DATA.value: 0.22,
            ActionNodeType.VALIDATE_GOAL_CONDITION.value: 0.1,
        }.get(action_type, 0.2)

    @staticmethod
    def _action_risk(action_type: str) -> float:
        """Return a small default risk prior for one action type."""

        return {
            ActionNodeType.ENUMERATE_HOST.value: 0.05,
            ActionNodeType.VALIDATE_SERVICE.value: 0.08,
            ActionNodeType.VALIDATE_REACHABILITY.value: 0.06,
            ActionNodeType.ENUMERATE_IDENTITY_CONTEXT.value: 0.04,
            ActionNodeType.ESTABLISH_MANAGED_SESSION.value: 0.18,
            ActionNodeType.VALIDATE_PRIVILEGE_STATE.value: 0.15,
            ActionNodeType.LOCATE_GOAL_RELEVANT_DATA.value: 0.06,
            ActionNodeType.VALIDATE_GOAL_CONDITION.value: 0.03,
        }.get(action_type, 0.08)

    @staticmethod
    def _action_noise(action_type: str) -> float:
        """Return a small default noise prior for one action type."""

        return {
            ActionNodeType.ENUMERATE_HOST.value: 0.15,
            ActionNodeType.VALIDATE_SERVICE.value: 0.2,
            ActionNodeType.VALIDATE_REACHABILITY.value: 0.12,
            ActionNodeType.ENUMERATE_IDENTITY_CONTEXT.value: 0.08,
            ActionNodeType.ESTABLISH_MANAGED_SESSION.value: 0.15,
            ActionNodeType.VALIDATE_PRIVILEGE_STATE.value: 0.1,
            ActionNodeType.LOCATE_GOAL_RELEVANT_DATA.value: 0.1,
            ActionNodeType.VALIDATE_GOAL_CONDITION.value: 0.03,
        }.get(action_type, 0.1)

    @staticmethod
    def _resource_keys(state_patch: dict[str, Any]) -> list[str]:
        """Build resource keys from one projected state patch."""

        properties = state_patch.get("properties", {})
        if not isinstance(properties, dict):
            return [f"ag_state:{state_patch['node_id']}"]
        keys = [f"{key[:-3]}:{value}" for key, value in properties.items() if key.endswith("_id")]
        return sorted(set(keys or [f"ag_state:{state_patch['node_id']}"]))

    @staticmethod
    def _activation_status_for_event(event: KGDeltaEvent) -> str:
        """Map KG event types onto AG action activation status."""

        if event.event_type == KGDeltaEventType.STATE_INVALIDATED:
            return ActivationStatus.DORMANT.value
        if event.event_type in {KGDeltaEventType.ENTITY_UPDATED, KGDeltaEventType.RELATION_UPDATED}:
            return ActivationStatus.ACTIVATABLE.value
        return ActivationStatus.UNKNOWN.value


__all__ = [
    "AGProjectionEvent",
    "GraphProjectionAgent",
]
