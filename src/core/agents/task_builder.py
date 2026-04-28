"""Task Builder agent that converts planner output into TG patch deltas.

The Task Builder is the TG-side owner for turning planner-selected action
chains into Task Graph subgraph patches. It emits only TG-scoped deltas and
events for an external TG store to apply later.
"""

from __future__ import annotations

from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_models import DecisionRecord, StateDeltaRecord, new_record_id
from src.core.agents.agent_protocol import (
    AgentInput,
    AgentKind,
    AgentOutput,
    BaseAgent,
    GraphRef,
    GraphScope,
    WritePermission,
)
from src.core.graph.tg_builder import TaskCandidate, TaskGraphBuilder, TaskGenerationRequest
from src.core.models.ag import GraphRef as TGGraphRef
from src.core.models.tg import (
    BaseTaskEdge,
    CheckpointNode,
    DecisionNode,
    DependencyType,
    TaskCheckpoint,
    TaskGraph,
    TaskGroupNode,
    TaskGroupType,
    TaskNode,
    stable_node_id,
)


class TaskBuildRequest(BaseModel):
    """Structured request consumed by `TaskBuilderAgent`."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    decision: dict[str, Any]
    candidate_actions: list[str] = Field(default_factory=list)
    task_candidates: list[TaskCandidate] = Field(default_factory=list)
    tg_refs: list[GraphRef] = Field(default_factory=list)
    runtime_hints: dict[str, Any] = Field(default_factory=dict)


class TaskBuildResult(BaseModel):
    """Structured TG patch output produced by `TaskBuilderAgent`."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    build_id: str = Field(default_factory=lambda: new_record_id("task-build"))
    task_node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    group_ids: list[str] = Field(default_factory=list)
    checkpoint_ids: list[str] = Field(default_factory=list)
    state_deltas: list[dict[str, Any]] = Field(default_factory=list)
    emitted_events: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)


class TGDeltaEvent(BaseModel):
    """Task Graph delta event emitted by the Task Builder."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    event_id: str = Field(default_factory=lambda: new_record_id("tg-event"))
    event_type: str = Field(default="tg.delta", min_length=1)
    source_agent: str = Field(min_length=1)
    target_refs: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskBuilderAgent(BaseAgent):
    """Convert planner decision/candidates into TG subgraph deltas."""

    def __init__(self, name: str = "task_builder_agent") -> None:
        self._graph_builder = TaskGraphBuilder()
        super().__init__(
            name=name,
            kind=AgentKind.TASK_BUILDER,
            write_permission=WritePermission(
                scopes=[GraphScope.TG],
                allow_structural_write=True,
                allow_state_write=True,
                allow_event_emit=True,
            ),
        )

    def validate_input(self, agent_input: AgentInput) -> None:
        """Require planner decision input and candidate actions or task candidates."""

        super().validate_input(agent_input)
        self._resolve_request(agent_input)

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Build TG node/edge patches from planner output without applying them."""

        request = self._resolve_request(agent_input)
        decision = self._parse_decision(request.decision)
        tg_snapshot = self._resolve_task_graph(agent_input)

        task_nodes = self._build_task_nodes(request=request)
        dependency_edges = self._build_dependency_edges(task_nodes=task_nodes)
        conflict_edges = self._build_conflict_edges(task_nodes=task_nodes)
        alternative_edges = self._build_alternative_edges(task_nodes=task_nodes)
        group_nodes = self._build_task_groups(task_nodes=task_nodes, request=request)
        checkpoint_nodes = self._build_checkpoint_nodes(task_nodes=task_nodes, request=request)

        state_deltas: list[dict[str, Any]] = []
        target_refs: list[GraphRef] = []

        for node in [*task_nodes, *group_nodes, *checkpoint_nodes]:
            delta = self._build_node_delta(node=node)
            state_deltas.append(delta)
            target_refs.append(GraphRef.model_validate(delta["target_ref"]))

        for edge in [*dependency_edges, *conflict_edges, *alternative_edges]:
            delta = self._build_edge_delta(edge=edge)
            state_deltas.append(delta)
            target_refs.append(GraphRef.model_validate(delta["target_ref"]))

        self._validate_tg_patch(
            tg_snapshot=tg_snapshot,
            task_nodes=task_nodes,
            dependency_edges=dependency_edges,
            conflict_edges=conflict_edges,
            alternative_edges=alternative_edges,
            group_nodes=group_nodes,
            checkpoint_nodes=checkpoint_nodes,
        )

        decisions = self._emit_task_builder_decisions(
            decision=decision,
            task_nodes=task_nodes,
            target_refs=target_refs,
        )
        emitted_events = self._build_tg_events(
            target_refs=target_refs,
            request=request,
            state_deltas=state_deltas,
        )

        logs = [
            f"built {len(task_nodes)} task node(s) from planner candidates",
            f"built {len(dependency_edges)} depends_on edge(s)",
            f"built {len(conflict_edges)} conflict edge(s) and {len(alternative_edges)} alternative edge(s)",
            f"built {len(group_nodes)} task group node(s) and {len(checkpoint_nodes)} checkpoint node(s)",
            "task builder emits TG patch/delta only and does not dispatch workers",
        ]

        return AgentOutput(
            state_deltas=state_deltas,
            emitted_events=[event.model_dump(mode="json") for event in emitted_events],
            decisions=[record.to_agent_output_fragment() for record in decisions],
            logs=logs,
        )

    def _build_task_nodes(self, *, request: TaskBuildRequest) -> list[TaskNode]:
        """Convert planner-selected task candidates into TG task nodes."""

        candidates = self._resolve_task_candidates(request)
        return [self._graph_builder.create_task_node(candidate) for candidate in candidates]

    def _build_dependency_edges(self, *, task_nodes: Sequence[TaskNode]) -> list[BaseTaskEdge]:
        """Create DEPENDS_ON edges from task precondition/output refs."""

        edges: list[BaseTaskEdge] = []
        output_keys = {
            node.id: {ref.key() for ref in node.expected_output_refs}
            for node in task_nodes
        }
        for consumer in task_nodes:
            required = {ref.key() for ref in consumer.precondition_refs}
            if not required:
                continue
            for producer in task_nodes:
                if producer.id == consumer.id:
                    continue
                if required & output_keys.get(producer.id, set()):
                    edges.append(
                        BaseTaskEdge(
                            id=TaskGraph.stable_edge_id(producer.id, consumer.id, DependencyType.DEPENDS_ON),
                            dependency_type=DependencyType.DEPENDS_ON,
                            source=producer.id,
                            target=consumer.id,
                            label="depends_on",
                        )
                    )
        return self._dedupe_edges(edges)

    def _build_conflict_edges(self, *, task_nodes: Sequence[TaskNode]) -> list[BaseTaskEdge]:
        """Create CONFLICTS_WITH edges for non-parallelizable resource collisions."""

        edges: list[BaseTaskEdge] = []
        for index, left in enumerate(task_nodes):
            for right in task_nodes[index + 1 :]:
                if not (left.resource_keys & right.resource_keys):
                    continue
                if left.parallelizable and right.parallelizable:
                    continue
                edges.extend(self._bidirectional_edges(left.id, right.id, DependencyType.CONFLICTS_WITH))
        return self._dedupe_edges(edges)

    def _build_alternative_edges(self, *, task_nodes: Sequence[TaskNode]) -> list[BaseTaskEdge]:
        """Create ALTERNATIVE_TO edges for same-target similar-priority tasks."""

        edges: list[BaseTaskEdge] = []
        for index, left in enumerate(task_nodes):
            left_targets = {ref.key() for ref in left.target_refs}
            for right in task_nodes[index + 1 :]:
                if left.task_type == right.task_type:
                    continue
                if not ({ref.key() for ref in right.target_refs} & left_targets):
                    continue
                if abs(left.goal_relevance - right.goal_relevance) > 0.25:
                    continue
                edges.extend(self._bidirectional_edges(left.id, right.id, DependencyType.ALTERNATIVE_TO))
        return self._dedupe_edges(edges)

    def _validate_tg_patch(
        self,
        *,
        tg_snapshot: TaskGraph | None,
        task_nodes: Sequence[TaskNode],
        dependency_edges: Sequence[BaseTaskEdge],
        conflict_edges: Sequence[BaseTaskEdge],
        alternative_edges: Sequence[BaseTaskEdge],
        group_nodes: Sequence[TaskGroupNode],
        checkpoint_nodes: Sequence[CheckpointNode],
    ) -> None:
        """Validate that the emitted TG patch is structurally self-consistent."""

        known_ids = set(tg_snapshot._nodes) if tg_snapshot is not None else set()
        known_ids.update(node.id for node in task_nodes)
        known_ids.update(node.id for node in group_nodes)
        known_ids.update(node.id for node in checkpoint_nodes)

        for edge in [*dependency_edges, *conflict_edges, *alternative_edges]:
            if edge.source not in known_ids:
                raise ValueError(f"tg patch edge source missing: {edge.id}")
            if edge.target not in known_ids:
                raise ValueError(f"tg patch edge target missing: {edge.id}")

    def _resolve_request(self, agent_input: AgentInput) -> TaskBuildRequest:
        """Normalize agent payload into a `TaskBuildRequest`."""

        raw_request = agent_input.raw_payload.get("task_build_request")
        if isinstance(raw_request, TaskBuildRequest):
            return raw_request
        if isinstance(raw_request, dict):
            return TaskBuildRequest.model_validate(raw_request)

        raw_decision = agent_input.raw_payload.get("decision")
        if raw_decision is None:
            raise ValueError("task builder input requires raw_payload.decision or task_build_request")
        return TaskBuildRequest(
            decision=raw_decision if isinstance(raw_decision, dict) else raw_decision.model_dump(mode="json"),
            candidate_actions=self._coerce_string_list(agent_input.raw_payload.get("candidate_actions")),
            task_candidates=self._parse_task_candidates(agent_input.raw_payload.get("task_candidates")),
            tg_refs=[ref for ref in agent_input.graph_refs if ref.graph == GraphScope.TG],
            runtime_hints=self._coerce_mapping(agent_input.raw_payload.get("runtime_hints")),
        )

    def _parse_decision(self, value: dict[str, Any]) -> DecisionRecord:
        """Parse one planner decision into a typed agent record."""

        return DecisionRecord.model_validate(value)

    def _resolve_task_graph(self, agent_input: AgentInput) -> TaskGraph | None:
        """Resolve an optional TG snapshot used only for validation."""

        raw_graph = agent_input.raw_payload.get("tg_graph") or agent_input.raw_payload.get("task_graph")
        if isinstance(raw_graph, TaskGraph):
            return raw_graph
        if isinstance(raw_graph, dict):
            return TaskGraph.from_dict(raw_graph)
        return None

    def _resolve_task_candidates(self, request: TaskBuildRequest) -> list[TaskCandidate]:
        """Resolve TG-compatible task candidates from request content."""

        if request.task_candidates:
            return list(request.task_candidates)

        planning_candidate = self._coerce_mapping(request.decision.get("payload", {})).get("planning_candidate")
        if isinstance(planning_candidate, dict):
            raw_task_candidates = planning_candidate.get("task_candidates")
            parsed = self._parse_task_candidates(raw_task_candidates)
            if parsed:
                return parsed

        candidate_actions = request.candidate_actions
        if not candidate_actions and isinstance(planning_candidate, dict):
            candidate_actions = self._coerce_string_list(planning_candidate.get("action_ids"))
        if not candidate_actions:
            raise ValueError("task builder input requires task_candidates or candidate_actions")

        # Fallback path when only action IDs are available: build skeletal task candidates.
        return [
            TaskCandidate(
                source_action_id=action_id,
                task_type=self._infer_task_type_from_action_id(action_id),
                input_bindings={"source_action_id": action_id},
                target_refs=[GraphRef(graph=GraphScope.AG, ref_id=action_id, ref_type="ActionNode")],
                expected_output_refs=[
                    GraphRef(graph=GraphScope.QUERY if hasattr(GraphScope, "QUERY") else GraphScope.AG, ref_id=f"task-output::{action_id}", ref_type="TaskOutput")
                ],
                tags={"planner_fallback"},
            )
            for action_id in candidate_actions
        ]

    def _build_task_groups(
        self,
        *,
        task_nodes: Sequence[TaskNode],
        request: TaskBuildRequest,
    ) -> list[TaskGroupNode]:
        """Create a task group node for this build batch."""

        if not task_nodes:
            return []
        label = str(request.runtime_hints.get("group_label") or "Planner Selected Tasks")
        group_type = TaskGroupType(str(request.runtime_hints.get("group_type") or TaskGroupType.STAGE.value))
        return [self._graph_builder.create_task_group(list(task_nodes), group_type, label)]

    def _build_checkpoint_nodes(
        self,
        *,
        task_nodes: Sequence[TaskNode],
        request: TaskBuildRequest,
    ) -> list[CheckpointNode]:
        """Create lightweight checkpoint anchors for the built subgraph."""

        if not task_nodes:
            return []
        anchor_task = task_nodes[0]
        checkpoint_id = stable_node_id(
            "tg-checkpoint",
            {"task_ids": [task.id for task in task_nodes], "runtime_hints": request.runtime_hints},
        )
        return [
            CheckpointNode(
                id=checkpoint_id,
                label=f"Checkpoint for {anchor_task.id}",
                anchor_refs=[TGGraphRef(graph="tg", ref_id=task.id, ref_type="Task") for task in task_nodes],
                source_task_id=anchor_task.id,
                properties={"generated_by": self.name},
            )
        ]

    def _build_node_delta(self, *, node: TaskNode | TaskGroupNode | CheckpointNode) -> dict[str, Any]:
        """Wrap one TG node patch as a structural TG delta."""

        ref_type = "TaskNode"
        if isinstance(node, TaskGroupNode):
            ref_type = "TaskGroupNode"
        if isinstance(node, CheckpointNode):
            ref_type = "CheckpointNode"
        return StateDeltaRecord(
            source_agent=self.name,
            summary=f"Upsert TG node {node.id}",
            graph_scope=GraphScope.TG,
            delta_type="upsert_node",
            target_ref=GraphRef(graph=GraphScope.TG, ref_id=node.id, ref_type=ref_type),
            patch={"node_kind": node.kind, "node": node.model_dump(mode="json")},
            payload={"patch_kind": "node"},
        ).to_agent_output_fragment() | {"write_type": "structural"}

    def _build_edge_delta(self, *, edge: BaseTaskEdge) -> dict[str, Any]:
        """Wrap one TG edge patch as a structural TG delta."""

        return StateDeltaRecord(
            source_agent=self.name,
            summary=f"Upsert TG edge {edge.id}",
            graph_scope=GraphScope.TG,
            delta_type="upsert_edge",
            target_ref=GraphRef(graph=GraphScope.TG, ref_id=edge.id, ref_type="TaskEdge"),
            patch={"edge": edge.model_dump(mode="json")},
            payload={"patch_kind": "edge"},
        ).to_agent_output_fragment() | {"write_type": "structural"}

    def _emit_task_builder_decisions(
        self,
        *,
        decision: DecisionRecord,
        task_nodes: Sequence[TaskNode],
        target_refs: Sequence[GraphRef],
    ) -> list[DecisionRecord]:
        """Emit an optional TG-side decision summary for downstream consumers."""

        return [
            DecisionRecord(
                source_agent=self.name,
                summary=f"Built TG subgraph from planner decision {decision.id}",
                confidence=decision.confidence,
                refs=[*decision.refs, *target_refs],
                payload={
                    "source_decision_id": decision.id,
                    "task_ids": [task.id for task in task_nodes],
                },
                decision_type="task_graph_build",
                score=decision.score,
                target_refs=list(target_refs),
                rationale=f"converted planner decision into {len(task_nodes)} TG task node(s)",
            )
        ]

    def _build_tg_events(
        self,
        *,
        target_refs: Sequence[GraphRef],
        request: TaskBuildRequest,
        state_deltas: Sequence[dict[str, Any]],
    ) -> list[TGDeltaEvent]:
        """Emit TG delta events describing the built subgraph patch."""

        if not state_deltas:
            return []
        return [
            TGDeltaEvent(
                source_agent=self.name,
                target_refs=[ref.model_dump(mode="json") for ref in target_refs],
                metadata={
                    "decision_id": request.decision.get("id"),
                    "delta_count": len(state_deltas),
                    "runtime_hints": dict(request.runtime_hints),
                },
            )
        ]

    @staticmethod
    def _dedupe_edges(edges: Sequence[BaseTaskEdge]) -> list[BaseTaskEdge]:
        """Drop duplicate edge IDs while preserving order."""

        result: list[BaseTaskEdge] = []
        seen: set[str] = set()
        for edge in edges:
            if edge.id in seen:
                continue
            seen.add(edge.id)
            result.append(edge)
        return result

    @staticmethod
    def _bidirectional_edges(left_id: str, right_id: str, dependency_type: DependencyType) -> list[BaseTaskEdge]:
        """Create both directions for symmetric TG edge categories."""

        return [
            BaseTaskEdge(
                id=TaskGraph.stable_edge_id(source, target, dependency_type),
                dependency_type=dependency_type,
                source=source,
                target=target,
                label=dependency_type.value.lower(),
            )
            for source, target in ((left_id, right_id), (right_id, left_id))
        ]

    @staticmethod
    def _parse_task_candidates(value: Any) -> list[TaskCandidate]:
        """Normalize serialized task candidates into typed models."""

        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [
            item if isinstance(item, TaskCandidate) else TaskCandidate.model_validate(item)
            for item in items
        ]

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any]:
        """Return a shallow mapping copy or an empty mapping."""

        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def _coerce_string_list(value: Any) -> list[str]:
        """Normalize scalar or list input into a stable string list."""

        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [str(item) for item in items if str(item).strip()]

    @staticmethod
    def _infer_task_type_from_action_id(action_id: str):  # type: ignore[no-untyped-def]
        """Infer a coarse task type when only an action ID is available."""

        from src.core.models.tg import TaskType

        lowered = action_id.lower()
        if "service" in lowered:
            return TaskType.SERVICE_VALIDATION
        if "reach" in lowered:
            return TaskType.REACHABILITY_VALIDATION
        if "identity" in lowered or "session" in lowered:
            return TaskType.IDENTITY_CONTEXT_CONFIRMATION
        if "privilege" in lowered:
            return TaskType.PRIVILEGE_CONFIGURATION_VALIDATION
        if "goal" in lowered:
            return TaskType.GOAL_CONDITION_VALIDATION
        return TaskType.ASSET_CONFIRMATION


__all__ = [
    "TaskBuildRequest",
    "TaskBuildResult",
    "TaskBuilderAgent",
]
