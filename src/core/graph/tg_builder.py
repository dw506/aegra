"""AG -> TG contract models and task graph generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.ag import AGEdgeType, ActionNode, ActionNodeType, ActivationStatus, AttackGraph, GraphRef, stable_node_id
from src.core.models.tg import (
    BaseTaskEdge,
    BaseTaskNode,
    DecisionNode,
    DependencyType,
    RetryPolicy,
    TaskGroupNode,
    TaskGroupType,
    TaskGraph,
    TaskStatus,
    TaskNode,
    TaskType,
)


class TaskCandidate(BaseModel):
    """TaskGraph-facing task candidate derived from one AG action."""

    model_config = ConfigDict(extra="forbid")

    source_action_id: str
    task_type: TaskType
    input_bindings: dict[str, object] = Field(default_factory=dict)
    target_refs: list[GraphRef] = Field(default_factory=list)
    precondition_refs: list[GraphRef] = Field(default_factory=list)
    expected_output_refs: list[GraphRef] = Field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_risk: float = 0.0
    estimated_noise: float = 0.0
    goal_relevance: float = 0.0
    resource_keys: set[str] = Field(default_factory=set)
    approval_required: bool = False
    parallelizable: bool = False
    tags: set[str] = Field(default_factory=set)


class TaskGenerationRequest(BaseModel):
    """Input contract for AG -> TG task generation."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[TaskCandidate] = Field(default_factory=list)
    action_ids: list[str] = Field(default_factory=list)
    include_blocked: bool = False
    include_evidence_tasks: bool = True
    group_type: TaskGroupType = TaskGroupType.STAGE
    group_label: str = "Selected Tasks"
    decision_node: DecisionNode | None = None


class TaskGenerationResult(BaseModel):
    """Output contract for AG -> TG task generation."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[TaskCandidate] = Field(default_factory=list)
    task_graph: dict[str, object] | None = None
    created_task_ids: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    source_ag_version: int | None = Field(default=None, ge=0)
    tg_version: int | None = Field(default=None, ge=0)
    frontier_version: str | None = None


class AttackGraphTaskBuilder:
    """Convert AG actions into TG-facing task candidates."""

    def build_candidates(
        self,
        graph: AttackGraph,
        request: TaskGenerationRequest,
    ) -> TaskGenerationResult:
        """Build task candidates from selected AG action IDs."""

        candidates: list[TaskCandidate] = []
        for action_id in request.action_ids:
            node = graph.get_node(action_id)
            if not isinstance(node, ActionNode):
                continue
            if not request.include_blocked and node.activation_status == ActivationStatus.BLOCKED:
                continue
            candidates.append(
                self._normalize_candidate(
                    TaskCandidate(
                        source_action_id=node.id,
                        task_type=self._map_action_to_task_type(node.action_type),
                        input_bindings=dict(node.bound_args),
                        target_refs=list(node.source_refs),
                    precondition_refs=[
                        GraphRef(graph="ag", ref_id=edge.source, ref_type="StateNode")
                        for edge in graph.list_edges()
                        if edge.target == node.id and edge.edge_type == AGEdgeType.REQUIRES
                    ],
                    estimated_cost=node.cost,
                    estimated_risk=node.risk,
                    estimated_noise=node.noise,
                    goal_relevance=node.goal_relevance,
                        resource_keys=set(node.resource_keys),
                        approval_required=node.approval_required,
                        parallelizable=node.parallelizable,
                        tags=set(node.tags),
                    )
                )
            )
        task_graph = self.build_task_graph(graph, request, candidates=candidates)
        frontier_version = TaskGraphBuilder._frontier_version(graph, request, candidates)
        task_graph.set_metadata(
            source_ag_version=graph.version,
            frontier_version=frontier_version,
            metadata={"projection_batch_id": graph.projection_batch_id},
            version=max(task_graph.version, 1),
        )
        return TaskGenerationResult(
            candidates=candidates,
            task_graph=task_graph.to_dict(),
            source_ag_version=graph.version,
            tg_version=task_graph.version,
            frontier_version=frontier_version,
        )

    def build_task_graph(
        self,
        graph: AttackGraph,
        request: TaskGenerationRequest,
        candidates: list[TaskCandidate] | None = None,
    ) -> TaskGraph:
        """Build a Task Graph from selected AG actions."""

        candidates = candidates or self.build_candidates_without_graph(graph, request)
        task_graph = TaskGraph()
        primary_task_ids: dict[str, str] = {}

        for candidate in candidates:
            task = self._candidate_to_task(candidate)
            task_graph.add_node(task)
            primary_task_ids[candidate.source_action_id] = task.id

        self._link_dependencies(graph, task_graph, primary_task_ids)
        self._link_conflicts(graph, task_graph, primary_task_ids)
        TaskGraphBuilder()._link_host_stage_dependencies(
            task_graph,
            [node for node in task_graph._nodes.values() if isinstance(node, BaseTaskNode)],
        )
        if request.include_evidence_tasks:
            self._attach_evidence_tasks(task_graph, candidates)
        task_graph.refresh_blocked_states()
        stable_time = self._stable_timestamp()
        for node in task_graph.list_nodes():
            if isinstance(node, BaseTaskNode):
                node.created_at = stable_time
                node.updated_at = stable_time
        return task_graph

    def build_candidates_without_graph(
        self,
        graph: AttackGraph,
        request: TaskGenerationRequest,
    ) -> list[TaskCandidate]:
        """Build task candidates without recursively creating a graph result."""

        candidates: list[TaskCandidate] = []
        producer_outputs: dict[str, list[GraphRef]] = {}
        for action_id in request.action_ids:
            node = graph.get_node(action_id)
            if not isinstance(node, ActionNode):
                continue
            if not request.include_blocked and node.activation_status == ActivationStatus.BLOCKED:
                continue
            candidate = self._normalize_candidate(
                TaskCandidate(
                source_action_id=node.id,
                task_type=self._map_action_to_task_type(node.action_type),
                input_bindings=dict(node.bound_args),
                target_refs=list(node.source_refs),
                precondition_refs=[],
                expected_output_refs=[
                    GraphRef(
                        graph="query",
                        ref_id=f"task-output::{node.id}::{index}::{ref.ref_id}",
                        ref_type="TaskOutput",
                        label=ref.label,
                    )
                    for index, ref in enumerate(list(node.source_refs) or [GraphRef(graph="ag", ref_id=node.id, ref_type="ActionNode", label=node.label)])
                ],
                estimated_cost=node.cost,
                estimated_risk=node.risk,
                estimated_noise=node.noise,
                goal_relevance=node.goal_relevance,
                resource_keys=set(node.resource_keys),
                approval_required=node.approval_required,
                parallelizable=node.parallelizable,
                tags=set(node.tags),
                )
            )
            producer_outputs[node.id] = list(candidate.expected_output_refs)
            candidates.append(candidate)

        for candidate in candidates:
            required_state_ids = [
                edge.source for edge in graph.list_edges(AGEdgeType.REQUIRES) if edge.target == candidate.source_action_id
            ]
            for required_state_id in required_state_ids:
                for edge in graph.list_edges(AGEdgeType.PRODUCES):
                    if edge.target == required_state_id and edge.source in producer_outputs:
                        candidate.precondition_refs.extend(producer_outputs[edge.source])
            unique = {ref.key(): ref for ref in candidate.precondition_refs}
            candidate.precondition_refs = list(unique.values())
        return candidates

    def _candidate_to_task(self, candidate: TaskCandidate) -> TaskNode:
        candidate = self._normalize_candidate(candidate)
        task_id = TaskGraph.stable_task_id(
            source_action_id=candidate.source_action_id,
            task_type=candidate.task_type,
            input_bindings=dict(candidate.input_bindings),
        )
        priority = min(100, max(1, int(candidate.goal_relevance * 100)))
        return TaskNode(
            id=task_id,
            label=f"{candidate.task_type.value}:{candidate.source_action_id}",
            task_type=candidate.task_type,
            status=TaskStatus.PENDING,
            source_action_id=candidate.source_action_id,
            input_bindings=dict(candidate.input_bindings),
            target_refs=list(candidate.target_refs),
            precondition_refs=list(candidate.precondition_refs),
            expected_output_refs=list(candidate.expected_output_refs),
            source_refs=list(candidate.target_refs),
            estimated_cost=candidate.estimated_cost,
            estimated_risk=candidate.estimated_risk,
            estimated_noise=candidate.estimated_noise,
            goal_relevance=candidate.goal_relevance,
            priority=priority,
            resource_keys=set(candidate.resource_keys),
            parallelizable=candidate.parallelizable,
            approval_required=candidate.approval_required,
            gate_ids={"approval_gate"} if candidate.approval_required else set(),
            retry_policy=RetryPolicy(max_attempts=2 if candidate.task_type == TaskType.EVIDENCE_COLLECTION_AND_ARCHIVAL else 1),
            reason="derived from selected AG action",
            tags=set(candidate.tags),
            created_at=self._stable_timestamp(),
            updated_at=self._stable_timestamp(),
        )

    def _link_dependencies(
        self,
        graph: AttackGraph,
        task_graph: TaskGraph,
        primary_task_ids: dict[str, str],
    ) -> None:
        produced_by_state: dict[str, set[str]] = {}
        for action_id, task_id in primary_task_ids.items():
            for edge in graph.list_edges(AGEdgeType.PRODUCES):
                if edge.source == action_id:
                    produced_by_state.setdefault(edge.target, set()).add(task_id)

        for consumer_action_id, consumer_task_id in primary_task_ids.items():
            for edge in graph.list_edges(AGEdgeType.REQUIRES):
                if edge.target != consumer_action_id:
                    continue
                for producer_task_id in produced_by_state.get(edge.source, set()):
                    dependency_edge = BaseTaskEdge(
                        id=TaskGraph.stable_edge_id(
                            source=producer_task_id,
                            target=consumer_task_id,
                            dependency_type=DependencyType.DEPENDS_ON,
                        ),
                        dependency_type=DependencyType.DEPENDS_ON,
                        source=producer_task_id,
                        target=consumer_task_id,
                        label="depends_on",
                    )
                    if dependency_edge.id not in task_graph._edges:
                        task_graph.add_edge(dependency_edge)

    def _link_conflicts(
        self,
        graph: AttackGraph,
        task_graph: TaskGraph,
        primary_task_ids: dict[str, str],
    ) -> None:
        action_nodes = {
            action_id: graph.get_node(action_id)
            for action_id in primary_task_ids
        }
        action_ids = sorted(primary_task_ids)
        for index, left_action_id in enumerate(action_ids):
            left_action = action_nodes[left_action_id]
            assert isinstance(left_action, ActionNode)
            for right_action_id in action_ids[index + 1 :]:
                right_action = action_nodes[right_action_id]
                assert isinstance(right_action, ActionNode)
                shares_resource = bool(left_action.resource_keys & right_action.resource_keys)
                shared_locks = self._shared_lock_keys(left_action.resource_keys, right_action.resource_keys)
                if shared_locks or (shares_resource and (not left_action.parallelizable or not right_action.parallelizable)):
                    self._add_conflict_edge(
                        task_graph,
                        primary_task_ids[left_action_id],
                        primary_task_ids[right_action_id],
                        DependencyType.CONFLICTS_WITH,
                    )

                for edge in graph.list_edges(AGEdgeType.COMPETES_WITH):
                    if {edge.source, edge.target} == {left_action_id, right_action_id}:
                        self._add_conflict_edge(
                            task_graph,
                            primary_task_ids[left_action_id],
                            primary_task_ids[right_action_id],
                            DependencyType.ALTERNATIVE_TO,
                        )

    def _attach_evidence_tasks(
        self,
        task_graph: TaskGraph,
        candidates: list[TaskCandidate],
    ) -> None:
        for candidate in candidates:
            parent_task_id = TaskGraph.stable_task_id(
                source_action_id=candidate.source_action_id,
                task_type=candidate.task_type,
                input_bindings=dict(candidate.input_bindings),
            )
            evidence_bindings = {
                "parent_task_id": parent_task_id,
                "source_action_id": candidate.source_action_id,
            }
            evidence_task = TaskNode(
                id=TaskGraph.stable_task_id(
                    source_action_id=f"{candidate.source_action_id}::evidence",
                    task_type=TaskType.EVIDENCE_COLLECTION_AND_ARCHIVAL,
                    input_bindings=evidence_bindings,
                ),
                label=f"EVIDENCE_COLLECTION_AND_ARCHIVAL:{candidate.source_action_id}",
                task_type=TaskType.EVIDENCE_COLLECTION_AND_ARCHIVAL,
                status=TaskStatus.PENDING,
                source_action_id=candidate.source_action_id,
                input_bindings=evidence_bindings,
                target_refs=list(candidate.target_refs),
                precondition_refs=[GraphRef(graph="tg", ref_id=parent_task_id, ref_type="Task")],
                expected_output_refs=self._derive_expected_outputs(candidate, f"{parent_task_id}::evidence"),
                source_refs=list(candidate.target_refs),
                estimated_cost=max(0.05, candidate.estimated_cost * 0.2),
                estimated_risk=min(1.0, candidate.estimated_risk * 0.5),
                estimated_noise=min(1.0, candidate.estimated_noise * 0.3),
                goal_relevance=candidate.goal_relevance,
                priority=max(1, int(candidate.goal_relevance * 100) - 5),
                resource_keys={*candidate.resource_keys, f"evidence:{candidate.source_action_id}"},
                parallelizable=True,
                approval_required=False,
                retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=1),
                reason="capture and archive evidence for the parent task",
                tags={"evidence"},
                created_at=self._stable_timestamp(),
                updated_at=self._stable_timestamp(),
            )
            if evidence_task.id not in task_graph._nodes:
                task_graph.add_node(evidence_task)
            depends_edge = BaseTaskEdge(
                id=TaskGraph.stable_edge_id(
                    source=parent_task_id,
                    target=evidence_task.id,
                    dependency_type=DependencyType.DEPENDS_ON,
                ),
                dependency_type=DependencyType.DEPENDS_ON,
                source=parent_task_id,
                target=evidence_task.id,
                label="depends_on",
            )
            if depends_edge.id not in task_graph._edges:
                task_graph.add_edge(depends_edge)
            evidence_edge = BaseTaskEdge(
                id=TaskGraph.stable_edge_id(
                    source=evidence_task.id,
                    target=parent_task_id,
                    dependency_type=DependencyType.PRODUCES_EVIDENCE_FOR,
                ),
                dependency_type=DependencyType.PRODUCES_EVIDENCE_FOR,
                source=evidence_task.id,
                target=parent_task_id,
                label="produces_evidence_for",
            )
            if evidence_edge.id not in task_graph._edges:
                task_graph.add_edge(evidence_edge)

    @staticmethod
    def _derive_expected_outputs(candidate: TaskCandidate, task_id: str) -> list[GraphRef]:
        refs = list(candidate.target_refs) or [GraphRef(graph="ag", ref_id=candidate.source_action_id, ref_type="ActionNode")]
        return [
            GraphRef(
                graph="query",
                ref_id=f"task-output::{task_id}::{index}::{ref.ref_id}",
                ref_type="TaskOutput",
                label=ref.label,
            )
            for index, ref in enumerate(refs)
        ]

    def _add_conflict_edge(
        self,
        task_graph: TaskGraph,
        left_task_id: str,
        right_task_id: str,
        dependency_type: DependencyType,
    ) -> None:
        for source, target in ((left_task_id, right_task_id), (right_task_id, left_task_id)):
            edge = BaseTaskEdge(
                id=TaskGraph.stable_edge_id(source=source, target=target, dependency_type=dependency_type),
                dependency_type=dependency_type,
                source=source,
                target=target,
                label=dependency_type.value.lower(),
            )
            if edge.id not in task_graph._edges:
                task_graph.add_edge(edge)

    @staticmethod
    def _map_action_to_task_type(action_type: ActionNodeType) -> TaskType:
        mapping = {
            ActionNodeType.ENUMERATE_HOST: TaskType.ASSET_CONFIRMATION,
            ActionNodeType.VALIDATE_SERVICE: TaskType.SERVICE_VALIDATION,
            ActionNodeType.VALIDATE_REACHABILITY: TaskType.REACHABILITY_VALIDATION,
            ActionNodeType.ESTABLISH_PIVOT_ROUTE: TaskType.REACHABILITY_VALIDATION,
            ActionNodeType.ESTABLISH_MANAGED_SESSION: TaskType.IDENTITY_CONTEXT_CONFIRMATION,
            ActionNodeType.REUSE_CREDENTIAL_ON_HOST: TaskType.IDENTITY_CONTEXT_CONFIRMATION,
            ActionNodeType.EXPLOIT_LATERAL_SERVICE: TaskType.IDENTITY_CONTEXT_CONFIRMATION,
            ActionNodeType.ENUMERATE_IDENTITY_CONTEXT: TaskType.IDENTITY_CONTEXT_CONFIRMATION,
            ActionNodeType.VALIDATE_PRIVILEGE_STATE: TaskType.PRIVILEGE_CONFIGURATION_VALIDATION,
            ActionNodeType.LOCATE_GOAL_RELEVANT_DATA: TaskType.EVIDENCE_COLLECTION_AND_ARCHIVAL,
            ActionNodeType.VALIDATE_GOAL_CONDITION: TaskType.GOAL_CONDITION_VALIDATION,
        }
        return mapping[action_type]

    @classmethod
    def _normalize_candidate(cls, candidate: TaskCandidate) -> TaskCandidate:
        resource_keys = set(candidate.resource_keys)
        host_ids = cls._derive_resource_ids(candidate, prefixes=("host",), binding_keys=("host_id", "target_host", "source_host", "via_host"))
        credential_ids = cls._derive_resource_ids(candidate, prefixes=("credential",), binding_keys=("credential_id",))
        session_ids = cls._derive_resource_ids(candidate, prefixes=("session",), binding_keys=("session_id", "route_id"))
        for host_id in host_ids:
            resource_keys.add(f"host:{host_id}")
        for credential_id in credential_ids:
            resource_keys.add(f"credential:{credential_id}")
        for session_id in session_ids:
            resource_keys.add(f"session:{session_id}")
        if len(host_ids) > 1:
            resource_keys.add("multi-host")

        cost = min(10.0, candidate.estimated_cost + (0.1 * len(host_ids)) + (0.08 * len(session_ids)))
        risk = min(1.0, candidate.estimated_risk + (0.08 if credential_ids else 0.0) + (0.12 if "pivot" in candidate.source_action_id.lower() else 0.0))
        noise = min(1.0, candidate.estimated_noise + (0.06 if session_ids else 0.0) + (0.1 if "exploit" in candidate.source_action_id.lower() else 0.0))
        return candidate.model_copy(
            update={
                "resource_keys": resource_keys,
                "estimated_cost": cost,
                "estimated_risk": risk,
                "estimated_noise": noise,
            }
        )

    @staticmethod
    def _shared_lock_keys(left: set[str], right: set[str]) -> set[str]:
        prefixes = ("host:", "credential:", "session:")
        return {key for key in left & right if key.startswith(prefixes)}

    @classmethod
    def _derive_resource_ids(
        cls,
        candidate: TaskCandidate,
        *,
        prefixes: tuple[str, ...],
        binding_keys: tuple[str, ...],
    ) -> set[str]:
        result: set[str] = set()
        for ref in candidate.target_refs:
            if ref.ref_type is None:
                continue
            if ref.ref_type.lower().startswith(tuple(prefix.title() for prefix in prefixes)):
                result.add(ref.ref_id)
        for key in binding_keys:
            cls._collect_binding_ids(candidate.input_bindings.get(key), result)
        return result

    @classmethod
    def _collect_binding_ids(cls, value: Any, result: set[str]) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            cls._collect_binding_ids(value.get("id"), result)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                cls._collect_binding_ids(item, result)
            return
        text = str(value).strip()
        if text:
            result.add(text)

    @staticmethod
    def _stable_timestamp() -> datetime:
        """Return a deterministic timestamp for graph-building output."""

        return datetime(2000, 1, 1, tzinfo=timezone.utc)

    @classmethod
    def _normalize_graph_timestamps(cls, task_graph: TaskGraph) -> None:
        """Normalize node timestamps after status recomputation mutates them."""

        stable_time = cls._stable_timestamp()
        for node in task_graph._nodes.values():
            if hasattr(node, "created_at"):
                node.created_at = stable_time
            if hasattr(node, "updated_at"):
                node.updated_at = stable_time


class TaskGraphBuilder:
    """Build a TaskGraph subgraph from planner-selected task candidates."""

    def build_from_candidates(self, request: TaskGenerationRequest) -> TaskGenerationResult:
        """Build a TaskGraph from task candidates only."""

        task_graph = TaskGraph()
        candidates = self._dedupe_candidates(request.candidates)
        tasks = [self.create_task_node(candidate) for candidate in candidates]

        for task in tasks:
            if task.id not in task_graph._nodes:
                task_graph.add_node(task)

        if tasks:
            group = self.create_task_group(tasks, request.group_type, request.group_label)
            if group.id not in task_graph._nodes:
                task_graph.add_node(group)
            for task in tasks:
                edge = BaseTaskEdge(
                    id=TaskGraph.stable_edge_id(group.id, task.id, DependencyType.DERIVED_FROM),
                    dependency_type=DependencyType.DERIVED_FROM,
                    source=group.id,
                    target=task.id,
                    label="group_contains",
                )
                if edge.id not in task_graph._edges:
                    task_graph.add_edge(edge)

        self.link_dependencies(task_graph)
        self.link_conflicts(task_graph)
        self.link_alternatives(task_graph)
        if request.include_evidence_tasks:
            self._attach_evidence_tasks(task_graph, candidates)
        if request.decision_node is not None:
            self.attach_decision(task_graph, request.decision_node)

        errors = self.validate_graph(task_graph)
        task_graph.refresh_blocked_states()
        self._normalize_graph_timestamps(task_graph)
        frontier_version = self._standalone_frontier_version(candidates)
        task_graph.set_metadata(
            source_ag_version=None,
            frontier_version=frontier_version,
            metadata={"group_label": request.group_label},
            version=max(task_graph.version, 1),
        )
        return TaskGenerationResult(
            candidates=candidates,
            task_graph=task_graph.to_dict(),
            created_task_ids=sorted(
                node.id for node in task_graph._nodes.values() if isinstance(node, BaseTaskNode)
            ),
            validation_errors=errors,
            source_ag_version=None,
            tg_version=task_graph.version,
            frontier_version=frontier_version,
        )

    @staticmethod
    def _frontier_version(
        graph: AttackGraph,
        request: TaskGenerationRequest,
        candidates: list[TaskCandidate],
    ) -> str:
        """为一次 AG->TG 构建生成稳定的 frontier 版本标识。"""

        payload = json.dumps(
            {
                "ag_version": graph.version,
                "source_action_ids": sorted(request.action_ids),
                "candidate_action_ids": sorted(candidate.source_action_id for candidate in candidates),
            },
            sort_keys=True,
        )
        return stable_node_id("tg-frontier", {"payload": payload})

    @staticmethod
    def _standalone_frontier_version(candidates: list[TaskCandidate]) -> str:
        """为无 AG 上下文的 TG 构建生成轻量 frontier 标识。"""

        payload = json.dumps(
            {
                "candidate_action_ids": sorted(candidate.source_action_id for candidate in candidates),
                "candidate_count": len(candidates),
            },
            sort_keys=True,
        )
        return stable_node_id("tg-frontier", {"payload": payload})

    def create_task_node(self, candidate: TaskCandidate) -> TaskNode:
        """Create a stable task node from one candidate."""

        candidate = AttackGraphTaskBuilder._normalize_candidate(candidate)
        task_id = TaskGraph.stable_task_id(
            source_action_id=candidate.source_action_id,
            task_type=candidate.task_type,
            input_bindings=dict(candidate.input_bindings),
        )
        priority = min(100, max(1, int(candidate.goal_relevance * 100)))
        expected_outputs = candidate.expected_output_refs or self._derive_expected_outputs(candidate, task_id)
        return TaskNode(
            id=task_id,
            label=f"{candidate.task_type.value}:{candidate.source_action_id}",
            task_type=candidate.task_type,
            status=TaskStatus.PENDING,
            source_action_id=candidate.source_action_id,
            input_bindings=dict(candidate.input_bindings),
            target_refs=list(candidate.target_refs),
            precondition_refs=list(candidate.precondition_refs),
            expected_output_refs=expected_outputs,
            source_refs=list(candidate.target_refs),
            estimated_cost=candidate.estimated_cost,
            estimated_risk=candidate.estimated_risk,
            estimated_noise=candidate.estimated_noise,
            goal_relevance=candidate.goal_relevance,
            priority=priority,
            resource_keys=set(candidate.resource_keys),
            parallelizable=candidate.parallelizable,
            approval_required=candidate.approval_required,
            gate_ids={"approval_gate"} if candidate.approval_required else set(),
            retry_policy=RetryPolicy(
                max_attempts=2 if candidate.task_type == TaskType.EVIDENCE_COLLECTION_AND_ARCHIVAL else 1
            ),
            reason="derived from selected AG action candidate",
            tags=set(candidate.tags),
            created_at=self._stable_timestamp(),
            updated_at=self._stable_timestamp(),
        )

    def create_task_group(
        self,
        tasks: list[TaskNode],
        group_type: TaskGroupType,
        label: str,
    ) -> TaskGroupNode:
        """Create a stable group node for same-stage tasks."""

        return TaskGroupNode(
            id=self._stable_group_id(tasks, group_type, label),
            group_type=group_type,
            label=label,
            task_ids=sorted(task.id for task in tasks),
            tags={group_type.value.lower()},
            created_at=self._stable_timestamp(),
        )

    def link_dependencies(self, task_graph: TaskGraph) -> None:
        """Link tasks whose preconditions are satisfied by upstream outputs."""

        tasks = [node for node in task_graph._nodes.values() if isinstance(node, BaseTaskNode)]
        for consumer in tasks:
            required = {ref.key() for ref in consumer.precondition_refs}
            if not required:
                continue
            for producer in tasks:
                if producer.id == consumer.id:
                    continue
                produced = {ref.key() for ref in producer.expected_output_refs}
                if required & produced:
                    edge = BaseTaskEdge(
                        id=TaskGraph.stable_edge_id(producer.id, consumer.id, DependencyType.DEPENDS_ON),
                        dependency_type=DependencyType.DEPENDS_ON,
                        source=producer.id,
                        target=consumer.id,
                        label="depends_on",
                    )
                    if edge.id not in task_graph._edges:
                        task_graph.add_edge(edge)
        self._link_host_stage_dependencies(task_graph, tasks)

    def link_conflicts(self, task_graph: TaskGraph) -> None:
        """Link resource-conflicting tasks."""

        tasks = [node for node in task_graph._nodes.values() if isinstance(node, BaseTaskNode)]
        for index, left in enumerate(tasks):
            for right in tasks[index + 1 :]:
                shared_resources = left.resource_keys & right.resource_keys
                if not shared_resources:
                    continue
                shared_locks = AttackGraphTaskBuilder._shared_lock_keys(left.resource_keys, right.resource_keys)
                if not shared_locks and left.parallelizable and right.parallelizable:
                    continue
                self._add_bidirectional_edge(task_graph, left.id, right.id, DependencyType.CONFLICTS_WITH)

    def link_alternatives(self, task_graph: TaskGraph) -> None:
        """Link same-target tasks with different task types as alternatives."""

        tasks = [node for node in task_graph._nodes.values() if isinstance(node, BaseTaskNode)]
        for index, left in enumerate(tasks):
            left_targets = {ref.key() for ref in left.target_refs}
            for right in tasks[index + 1 :]:
                if left.task_type == right.task_type:
                    continue
                if not ({ref.key() for ref in right.target_refs} & left_targets):
                    continue
                if abs(left.goal_relevance - right.goal_relevance) > 0.25:
                    continue
                self._add_bidirectional_edge(task_graph, left.id, right.id, DependencyType.ALTERNATIVE_TO)

    def attach_decision(self, task_graph: TaskGraph, decision_node: DecisionNode) -> None:
        """Attach a decision node and connect it to its option tasks."""

        if decision_node.id not in task_graph._nodes:
            task_graph.add_node(decision_node)
        for task_id in decision_node.option_task_ids:
            if task_id not in task_graph._nodes:
                continue
            edge = BaseTaskEdge(
                id=TaskGraph.stable_edge_id(decision_node.id, task_id, DependencyType.DERIVED_FROM),
                dependency_type=DependencyType.DERIVED_FROM,
                source=decision_node.id,
                target=task_id,
                label="decision_option",
            )
            if edge.id not in task_graph._edges:
                task_graph.add_edge(edge)

    def validate_graph(self, task_graph: TaskGraph) -> list[str]:
        """Validate task graph integrity and duplicate-equivalent tasks."""

        errors: list[str] = []
        signatures: set[str] = set()
        for node in task_graph._nodes.values():
            if isinstance(node, BaseTaskNode):
                signature = json.dumps(
                    {
                        "source_action_id": node.source_action_id or "",
                        "task_type": node.task_type.value,
                        "input_bindings": node.input_bindings,
                    },
                    sort_keys=True,
                    default=str,
                )
                if signature in signatures:
                    errors.append(f"duplicate-equivalent-task:{node.id}")
                signatures.add(signature)
        for edge in task_graph._edges.values():
            if edge.source not in task_graph._nodes:
                errors.append(f"missing-edge-source:{edge.id}")
            if edge.target not in task_graph._nodes:
                errors.append(f"missing-edge-target:{edge.id}")
            if edge.source == edge.target:
                errors.append(f"self-loop:{edge.id}")
        return errors

    def _attach_evidence_tasks(self, task_graph: TaskGraph, candidates: list[TaskCandidate]) -> None:
        for candidate in candidates:
            parent_task_id = TaskGraph.stable_task_id(
                source_action_id=candidate.source_action_id,
                task_type=candidate.task_type,
                input_bindings=dict(candidate.input_bindings),
            )
            evidence_candidate = TaskCandidate(
                source_action_id=f"{candidate.source_action_id}::evidence",
                task_type=TaskType.EVIDENCE_COLLECTION_AND_ARCHIVAL,
                input_bindings={
                    "parent_task_id": parent_task_id,
                    "source_action_id": candidate.source_action_id,
                },
                target_refs=list(candidate.target_refs),
                precondition_refs=self._derive_expected_outputs(candidate, parent_task_id),
                expected_output_refs=self._derive_expected_outputs(candidate, f"{parent_task_id}::evidence"),
                estimated_cost=max(0.05, candidate.estimated_cost * 0.2),
                estimated_risk=min(1.0, candidate.estimated_risk * 0.5),
                estimated_noise=min(1.0, candidate.estimated_noise * 0.3),
                goal_relevance=candidate.goal_relevance,
                resource_keys={*candidate.resource_keys, f"evidence:{candidate.source_action_id}"},
                parallelizable=True,
                approval_required=False,
                tags={"evidence"},
            )
            evidence_task = self.create_task_node(evidence_candidate)
            if evidence_task.id not in task_graph._nodes:
                task_graph.add_node(evidence_task)
            dep_edge = BaseTaskEdge(
                id=TaskGraph.stable_edge_id(parent_task_id, evidence_task.id, DependencyType.DEPENDS_ON),
                dependency_type=DependencyType.DEPENDS_ON,
                source=parent_task_id,
                target=evidence_task.id,
                label="depends_on",
            )
            if dep_edge.id not in task_graph._edges:
                task_graph.add_edge(dep_edge)
            ev_edge = BaseTaskEdge(
                id=TaskGraph.stable_edge_id(evidence_task.id, parent_task_id, DependencyType.PRODUCES_EVIDENCE_FOR),
                dependency_type=DependencyType.PRODUCES_EVIDENCE_FOR,
                source=evidence_task.id,
                target=parent_task_id,
                label="produces_evidence_for",
            )
            if ev_edge.id not in task_graph._edges:
                task_graph.add_edge(ev_edge)

    @staticmethod
    def _dedupe_candidates(candidates: list[TaskCandidate]) -> list[TaskCandidate]:
        deduped: dict[str, TaskCandidate] = {}
        for candidate in candidates:
            key = TaskGraph.stable_task_id(
                source_action_id=candidate.source_action_id,
                task_type=candidate.task_type,
                input_bindings=dict(candidate.input_bindings),
            )
            deduped[key] = candidate
        return [deduped[key] for key in sorted(deduped)]

    @staticmethod
    def _derive_expected_outputs(candidate: TaskCandidate, task_id: str) -> list[GraphRef]:
        refs = list(candidate.target_refs) or [GraphRef(graph="ag", ref_id=candidate.source_action_id, ref_type="ActionNode")]
        return [
            GraphRef(
                graph="query",
                ref_id=f"task-output::{task_id}::{index}::{ref.ref_id}",
                ref_type="TaskOutput",
                label=ref.label,
            )
            for index, ref in enumerate(refs)
        ]

    @staticmethod
    def _add_bidirectional_edge(
        task_graph: TaskGraph,
        left_task_id: str,
        right_task_id: str,
        dependency_type: DependencyType,
    ) -> None:
        for source, target in ((left_task_id, right_task_id), (right_task_id, left_task_id)):
            edge = BaseTaskEdge(
                id=TaskGraph.stable_edge_id(source, target, dependency_type),
                dependency_type=dependency_type,
                source=source,
                target=target,
                label=dependency_type.value.lower(),
            )
            if edge.id not in task_graph._edges:
                task_graph.add_edge(edge)

    def _link_host_stage_dependencies(self, task_graph: TaskGraph, tasks: list[BaseTaskNode]) -> None:
        """Add same-host discovery -> validation -> lateral progression edges."""

        for producer in tasks:
            producer_hosts = {key for key in producer.resource_keys if key.startswith("host:")}
            if not producer_hosts:
                continue
            for consumer in tasks:
                if producer.id == consumer.id:
                    continue
                if not producer_hosts & {key for key in consumer.resource_keys if key.startswith("host:")}:
                    continue
                if not self._should_depend_by_stage(producer.task_type, consumer.task_type):
                    continue
                edge = BaseTaskEdge(
                    id=TaskGraph.stable_edge_id(producer.id, consumer.id, DependencyType.DEPENDS_ON),
                    dependency_type=DependencyType.DEPENDS_ON,
                    source=producer.id,
                    target=consumer.id,
                    label="depends_on",
                )
                if edge.id not in task_graph._edges:
                    task_graph.add_edge(edge)

    @staticmethod
    def _should_depend_by_stage(producer: TaskType, consumer: TaskType) -> bool:
        stage_rank = {
            TaskType.ASSET_CONFIRMATION: 0,
            TaskType.SERVICE_VALIDATION: 1,
            TaskType.REACHABILITY_VALIDATION: 2,
            TaskType.IDENTITY_CONTEXT_CONFIRMATION: 3,
            TaskType.PRIVILEGE_CONFIGURATION_VALIDATION: 4,
            TaskType.GOAL_CONDITION_VALIDATION: 5,
        }
        return stage_rank.get(producer, -1) >= 0 and stage_rank.get(producer, -1) < stage_rank.get(consumer, -1)

    @staticmethod
    def _stable_group_id(tasks: list[TaskNode], group_type: TaskGroupType, label: str) -> str:
        return TaskGraph.stable_edge_id(
            source=group_type.value,
            target="|".join(sorted(task.id for task in tasks)) + f"::{label}",
            dependency_type=DependencyType.DERIVED_FROM,
        ).replace("tg-edge::", "tg-group::", 1)

    @staticmethod
    def _stable_timestamp() -> datetime:
        """Return a deterministic timestamp for graph-building output."""

        return datetime(2000, 1, 1, tzinfo=timezone.utc)

    @classmethod
    def _normalize_graph_timestamps(cls, task_graph: TaskGraph) -> None:
        """Normalize node timestamps after status recomputation mutates them."""

        stable_time = cls._stable_timestamp()
        for node in task_graph._nodes.values():
            if hasattr(node, "created_at"):
                node.created_at = stable_time
            if hasattr(node, "updated_at"):
                node.updated_at = stable_time
