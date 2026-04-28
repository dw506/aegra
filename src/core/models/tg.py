"""Task Graph models and container implementation."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.models.ag import GraphRef, stable_node_id


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


class TaskType(str, Enum):
    """Authorized task categories for the validation platform."""

    ASSET_CONFIRMATION = "ASSET_CONFIRMATION"
    SERVICE_VALIDATION = "SERVICE_VALIDATION"
    REACHABILITY_VALIDATION = "REACHABILITY_VALIDATION"
    IDENTITY_CONTEXT_CONFIRMATION = "IDENTITY_CONTEXT_CONFIRMATION"
    PRIVILEGE_CONFIGURATION_VALIDATION = "PRIVILEGE_CONFIGURATION_VALIDATION"
    GOAL_CONDITION_VALIDATION = "GOAL_CONDITION_VALIDATION"
    EVIDENCE_COLLECTION_AND_ARCHIVAL = "EVIDENCE_COLLECTION_AND_ARCHIVAL"


class TaskStatus(str, Enum):
    """Lifecycle status of a task instance."""

    DRAFT = "draft"
    PENDING = "draft"
    QUEUED = "queued"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class DependencyType(str, Enum):
    """Edge types inside the Task Graph."""

    DEPENDS_ON = "DEPENDS_ON"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    ALTERNATIVE_TO = "ALTERNATIVE_TO"
    DERIVED_FROM = "DERIVED_FROM"
    PRODUCES_EVIDENCE_FOR = "PRODUCES_EVIDENCE_FOR"
    RECOVERS_FROM = "RECOVERS_FROM"
    HAS_OUTCOME = "HAS_OUTCOME"


class TaskGroupType(str, Enum):
    """Grouping types for one execution wave."""

    PHASE = "PHASE"
    STAGE = "STAGE"
    BATCH = "BATCH"
    ALTERNATIVE_SET = "ALTERNATIVE_SET"


class DecisionStatus(str, Enum):
    """Lifecycle status of a decision node."""

    PENDING = "pending"
    DECIDED = "decided"
    SUPERSEDED = "superseded"


class RetryPolicy(BaseModel):
    """Retry policy attached to a task instance."""

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=1, ge=1)
    backoff_seconds: int = Field(default=0, ge=0)
    retryable_statuses: set[TaskStatus] = Field(default_factory=lambda: {TaskStatus.FAILED})


class TaskBinding(BaseModel):
    """Structured binding carried by a task instance."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    value: str | int | float | bool | None = None
    ref: GraphRef | None = None


class TaskCheckpoint(BaseModel):
    """Checkpoint summary used for re-planning and audit trails."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    status: TaskStatus
    note: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class BaseTaskNode(BaseModel):
    """A schedulable task instance derived from one AG action."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    kind: str = "task"
    task_type: TaskType
    status: TaskStatus = TaskStatus.DRAFT
    source_action_id: str | None = None
    input_bindings: dict[str, Any] = Field(default_factory=dict)
    target_refs: list[GraphRef] = Field(default_factory=list)
    precondition_refs: list[GraphRef] = Field(default_factory=list)
    expected_output_refs: list[GraphRef] = Field(default_factory=list)
    source_refs: list[GraphRef] = Field(default_factory=list)
    evidence_output_refs: list[GraphRef] = Field(default_factory=list)
    estimated_cost: float = Field(default=0.0, ge=0.0)
    estimated_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    estimated_noise: float = Field(default=0.0, ge=0.0, le=1.0)
    goal_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    priority: int = Field(default=50, ge=0, le=100)
    resource_keys: set[str] = Field(default_factory=set)
    parallelizable: bool = False
    approval_required: bool = False
    assigned_agent: str | None = None
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=1, ge=1)
    deadline: datetime | None = None
    gate_ids: set[str] = Field(default_factory=set)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    attempts: int = Field(default=0, ge=0)
    reason: str | None = None
    tags: set[str] = Field(default_factory=set)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_time_window(self) -> "BaseTaskNode":
        """Ensure timestamps remain monotonic."""

        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        if self.attempt_count > self.max_attempts:
            raise ValueError("attempt_count must be less than or equal to max_attempts")
        return self


class BaseTaskEdge(BaseModel):
    """Relationship between task instances."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)
    dependency_type: DependencyType
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    label: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)


class TaskNode(BaseTaskNode):
    """Concrete task instance node."""

    kind: str = "task"


class TaskGroupNode(BaseModel):
    """Group node that clusters same-stage tasks."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)
    kind: str = "group"
    group_type: TaskGroupType
    label: str = Field(min_length=1)
    task_ids: list[str] = Field(default_factory=list)
    tags: set[str] = Field(default_factory=set)
    created_at: datetime = Field(default_factory=utc_now)


class DecisionNode(BaseModel):
    """Decision node representing a branch or human approval choice."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)
    kind: str = "decision"
    label: str = Field(min_length=1)
    decision_type: str = Field(min_length=1)
    option_task_ids: list[str] = Field(default_factory=list)
    selected_task_ids: list[str] = Field(default_factory=list)
    status: DecisionStatus = DecisionStatus.PENDING
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class CheckpointNode(BaseModel):
    """Recovery anchor node used by checkpoint storage integration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)
    kind: str = "checkpoint"
    label: str = Field(min_length=1)
    anchor_refs: list[GraphRef] = Field(default_factory=list)
    source_task_id: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class OutcomeNode(BaseModel):
    """Structured outcome attached to one task for critic and replan use."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)
    kind: str = "outcome"
    label: str = Field(min_length=1)
    outcome_type: str = Field(min_length=1)
    source_task_id: str | None = None
    invalidated_refs: list[GraphRef] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ReplanFrontier(BaseModel):
    """Minimal frontier snapshot used for local replanning."""

    model_config = ConfigDict(extra="forbid")

    root_task_id: str
    upstream_task_ids: list[str] = Field(default_factory=list)
    downstream_task_ids: list[str] = Field(default_factory=list)
    conflicting_task_ids: list[str] = Field(default_factory=list)
    blocked_task_ids: list[str] = Field(default_factory=list)


TaskGraphNode = TaskNode | TaskGroupNode | DecisionNode | CheckpointNode | OutcomeNode


class TaskGraph:
    """In-memory Task Graph for one execution wave."""

    TERMINAL_STATES = {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.SKIPPED,
        TaskStatus.CANCELLED,
        TaskStatus.SUPERSEDED,
    }

    def __init__(self) -> None:
        self._nodes: dict[str, TaskGraphNode] = {}
        self._edges: dict[str, BaseTaskEdge] = {}
        self._task_type_index: dict[str, set[str]] = defaultdict(set)
        self._status_index: dict[str, set[str]] = defaultdict(set)
        self._outgoing_index: dict[str, set[str]] = defaultdict(set)
        self._incoming_index: dict[str, set[str]] = defaultdict(set)
        self._resource_index: dict[str, set[str]] = defaultdict(set)
        self._source_action_index: dict[str, set[str]] = defaultdict(set)
        self._version: int = 0
        self._source_ag_version: int | None = None
        self._frontier_version: str | None = None
        self._metadata: dict[str, Any] = {}

    @property
    def version(self) -> int:
        """Return the TG logical version."""

        return self._version

    @property
    def source_ag_version(self) -> int | None:
        """Return the AG version this TG snapshot was built from."""

        return self._source_ag_version

    @property
    def frontier_version(self) -> str | None:
        """Return the local frontier / rebuild token associated with the TG snapshot."""

        return self._frontier_version

    def set_metadata(
        self,
        *,
        source_ag_version: int | None,
        frontier_version: str | None,
        metadata: dict[str, Any] | None = None,
        version: int | None = None,
    ) -> None:
        """Attach graph lineage metadata to the TG snapshot."""

        self._source_ag_version = source_ag_version
        self._frontier_version = frontier_version
        if metadata is not None:
            self._metadata = dict(metadata)
        if version is not None:
            self._version = max(version, 0)

    def add_node(self, node: TaskGraphNode) -> TaskGraphNode:
        """Add a task node and index it."""

        if node.id in self._nodes:
            raise ValueError(f"task '{node.id}' already exists")
        self._nodes[node.id] = node
        if isinstance(node, BaseTaskNode):
            self._task_type_index[node.task_type.value].add(node.id)
            self._status_index[node.status.value].add(node.id)
            if node.source_action_id:
                self._source_action_index[node.source_action_id].add(node.id)
            for key in node.resource_keys:
                self._resource_index[key].add(node.id)
        self._version += 1
        return node

    def add_edge(self, edge: BaseTaskEdge) -> BaseTaskEdge:
        """Add a relationship between existing task nodes."""

        if edge.id in self._edges:
            raise ValueError(f"edge '{edge.id}' already exists")
        if edge.source not in self._nodes:
            raise ValueError(f"edge source '{edge.source}' does not exist")
        if edge.target not in self._nodes:
            raise ValueError(f"edge target '{edge.target}' does not exist")
        self._edges[edge.id] = edge
        self._outgoing_index[edge.source].add(edge.id)
        self._incoming_index[edge.target].add(edge.id)
        self._version += 1
        return edge

    def get_node(self, node_id: str) -> TaskGraphNode:
        """Return one task node by ID."""

        return self._nodes[node_id]

    def get_edge(self, edge_id: str) -> BaseTaskEdge:
        """Return one edge by ID."""

        return self._edges[edge_id]

    def remove_node(self, node_id: str) -> TaskGraphNode:
        """Remove a task node and all incident edges."""

        node = self._nodes[node_id]
        incident = set(self._incoming_index.get(node_id, set()))
        incident.update(self._outgoing_index.get(node_id, set()))
        for edge_id in list(incident):
            self.remove_edge(edge_id)
        del self._nodes[node_id]
        if isinstance(node, BaseTaskNode):
            self._task_type_index[node.task_type.value].discard(node_id)
            self._status_index[node.status.value].discard(node_id)
            if node.source_action_id:
                self._source_action_index[node.source_action_id].discard(node_id)
            for key in node.resource_keys:
                self._resource_index[key].discard(node_id)
        self._version += 1
        return node

    def remove_edge(self, edge_id: str) -> BaseTaskEdge:
        """Remove one edge by ID."""

        edge = self._edges[edge_id]
        self._outgoing_index[edge.source].discard(edge_id)
        self._incoming_index[edge.target].discard(edge_id)
        del self._edges[edge_id]
        self._version += 1
        return edge

    def list_nodes(
        self,
        task_type: TaskType | str | None = None,
        status: TaskStatus | str | None = None,
    ) -> list[TaskGraphNode]:
        """List tasks optionally filtered by type and status."""

        if task_type is None:
            node_ids = set(self._nodes)
        else:
            key = task_type.value if isinstance(task_type, Enum) else task_type
            node_ids = set(self._task_type_index.get(key, set()))
        if status is not None:
            status_key = status.value if isinstance(status, Enum) else status
            node_ids &= set(self._status_index.get(status_key, set()))
        return sorted((self._nodes[node_id] for node_id in node_ids), key=lambda item: item.id)

    def list_edges(self, dependency_type: DependencyType | str | None = None) -> list[BaseTaskEdge]:
        """List task edges optionally filtered by dependency type."""

        if dependency_type is None:
            edges = self._edges.values()
        else:
            key = dependency_type.value if isinstance(dependency_type, Enum) else dependency_type
            edges = (edge for edge in self._edges.values() if edge.dependency_type.value == key)
        return sorted(edges, key=lambda item: item.id)

    def predecessors(
        self,
        node_id: str,
        dependency_type: DependencyType | str | None = None,
    ) -> list[BaseTaskNode]:
        """Return predecessor tasks for one task."""

        edge_ids = set(self._incoming_index.get(node_id, set()))
        if dependency_type is not None:
            key = dependency_type.value if isinstance(dependency_type, Enum) else dependency_type
            edge_ids = {edge_id for edge_id in edge_ids if self._edges[edge_id].dependency_type.value == key}
        return sorted(
            (
                self._nodes[self._edges[edge_id].source]
                for edge_id in edge_ids
                if isinstance(self._nodes[self._edges[edge_id].source], BaseTaskNode)
            ),
            key=lambda item: item.id,
        )

    def successors(
        self,
        node_id: str,
        dependency_type: DependencyType | str | None = None,
    ) -> list[BaseTaskNode]:
        """Return successor tasks for one task."""

        edge_ids = set(self._outgoing_index.get(node_id, set()))
        if dependency_type is not None:
            key = dependency_type.value if isinstance(dependency_type, Enum) else dependency_type
            edge_ids = {edge_id for edge_id in edge_ids if self._edges[edge_id].dependency_type.value == key}
        return sorted(
            (
                self._nodes[self._edges[edge_id].target]
                for edge_id in edge_ids
                if isinstance(self._nodes[self._edges[edge_id].target], BaseTaskNode)
            ),
            key=lambda item: item.id,
        )

    def tasks_for_action(self, action_id: str) -> list[BaseTaskNode]:
        """Return all task instances derived from one AG action."""

        return sorted(
            (
                self._nodes[node_id]
                for node_id in self._source_action_index.get(action_id, set())
                if isinstance(self._nodes[node_id], BaseTaskNode)
            ),
            key=lambda item: item.id,
        )

    def tasks_using_resource(self, resource_key: str) -> list[BaseTaskNode]:
        """Return all tasks that claim one resource key."""

        return sorted(
            (
                self._nodes[node_id]
                for node_id in self._resource_index.get(resource_key, set())
                if isinstance(self._nodes[node_id], BaseTaskNode)
            ),
            key=lambda item: item.id,
        )

    def mark_task_status(self, task_id: str, status: TaskStatus, reason: str | None = None) -> BaseTaskNode:
        """Update a task status and refresh its indexes."""

        task = self._nodes[task_id]
        self._status_index[task.status.value].discard(task.id)
        task.status = status
        task.reason = reason
        task.updated_at = utc_now()
        self._status_index[task.status.value].add(task.id)
        self.refresh_blocked_states()
        return task

    def ready_tasks(self) -> list[BaseTaskNode]:
        """Return tasks that have all dependencies satisfied and no active conflicts."""

        self.refresh_blocked_states()
        return [task for task in self.list_nodes(status=TaskStatus.READY)]

    def blocked_tasks(self) -> list[BaseTaskNode]:
        """Return blocked task instances."""

        self.refresh_blocked_states()
        return self.list_nodes(status=TaskStatus.BLOCKED)

    def find_schedulable_tasks(self) -> list[BaseTaskNode]:
        """Return tasks that scheduler may queue now."""

        self.refresh_blocked_states()
        return self.list_nodes(status=TaskStatus.READY)

    def find_conflicting_tasks(self, task_id: str) -> list[BaseTaskNode]:
        """Return tasks that conflict with the given task."""

        return sorted(
            {
                task.id: task
                for task in (
                    self.predecessors(task_id, DependencyType.CONFLICTS_WITH)
                    + self.successors(task_id, DependencyType.CONFLICTS_WITH)
                    + self.predecessors(task_id, DependencyType.ALTERNATIVE_TO)
                    + self.successors(task_id, DependencyType.ALTERNATIVE_TO)
                )
            }.values(),
            key=lambda item: item.id,
        )

    def find_tasks_blocked_by_gate(self, gate_id: str) -> list[BaseTaskNode]:
        """Return tasks blocked by the given gate identifier."""

        return sorted(
            (
                node
                for node in self._nodes.values()
                if isinstance(node, BaseTaskNode)
                and gate_id in node.gate_ids
                and node.status == TaskStatus.BLOCKED
            ),
            key=lambda item: item.id,
        )

    def find_tasks_requiring_resource(self, resource_key: str) -> list[BaseTaskNode]:
        """Return tasks requiring a given resource key."""

        return self.tasks_using_resource(resource_key)

    def find_retryable_tasks(self) -> list[BaseTaskNode]:
        """Return failed tasks that still have retry budget."""

        return sorted(
            (
                node
                for node in self._nodes.values()
                if isinstance(node, BaseTaskNode)
                and node.status == TaskStatus.FAILED
                and node.attempt_count < node.max_attempts
            ),
            key=lambda item: item.id,
        )

    def can_transition(self, task_id: str, new_status: TaskStatus) -> bool:
        """Return True when a task may transition into the requested status."""

        task = self.get_node(task_id)
        if not isinstance(task, BaseTaskNode):
            return False
        allowed = {
            TaskStatus.DRAFT: {TaskStatus.BLOCKED, TaskStatus.READY},
            TaskStatus.READY: {TaskStatus.QUEUED, TaskStatus.CANCELLED},
            TaskStatus.QUEUED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
            TaskStatus.RUNNING: {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED},
            TaskStatus.FAILED: {TaskStatus.READY, TaskStatus.SUPERSEDED},
            TaskStatus.BLOCKED: {TaskStatus.SUPERSEDED},
        }
        if task.status == TaskStatus.FAILED and new_status == TaskStatus.READY:
            return task.attempt_count < task.max_attempts
        return new_status in allowed.get(task.status, set())

    def transition_task(
        self,
        task_id: str,
        new_status: TaskStatus,
        reason: str | None = None,
    ) -> BaseTaskNode:
        """Transition a task if it satisfies the minimal TG state machine."""

        task = self.get_node(task_id)
        if not isinstance(task, BaseTaskNode):
            raise ValueError(f"node '{task_id}' is not a task node")
        if not self.can_transition(task_id, new_status):
            raise ValueError(f"invalid transition: {task.status.value} -> {new_status.value}")
        if task.status == TaskStatus.RUNNING and new_status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED}:
            task.attempt_count += 1
            task.attempts = task.attempt_count
        self._set_status(task, new_status, reason)
        self.refresh_blocked_states()
        return task

    def create_checkpoint_node(
        self,
        task_id: str,
        anchor_refs: list[GraphRef],
        label: str | None = None,
    ) -> CheckpointNode:
        """Create and attach a lightweight checkpoint node for a task."""

        checkpoint = CheckpointNode(
            id=stable_node_id(
                "tg-checkpoint",
                {"task_id": task_id, "anchor_refs": [ref.key() for ref in anchor_refs]},
            ),
            label=label or f"Checkpoint for {task_id}",
            anchor_refs=anchor_refs,
            source_task_id=task_id,
        )
        if checkpoint.id not in self._nodes:
            self.add_node(checkpoint)
        recovery_edge = BaseTaskEdge(
            id=TaskGraph.stable_edge_id(task_id, checkpoint.id, DependencyType.RECOVERS_FROM),
            dependency_type=DependencyType.RECOVERS_FROM,
            source=task_id,
            target=checkpoint.id,
            label="recovers_from",
        )
        if recovery_edge.id not in self._edges:
            self.add_edge(recovery_edge)
        return checkpoint

    def link_recovery_anchor(self, task_id: str, anchor_node_id: str) -> BaseTaskEdge:
        """Link a task to a checkpoint or failed task as a recovery anchor."""

        if anchor_node_id not in self._nodes:
            raise ValueError(f"recovery anchor '{anchor_node_id}' does not exist")
        edge = BaseTaskEdge(
            id=TaskGraph.stable_edge_id(task_id, anchor_node_id, DependencyType.RECOVERS_FROM),
            dependency_type=DependencyType.RECOVERS_FROM,
            source=task_id,
            target=anchor_node_id,
            label="recovers_from",
        )
        if edge.id not in self._edges:
            self.add_edge(edge)
        return edge

    def mark_task_superseded(
        self,
        task_id: str,
        replacement_task_id: str | None = None,
    ) -> BaseTaskNode:
        """Mark a task as superseded and optionally link its replacement."""

        task = self.get_node(task_id)
        if not isinstance(task, BaseTaskNode):
            raise ValueError(f"node '{task_id}' is not a task node")
        self._set_status(task, TaskStatus.SUPERSEDED, "superseded by critic or replanning")
        if replacement_task_id is not None:
            self.link_recovery_anchor(replacement_task_id, task_id)
        self.refresh_blocked_states()
        return task

    def cancel_task(self, task_id: str, reason: str) -> BaseTaskNode:
        """Cancel one task when the TG state machine allows it."""

        return self.transition_task(task_id, TaskStatus.CANCELLED, reason=reason)

    def attach_outcome(self, task_id: str, outcome_node: OutcomeNode) -> OutcomeNode:
        """Attach a structured outcome node to one task."""

        task = self.get_node(task_id)
        if not isinstance(task, BaseTaskNode):
            raise ValueError(f"node '{task_id}' is not a task node")
        if outcome_node.source_task_id is None:
            outcome_node.source_task_id = task_id
        elif outcome_node.source_task_id != task_id:
            raise ValueError("outcome_node.source_task_id must match task_id")
        if outcome_node.id not in self._nodes:
            self.add_node(outcome_node)
        edge = BaseTaskEdge(
            id=TaskGraph.stable_edge_id(task_id, outcome_node.id, DependencyType.HAS_OUTCOME),
            dependency_type=DependencyType.HAS_OUTCOME,
            source=task_id,
            target=outcome_node.id,
            label="has_outcome",
        )
        if edge.id not in self._edges:
            self.add_edge(edge)
        return outcome_node

    def replace_subgraph(
        self,
        failed_task_id: str,
        new_tasks: list[TaskNode],
        new_edges: list[BaseTaskEdge],
    ) -> list[str]:
        """Replace the local subgraph around a failed task without full rebuild."""

        failed_task = self.get_node(failed_task_id)
        if not isinstance(failed_task, BaseTaskNode):
            raise ValueError(f"node '{failed_task_id}' is not a task node")

        upstream_ids = [
            edge.source
            for edge in self.list_edges(DependencyType.DEPENDS_ON)
            if edge.target == failed_task_id
        ]
        downstream_ids = [
            edge.target
            for edge in self.list_edges(DependencyType.DEPENDS_ON)
            if edge.source == failed_task_id
        ]

        created_task_ids: list[str] = []
        new_task_ids = {task.id for task in new_tasks}
        for task in new_tasks:
            if task.id not in self._nodes:
                self.add_node(task)
                created_task_ids.append(task.id)

        for edge in new_edges:
            if edge.id not in self._edges:
                self.add_edge(edge)

        new_dep_edges = [
            edge
            for edge in self._edges.values()
            if edge.dependency_type == DependencyType.DEPENDS_ON
            and edge.source in new_task_ids
            and edge.target in new_task_ids
        ]
        entry_ids = new_task_ids - {edge.target for edge in new_dep_edges}
        exit_ids = new_task_ids - {edge.source for edge in new_dep_edges}
        if not entry_ids:
            entry_ids = set(new_task_ids)
        if not exit_ids:
            exit_ids = set(new_task_ids)

        for edge in list(self._edges.values()):
            if edge.dependency_type != DependencyType.DEPENDS_ON:
                continue
            if edge.source == failed_task_id or edge.target == failed_task_id:
                self.remove_edge(edge.id)

        for upstream_id in upstream_ids:
            for entry_id in sorted(entry_ids):
                edge = BaseTaskEdge(
                    id=TaskGraph.stable_edge_id(upstream_id, entry_id, DependencyType.DEPENDS_ON),
                    dependency_type=DependencyType.DEPENDS_ON,
                    source=upstream_id,
                    target=entry_id,
                    label="depends_on",
                )
                if edge.id not in self._edges:
                    self.add_edge(edge)

        for exit_id in sorted(exit_ids):
            for downstream_id in downstream_ids:
                edge = BaseTaskEdge(
                    id=TaskGraph.stable_edge_id(exit_id, downstream_id, DependencyType.DEPENDS_ON),
                    dependency_type=DependencyType.DEPENDS_ON,
                    source=exit_id,
                    target=downstream_id,
                    label="depends_on",
                )
                if edge.id not in self._edges:
                    self.add_edge(edge)

        replacement_id = sorted(entry_ids)[0] if entry_ids else None
        self.mark_task_superseded(failed_task_id, replacement_task_id=replacement_id)
        return sorted(created_task_ids)

    def collect_replan_frontier(self, task_id: str) -> ReplanFrontier:
        """Collect the local frontier affected by one task for replanning."""

        task = self.get_node(task_id)
        if not isinstance(task, BaseTaskNode):
            raise ValueError(f"node '{task_id}' is not a task node")
        upstream = [node.id for node in self.predecessors(task_id, DependencyType.DEPENDS_ON)]
        downstream = [node.id for node in self.successors(task_id, DependencyType.DEPENDS_ON)]
        conflicting = [node.id for node in self.find_conflicting_tasks(task_id)]
        blocked = sorted(
            node.id
            for node in self._nodes.values()
            if isinstance(node, BaseTaskNode)
            and node.status == TaskStatus.BLOCKED
            and (
                task_id in {pred.id for pred in self.predecessors(node.id, DependencyType.DEPENDS_ON)}
                or node.id in conflicting
            )
        )
        return ReplanFrontier(
            root_task_id=task_id,
            upstream_task_ids=sorted(upstream),
            downstream_task_ids=sorted(downstream),
            conflicting_task_ids=sorted(conflicting),
            blocked_task_ids=blocked,
        )

    def refresh_blocked_states(self) -> None:
        """Recompute READY/BLOCKED/PENDING states from graph dependencies."""

        for task in self._nodes.values():
            if not isinstance(task, BaseTaskNode):
                continue
            if task.status in self.TERMINAL_STATES or task.status in {TaskStatus.RUNNING, TaskStatus.QUEUED}:
                continue
            if task.gate_ids:
                self._set_status(task, TaskStatus.BLOCKED, "waiting for scheduling gate")
                continue
            if self._has_active_conflict(task.id):
                self._set_status(task, TaskStatus.BLOCKED, "resource conflict or exclusive alternative")
                continue
            dependency_nodes = self.predecessors(task.id, DependencyType.DEPENDS_ON)
            if any(dep.status in {TaskStatus.FAILED, TaskStatus.CANCELLED} for dep in dependency_nodes):
                self._set_status(task, TaskStatus.BLOCKED, "upstream dependency failed")
                continue
            if all(dep.status == TaskStatus.SUCCEEDED for dep in dependency_nodes):
                self._set_status(task, TaskStatus.READY, task.reason)
            else:
                self._set_status(task, TaskStatus.DRAFT, task.reason)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the task graph to a JSON-safe dictionary."""

        return {
            "metadata": {
                "version": self._version,
                "source_ag_version": self._source_ag_version,
                "frontier_version": self._frontier_version,
                **self._metadata,
            },
            "nodes": [node.model_dump(mode="json") for node in self.list_nodes()],
            "edges": [edge.model_dump(mode="json") for edge in self.list_edges()],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskGraph":
        """Restore a task graph from serialized data."""

        graph = cls()
        for node_data in payload.get("nodes", []):
            node_kind = node_data.get("kind", "task")
            if node_kind == "task":
                graph.add_node(TaskNode.model_validate(node_data))
            elif node_kind == "group":
                graph.add_node(TaskGroupNode.model_validate(node_data))
            elif node_kind == "decision":
                graph.add_node(DecisionNode.model_validate(node_data))
            elif node_kind == "checkpoint":
                graph.add_node(CheckpointNode.model_validate(node_data))
            elif node_kind == "outcome":
                graph.add_node(OutcomeNode.model_validate(node_data))
            else:
                raise ValueError(f"unsupported task graph node kind: {node_kind}")
        for edge_data in payload.get("edges", []):
            graph.add_edge(BaseTaskEdge.model_validate(edge_data))
        graph.refresh_blocked_states()
        metadata = payload.get("metadata") or {}
        if isinstance(metadata, dict):
            graph.set_metadata(
                source_ag_version=metadata.get("source_ag_version"),
                frontier_version=metadata.get("frontier_version"),
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if key not in {"version", "source_ag_version", "frontier_version"}
                },
                version=metadata.get("version"),
            )
        return graph

    @staticmethod
    def stable_task_id(source_action_id: str, task_type: TaskType, input_bindings: dict[str, Any]) -> str:
        """Return a deterministic task ID."""

        return stable_node_id(
            "tg-task",
            {
                "source_action_id": source_action_id,
                "task_type": task_type.value,
                "input_bindings": input_bindings,
            },
        )

    @staticmethod
    def stable_edge_id(source: str, target: str, dependency_type: DependencyType) -> str:
        """Return a deterministic edge ID."""

        return stable_node_id(
            "tg-edge",
            {
                "source": source,
                "target": target,
                "dependency_type": dependency_type.value,
            },
        )

    def _has_active_conflict(self, task_id: str) -> bool:
        conflict_edges = {
            edge_id
            for edge_id in self._incoming_index.get(task_id, set()) | self._outgoing_index.get(task_id, set())
            if self._edges[edge_id].dependency_type in {DependencyType.CONFLICTS_WITH, DependencyType.ALTERNATIVE_TO}
        }
        for edge_id in conflict_edges:
            edge = self._edges[edge_id]
            other_id = edge.target if edge.source == task_id else edge.source
            other = self._nodes[other_id]
            if edge.dependency_type == DependencyType.CONFLICTS_WITH and other.status == TaskStatus.RUNNING:
                return True
            if edge.dependency_type == DependencyType.ALTERNATIVE_TO and other.status == TaskStatus.SUCCEEDED:
                return True
        return False

    def _set_status(self, task: BaseTaskNode, status: TaskStatus, reason: str | None) -> None:
        if task.status == status and reason == task.reason:
            return
        self._status_index[task.status.value].discard(task.id)
        task.status = status
        task.reason = reason
        task.updated_at = utc_now()
        self._status_index[task.status.value].add(task.id)
        self._version += 1
