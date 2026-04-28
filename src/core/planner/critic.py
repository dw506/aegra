"""Critic for low-value, blocked or overused planning branches."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.ag import AGEdgeType, ActivationStatus, ActionNode, AttackGraph, ConstraintNode
from src.core.models.tg import (
    BaseTaskEdge,
    BaseTaskNode,
    OutcomeNode,
    ReplanFrontier,
    TaskGraph,
    TaskNode,
    TaskStatus,
)
from src.core.planner.planner import ActionChainCandidate
from src.core.planner.scorer import HeuristicScorer


class CriticContext(BaseModel):
    """Context used when critiquing branches and actions."""

    model_config = ConfigDict(extra="forbid")

    failure_counts: dict[str, int] = Field(default_factory=dict)
    low_value_threshold: float = 0.2
    failure_threshold: int = 2


class CriticFinding(BaseModel):
    """Structured finding from the critic."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    subject_id: str
    severity: str
    reason: str
    recommendation: str


class CriticResult(BaseModel):
    """Structured critique output."""

    model_config = ConfigDict(extra="forbid")

    findings: list[CriticFinding] = Field(default_factory=list)
    blocked_action_ids: list[str] = Field(default_factory=list)
    low_value_action_ids: list[str] = Field(default_factory=list)
    replan_hints: list[str] = Field(default_factory=list)


class TaskCriticContext(BaseModel):
    """Context used when critiquing a Task Graph."""

    model_config = ConfigDict(extra="forbid")

    low_value_threshold: float = 0.25
    failure_threshold: int = 2
    invalidated_ref_keys: set[str] = Field(default_factory=set)


class TaskCriticResult(BaseModel):
    """Structured TG critique output."""

    model_config = ConfigDict(extra="forbid")

    findings: list[CriticFinding] = Field(default_factory=list)
    duplicate_task_ids: list[str] = Field(default_factory=list)
    blocked_task_ids: list[str] = Field(default_factory=list)
    failed_branch_task_ids: list[str] = Field(default_factory=list)
    invalidated_task_ids: list[str] = Field(default_factory=list)
    replan_frontiers: list[ReplanFrontier] = Field(default_factory=list)


class AttackGraphCritic:
    """Review candidate branches and suggest pruning or replanning."""

    def __init__(self, scorer: HeuristicScorer | None = None) -> None:
        self.scorer = scorer or HeuristicScorer()

    def critique(
        self,
        graph: AttackGraph,
        candidate_paths: list[ActionChainCandidate] | None = None,
        context: CriticContext | None = None,
    ) -> CriticResult:
        """Return structured pruning and replanning suggestions."""

        ctx = context or CriticContext()
        result = CriticResult()

        for action in graph.find_actions():
            if action.activation_status == ActivationStatus.BLOCKED:
                blockers = [
                    graph.get_node(edge.target)
                    for edge in graph.list_edges(AGEdgeType.BLOCKED_BY)
                    if edge.source == action.id
                ]
                labels = [blocker.label for blocker in blockers if isinstance(blocker, ConstraintNode)]
                result.findings.append(
                    CriticFinding(
                        kind="blocked_action",
                        subject_id=action.id,
                        severity="high",
                        reason=f"action is blocked by constraints: {', '.join(labels) or 'unknown'}",
                        recommendation="prune this action until the blocking constraint changes",
                    )
                )
                result.blocked_action_ids.append(action.id)

            action_score = self.scorer.score_action(action)
            if action_score < ctx.low_value_threshold:
                result.findings.append(
                    CriticFinding(
                        kind="low_value_action",
                        subject_id=action.id,
                        severity="medium",
                        reason=f"heuristic action score {action_score:.3f} is below threshold",
                        recommendation="deprioritize this branch behind higher-value actions",
                    )
                )
                result.low_value_action_ids.append(action.id)

            if ctx.failure_counts.get(action.id, 0) >= ctx.failure_threshold:
                result.findings.append(
                    CriticFinding(
                        kind="failure_saturated_branch",
                        subject_id=action.id,
                        severity="medium",
                        reason="recent failure history materially lowers branch quality",
                        recommendation="replan around this action or require fresh evidence before retrying",
                    )
                )

        for path in candidate_paths or []:
            if path.score < ctx.low_value_threshold:
                result.findings.append(
                    CriticFinding(
                        kind="low_value_path",
                        subject_id="->".join(path.action_ids),
                        severity="medium",
                        reason="path score is below the configured low-value threshold",
                        recommendation="prune this path in favor of higher-value chains",
                    )
                )

        if result.blocked_action_ids:
            result.replan_hints.append("prefer branches with no active BLOCKED_BY constraints")
        if result.low_value_action_ids:
            result.replan_hints.append("raise score thresholds or bias toward goal-proximal actions")
        if any(f.kind == "failure_saturated_branch" for f in result.findings):
            result.replan_hints.append("refresh KG evidence before revisiting failure-saturated branches")
        return result


class TaskGraphCritic:
    """Critique TG branches and expose local graph repair operations."""

    def __init__(self, scorer: HeuristicScorer | None = None) -> None:
        self.scorer = scorer or HeuristicScorer()

    def critique_task_graph(
        self,
        task_graph: TaskGraph,
        context: TaskCriticContext | None = None,
    ) -> TaskCriticResult:
        """Inspect a Task Graph for duplicate, blocked, failed and invalidated tasks."""

        ctx = context or TaskCriticContext()
        result = TaskCriticResult()
        duplicate_groups = self._duplicate_tasks(task_graph)

        for group in duplicate_groups:
            result.duplicate_task_ids.extend(task.id for task in group)
            result.findings.append(
                CriticFinding(
                    kind="duplicate_task",
                    subject_id=",".join(task.id for task in group),
                    severity="medium",
                    reason="multiple equivalent tasks share the same source action, type and bindings",
                    recommendation="supersede lower-priority duplicates and keep one canonical task",
                )
            )

        for task in self._task_nodes(task_graph):
            if self._is_permanently_blocked(task_graph, task):
                result.blocked_task_ids.append(task.id)
                result.findings.append(
                    CriticFinding(
                        kind="permanently_blocked_task",
                        subject_id=task.id,
                        severity="high",
                        reason="task is blocked by an approval gate or a terminal upstream dependency",
                        recommendation="cancel or supersede this task, then collect a local replanning frontier",
                    )
                )
                result.replan_frontiers.append(task_graph.collect_replan_frontier(task.id))

            if task.status == TaskStatus.FAILED and task.attempt_count >= min(task.max_attempts, ctx.failure_threshold):
                result.failed_branch_task_ids.append(task.id)
                result.findings.append(
                    CriticFinding(
                        kind="failure_saturated_task",
                        subject_id=task.id,
                        severity="medium",
                        reason="task has reached the failure threshold without remaining retry budget",
                        recommendation="replace the local subgraph or refresh supporting evidence before retrying",
                    )
                )
                result.replan_frontiers.append(task_graph.collect_replan_frontier(task.id))

            task_score = self._task_value_score(task)
            if task_score < ctx.low_value_threshold:
                result.findings.append(
                    CriticFinding(
                        kind="low_value_task",
                        subject_id=task.id,
                        severity="low",
                        reason=f"task value score {task_score:.3f} is below threshold",
                        recommendation="deprioritize or replace this task with a higher-value alternative",
                    )
                )

            task_ref_keys = self._task_ref_keys(task)
            if ctx.invalidated_ref_keys and task_ref_keys & ctx.invalidated_ref_keys:
                result.invalidated_task_ids.append(task.id)
                result.findings.append(
                    CriticFinding(
                        kind="invalidated_task",
                        subject_id=task.id,
                        severity="high",
                        reason="new evidence invalidates one or more task target references",
                        recommendation="supersede this task and collect a replanning frontier from the invalidated node",
                    )
                )
                result.replan_frontiers.append(task_graph.collect_replan_frontier(task.id))

        result.duplicate_task_ids = sorted(set(result.duplicate_task_ids))
        result.blocked_task_ids = sorted(set(result.blocked_task_ids))
        result.failed_branch_task_ids = sorted(set(result.failed_branch_task_ids))
        result.invalidated_task_ids = sorted(set(result.invalidated_task_ids))
        return result

    def mark_task_superseded(
        self,
        task_graph: TaskGraph,
        task_id: str,
        replacement_task_id: str | None = None,
    ) -> BaseTaskNode:
        """Mark one task as superseded in the TG."""

        return task_graph.mark_task_superseded(task_id, replacement_task_id=replacement_task_id)

    def cancel_task(self, task_graph: TaskGraph, task_id: str, reason: str) -> BaseTaskNode:
        """Cancel one task in the TG."""

        return task_graph.cancel_task(task_id, reason)

    def attach_outcome(
        self,
        task_graph: TaskGraph,
        task_id: str,
        outcome_node: OutcomeNode,
    ) -> OutcomeNode:
        """Attach a structured outcome node to a TG task."""

        return task_graph.attach_outcome(task_id, outcome_node)

    def replace_subgraph(
        self,
        task_graph: TaskGraph,
        failed_task_id: str,
        new_tasks: list[TaskNode],
        new_edges: list[BaseTaskEdge],
    ) -> list[str]:
        """Replace the local TG subgraph around one failed task."""

        return task_graph.replace_subgraph(failed_task_id, new_tasks, new_edges)

    def collect_replan_frontier(self, task_graph: TaskGraph, task_id: str) -> ReplanFrontier:
        """Collect a local replanning frontier for one TG task."""

        return task_graph.collect_replan_frontier(task_id)

    @staticmethod
    def _task_nodes(task_graph: TaskGraph) -> list[BaseTaskNode]:
        return sorted(
            (
                node
                for node in task_graph.list_nodes()
                if isinstance(node, BaseTaskNode)
            ),
            key=lambda item: item.id,
        )

    def _duplicate_tasks(self, task_graph: TaskGraph) -> list[list[BaseTaskNode]]:
        groups: dict[str, list[BaseTaskNode]] = {}
        for task in self._task_nodes(task_graph):
            signature = json.dumps(
                {
                    "source_action_id": task.source_action_id,
                    "task_type": task.task_type.value,
                    "input_bindings": task.input_bindings,
                },
                sort_keys=True,
                default=str,
            )
            groups.setdefault(signature, []).append(task)
        return [group for group in groups.values() if len(group) > 1]

    @staticmethod
    def _is_permanently_blocked(task_graph: TaskGraph, task: BaseTaskNode) -> bool:
        if task.status != TaskStatus.BLOCKED:
            return False
        if task.gate_ids:
            return True
        upstream = task_graph.predecessors(task.id)
        return any(dep.status in {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.SUPERSEDED} for dep in upstream)

    @staticmethod
    def _task_value_score(task: BaseTaskNode) -> float:
        return max(
            0.0,
            task.goal_relevance
            - (task.estimated_cost * 0.2)
            - (task.estimated_risk * 0.4)
            - (task.estimated_noise * 0.2),
        )

    @staticmethod
    def _task_ref_keys(task: BaseTaskNode) -> set[str]:
        """Return multiple stable key variants for task refs.

        Older callers sometimes include `ref_type` in invalidation keys while
        `GraphRef.key()` only returns `graph:ref_id`. The critic accepts both
        forms to remain compatible with existing callers and tests.
        """

        keys: set[str] = set()
        for ref in [*task.target_refs, *task.source_refs]:
            keys.add(ref.key())
            if ref.ref_type:
                keys.add(f"{ref.graph}:{ref.ref_type}:{ref.ref_id}")
        return keys
