"""Attack Graph planner built on deterministic best-first search."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.ag import (
    AGEdgeType,
    ActivationStatus,
    ActionNode,
    AttackGraph,
    GoalNode,
    StateNode,
    TruthStatus,
)
from src.core.graph.tg_builder import AttackGraphTaskBuilder, TaskCandidate, TaskGenerationRequest
from src.core.planner.scorer import HeuristicScorer


class GoalPlanningView(BaseModel):
    """Summarized planner view for one goal."""

    model_config = ConfigDict(extra="forbid")

    goal_id: str
    goal_label: str
    target_state_ids: list[str] = Field(default_factory=list)
    satisfied_state_ids: list[str] = Field(default_factory=list)
    unsatisfied_state_ids: list[str] = Field(default_factory=list)


class ActionCandidate(BaseModel):
    """Single action candidate with an explanation."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    score: float
    reason: str
    required_state_ids: list[str] = Field(default_factory=list)
    produced_state_ids: list[str] = Field(default_factory=list)


class ActionChainCandidate(BaseModel):
    """Candidate action chain returned by the planner."""

    model_config = ConfigDict(extra="forbid")

    action_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    goal_reached: bool = False
    reason: str


class PlanningResult(BaseModel):
    """Structured planner result."""

    model_config = ConfigDict(extra="forbid")

    goal: GoalPlanningView
    candidates: list[ActionCandidate] = Field(default_factory=list)
    chains: list[ActionChainCandidate] = Field(default_factory=list)
    task_candidates: list[TaskCandidate] = Field(default_factory=list)


@dataclass(slots=True)
class _SearchState:
    priority: float
    action_ids: list[str]
    available_states: set[str]


class AttackGraphPlanner:
    """Plan top-k actions or short chains for a goal node."""

    def __init__(self, scorer: HeuristicScorer | None = None) -> None:
        self.scorer = scorer or HeuristicScorer()

    def plan(
        self,
        graph: AttackGraph,
        goal_id: str,
        top_k: int = 3,
        max_depth: int = 3,
    ) -> PlanningResult:
        """Return top-k action candidates and chains for one goal."""

        goal = graph.get_node(goal_id)
        if not isinstance(goal, GoalNode):
            raise ValueError(f"node '{goal_id}' is not a GoalNode")

        goal_view = self._goal_view(graph, goal)
        active_states = {
            state.id
            for state in graph.find_states(active_only=True)
            if state.truth_status in {TruthStatus.ACTIVE, TruthStatus.VALIDATED}
        }
        candidate_actions = self._activatable_actions(graph, active_states)
        all_candidates = [
            ActionCandidate(
                action_id=action.id,
                score=self.scorer.score_action(action),
                reason=self._reason_for_action(graph, action, goal_view),
                required_state_ids=self._required_state_ids(graph, action.id),
                produced_state_ids=self._produced_state_ids(graph, action.id),
            )
            for action in candidate_actions
        ]
        all_candidates.sort(key=lambda item: item.score, reverse=True)
        chains = self._search_chains(graph, goal_view, active_states, top_k=top_k, max_depth=max_depth)
        candidates = all_candidates[:top_k]
        task_candidates = self.export_task_candidates(
            graph,
            action_ids=self._select_task_candidate_action_ids(candidates, chains, top_k=top_k),
        )
        return PlanningResult(
            goal=goal_view,
            candidates=candidates,
            chains=chains,
            task_candidates=task_candidates,
        )

    def export_task_candidates(
        self,
        graph: AttackGraph,
        action_ids: list[str],
        include_blocked: bool = False,
    ) -> list[TaskCandidate]:
        """Export planner-selected AG actions as TG TaskCandidate objects."""

        request = TaskGenerationRequest(action_ids=action_ids, include_blocked=include_blocked)
        return AttackGraphTaskBuilder().build_candidates_without_graph(graph, request)

    def build_task_generation_request(
        self,
        graph: AttackGraph,
        planning_result: PlanningResult,
        include_evidence_tasks: bool = True,
        include_blocked: bool = False,
    ) -> TaskGenerationRequest:
        """Build a TG generation request directly from one planning result."""

        candidates = planning_result.task_candidates or self.export_task_candidates(
            graph,
            action_ids=self._select_task_candidate_action_ids(
                planning_result.candidates,
                planning_result.chains,
                top_k=max(len(planning_result.candidates), 1),
            ),
            include_blocked=include_blocked,
        )
        return TaskGenerationRequest(
            candidates=candidates,
            include_blocked=include_blocked,
            include_evidence_tasks=include_evidence_tasks,
            group_label=f"Plan for {planning_result.goal.goal_label}",
        )

    def _goal_view(self, graph: AttackGraph, goal: GoalNode) -> GoalPlanningView:
        target_states = [
            edge.source
            for edge in graph.list_edges(AGEdgeType.ENABLES)
            if edge.target == goal.id and isinstance(graph.get_node(edge.source), StateNode)
        ]
        satisfied = [
            state_id
            for state_id in target_states
            if graph.get_node(state_id).truth_status in {TruthStatus.ACTIVE, TruthStatus.VALIDATED}
        ]
        unsatisfied = [state_id for state_id in target_states if state_id not in satisfied]
        return GoalPlanningView(
            goal_id=goal.id,
            goal_label=goal.label,
            target_state_ids=sorted(target_states),
            satisfied_state_ids=sorted(satisfied),
            unsatisfied_state_ids=sorted(unsatisfied),
        )

    def _activatable_actions(self, graph: AttackGraph, available_states: set[str]) -> list[ActionNode]:
        actions: list[ActionNode] = []
        for action in graph.find_actions():
            if action.activation_status == ActivationStatus.BLOCKED:
                continue
            required = set(self._required_state_ids(graph, action.id))
            if required.issubset(available_states):
                actions.append(action)
        return sorted(actions, key=self.scorer.score_action, reverse=True)

    def _search_chains(
        self,
        graph: AttackGraph,
        goal_view: GoalPlanningView,
        active_states: set[str],
        top_k: int,
        max_depth: int,
    ) -> list[ActionChainCandidate]:
        results: list[ActionChainCandidate] = []
        frontier: list[tuple[float, int, _SearchState]] = []
        sequence = count()
        heappush(
            frontier,
            (-0.0, next(sequence), _SearchState(priority=0.0, action_ids=[], available_states=set(active_states))),
        )
        seen: set[tuple[str, ...]] = set()

        while frontier and len(results) < top_k:
            _, _, current = heappop(frontier)
            state_key = tuple(sorted(current.action_ids))
            if state_key in seen:
                continue
            seen.add(state_key)

            goal_reached = any(state_id in current.available_states for state_id in goal_view.target_state_ids)
            if current.action_ids and goal_reached:
                results.append(
                    ActionChainCandidate(
                        action_ids=current.action_ids,
                        score=current.priority,
                        goal_reached=True,
                        reason="chain reaches at least one goal-enabling state",
                    )
                )
                continue
            if len(current.action_ids) >= max_depth:
                continue

            for action in self._activatable_actions(graph, current.available_states):
                if action.id in current.action_ids:
                    continue
                produced = set(self._produced_state_ids(graph, action.id))
                next_states = set(current.available_states) | produced
                next_actions = [*current.action_ids, action.id]
                path_actions = [graph.get_node(action_id) for action_id in next_actions]
                score = self.scorer.score_path(path_actions)
                heappush(
                    frontier,
                    (
                        -score,
                        next(sequence),
                        _SearchState(
                            priority=score,
                            action_ids=next_actions,
                            available_states=next_states,
                        ),
                    ),
                )

        if not results:
            for action in self._activatable_actions(graph, active_states)[:top_k]:
                results.append(
                    ActionChainCandidate(
                        action_ids=[action.id],
                        score=self.scorer.score_action(action),
                        goal_reached=False,
                        reason="best currently activatable action",
                    )
                )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _reason_for_action(
        self,
        graph: AttackGraph,
        action: ActionNode,
        goal_view: GoalPlanningView,
    ) -> str:
        produced = self._produced_state_ids(graph, action.id)
        if any(state_id in goal_view.unsatisfied_state_ids for state_id in produced):
            return "action directly produces an unsatisfied goal-enabling state"
        if action.activation_status == ActivationStatus.ACTIVATABLE:
            return "action is activatable with current projected states"
        return "action is relevant but depends on additional states"

    @staticmethod
    def _required_state_ids(graph: AttackGraph, action_id: str) -> list[str]:
        return sorted(
            edge.source
            for edge in graph.list_edges(AGEdgeType.REQUIRES)
            if edge.target == action_id
        )

    @staticmethod
    def _produced_state_ids(graph: AttackGraph, action_id: str) -> list[str]:
        return sorted(
            edge.target
            for edge in graph.list_edges(AGEdgeType.PRODUCES)
            if edge.source == action_id
        )

    @staticmethod
    def _select_task_candidate_action_ids(
        candidates: list[ActionCandidate],
        chains: list[ActionChainCandidate],
        top_k: int,
    ) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()
        for candidate in candidates[:top_k]:
            if candidate.action_id not in seen:
                selected.append(candidate.action_id)
                seen.add(candidate.action_id)
        for chain in chains[:top_k]:
            for action_id in chain.action_ids:
                if action_id not in seen:
                    selected.append(action_id)
                    seen.add(action_id)
        return selected
