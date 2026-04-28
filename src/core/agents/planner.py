"""Planner agent that searches AG candidates and emits planning decisions.

The Planner reads Attack Graph context plus goal references and produces only
structured decisions and task-candidate-style planning proposals. It does not
write KG, does not dispatch tools, and does not mutate TG directly.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_models import DecisionRecord, new_record_id
from src.core.agents.agent_protocol import (
    AgentInput,
    AgentKind,
    AgentOutput,
    BaseAgent,
    GraphRef,
    GraphScope,
    WritePermission,
)
from src.core.agents.llm_decision import (
    LLMDecision,
    LLMDecisionSource,
    LLMDecisionStatus,
    LLMDecisionValidationResult,
    LLMDecisionValidator,
)
from src.core.graph.tg_builder import TaskCandidate
from src.core.models.ag import (
    AGEdgeType,
    ActionNode,
    AttackGraph,
    GoalNode,
    GraphRef as AGGraphRef,
    StateNode,
)
from src.core.planner.planner import ActionChainCandidate, AttackGraphPlanner, PlanningResult
from src.core.planner.scorer import HeuristicScorer, ScoringContext


class PlanningContext(BaseModel):
    """Execution context and search bounds for one planner invocation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    top_k: int = Field(default=3, ge=1, le=20)
    max_depth: int = Field(default=3, ge=1, le=10)
    budget_summary: dict[str, Any] = Field(default_factory=dict)
    policy_context: dict[str, Any] = Field(default_factory=dict)
    runtime_summary: dict[str, Any] = Field(default_factory=dict)
    critic_hints: list[dict[str, Any]] = Field(default_factory=list)
    scorer_config: dict[str, Any] = Field(default_factory=dict)


class PlanningCandidate(BaseModel):
    """Planner-selected action chain plus TG-compatible task candidates."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    candidate_id: str = Field(default_factory=lambda: new_record_id("plan-candidate"))
    goal_ref: GraphRef
    action_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    rationale: str = Field(min_length=1)
    target_refs: list[GraphRef] = Field(default_factory=list)
    task_candidates: list[TaskCandidate] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlannerLLMAdvice(BaseModel):
    """可选的 LLM 规划建议。

    这里只允许补充“任务选择建议”和解释信息，不允许直接生成底层工具参数。
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    candidate_id: str = Field(min_length=1)
    score_delta: float = 0.0
    rationale_suffix: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    decision: LLMDecision | None = None
    validation: LLMDecisionValidationResult | None = None


class PlannerLLMRankAdjustment(BaseModel):
    """Bounded rank adjustment for one existing planning candidate."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    candidate_id: str = Field(min_length=1)
    score_delta: float = 0.0
    rationale_suffix: str | None = None
    risk_notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlannerLLMDecision(BaseModel):
    """Planner-level LLM strategy decision over existing candidates only."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    selected_candidate_ids: list[str] = Field(default_factory=list)
    rank_adjustments: list[PlannerLLMRankAdjustment] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    defer_reason: str | None = None
    requires_human_review: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    decision: LLMDecision | None = None
    validation: LLMDecisionValidationResult | None = None


class PlannerLLMAdvisor(Protocol):
    """Planner 可选 LLM 接口。

    中文注释：
    LLM 只能在这里做候选排序建议和解释增强，不能直接下发执行参数或攻击动作。
    """

    def advise(
        self,
        *,
        graph: AttackGraph,
        goal_ref: GraphRef,
        candidates: Sequence[PlanningCandidate],
        planning_context: PlanningContext,
    ) -> "PlannerLLMDecision | list[PlannerLLMAdvice]":
        """Return optional bounded planner strategy advice."""


class PlannerAgent(BaseAgent):
    """Read AG plus goals and emit ranked planning decisions only."""

    def __init__(
        self,
        name: str = "planner_agent",
        planner: AttackGraphPlanner | None = None,
        scorer: HeuristicScorer | None = None,
        llm_advisor: PlannerLLMAdvisor | None = None,
    ) -> None:
        self._scorer = scorer or HeuristicScorer()
        self._planner = planner or AttackGraphPlanner(scorer=self._scorer)
        self._llm_advisor = llm_advisor
        self._llm_decision_validator = LLMDecisionValidator()
        super().__init__(
            name=name,
            kind=AgentKind.PLANNER,
            write_permission=WritePermission(
                scopes=[],
                allow_structural_write=False,
                allow_state_write=False,
                allow_event_emit=True,
            ),
        )

    def validate_input(self, agent_input: AgentInput) -> None:
        """Require AG context, a graph snapshot, and at least one goal ref."""

        super().validate_input(agent_input)
        if not any(ref.graph == GraphScope.AG for ref in agent_input.graph_refs):
            raise ValueError("planner input requires at least one AG ref")
        self._resolve_attack_graph(agent_input)
        if not self._resolve_goal_refs(agent_input):
            raise ValueError("planner input requires at least one goal ref")

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Search AG candidate paths and emit top-k planner decisions."""

        graph = self._resolve_attack_graph(agent_input)
        planning_context = self._resolve_planning_context(agent_input)
        goal_refs = self._resolve_goal_refs(agent_input)

        decisions: list[dict[str, Any]] = []
        logs: list[str] = [
            "planner is read-only: no KG writes, no TG writes, no task dispatch",
            f"planning across {len(goal_refs)} goal ref(s)",
            f"top_k={planning_context.top_k} max_depth={planning_context.max_depth}",
        ]

        for goal_ref in goal_refs:
            goal_id = self._resolve_goal_node_id(graph=graph, goal_ref=goal_ref)
            planning_result = self._plan_with_existing_modules(
                graph=graph,
                goal_id=goal_id,
                planning_context=planning_context,
            )
            candidate_paths = self._collect_candidate_paths(
                graph=graph,
                planning_result=planning_result,
                goal_ref=goal_ref,
                planning_context=planning_context,
            )
            scored_candidates = [
                self._score_candidate(
                    candidate=candidate,
                    graph=graph,
                    planning_context=planning_context,
                )
                for candidate in candidate_paths
            ]
            scored_candidates = self._apply_llm_advice(
                graph=graph,
                goal_ref=goal_ref,
                candidates=scored_candidates,
                planning_context=planning_context,
                logs=logs,
            )
            selected = self._select_top_k(
                candidates=scored_candidates,
                top_k=planning_context.top_k,
            )
            emitted = self._emit_decisions(
                selected=selected,
                goal_ref=goal_ref,
                planning_context=planning_context,
            )
            decisions.extend([decision.to_agent_output_fragment() for decision in emitted])
            logs.append(
                f"goal {goal_ref.ref_id}: collected {len(candidate_paths)} candidate path(s), selected {len(selected)}"
            )

        return AgentOutput(decisions=decisions, logs=logs)

    def plan_with_existing_planner(
        self,
        *,
        graph: AttackGraph,
        goal_id: str,
        top_k: int,
        max_depth: int,
    ) -> PlanningResult:
        """Compatibility wrapper around the existing `AttackGraphPlanner`."""

        return self._planner.plan(graph=graph, goal_id=goal_id, top_k=top_k, max_depth=max_depth)

    def build_task_candidates(
        self,
        *,
        graph: AttackGraph,
        action_ids: list[str],
    ) -> list[TaskCandidate]:
        """Compatibility wrapper for AG -> TG candidate export without TG writes."""

        return self._planner.export_task_candidates(graph=graph, action_ids=action_ids)

    def _collect_candidate_paths(
        self,
        *,
        graph: AttackGraph,
        planning_result: PlanningResult,
        goal_ref: GraphRef,
        planning_context: PlanningContext,
    ) -> list[PlanningCandidate]:
        """Collect TG-compatible planning candidates from planner output."""

        candidates: list[PlanningCandidate] = []

        for chain in planning_result.chains:
            target_refs = self._target_refs_for_actions(graph=graph, action_ids=chain.action_ids)
            task_candidates = self.build_task_candidates(graph=graph, action_ids=chain.action_ids)
            candidates.append(
                PlanningCandidate(
                    goal_ref=goal_ref,
                    action_ids=list(chain.action_ids),
                    score=chain.score,
                    rationale=self._rationale_for_chain(chain, task_candidates),
                    target_refs=target_refs,
                    task_candidates=task_candidates,
                    metadata={
                        "goal_reached": chain.goal_reached,
                        "reason": chain.reason,
                        "source": "planning_result.chains",
                    },
                )
            )

        if not candidates:
            for task_candidate in planning_result.task_candidates[: planning_context.top_k]:
                candidates.append(
                    PlanningCandidate(
                        goal_ref=goal_ref,
                        action_ids=[task_candidate.source_action_id],
                        score=0.0,
                        rationale="fallback single-action candidate exported from planner task candidates",
                        target_refs=[self._to_agent_ref(ref) for ref in task_candidate.target_refs],
                        task_candidates=[task_candidate],
                        metadata={"source": "planning_result.task_candidates"},
                    )
                )

        return candidates

    def _score_candidate(
        self,
        *,
        candidate: PlanningCandidate,
        graph: AttackGraph,
        planning_context: PlanningContext,
    ) -> PlanningCandidate:
        """Score one candidate path with scorer and budget/policy adjustments."""

        actions = [
            graph.get_node(action_id)
            for action_id in candidate.action_ids
            if isinstance(graph.get_node(action_id), ActionNode)
        ]
        action_nodes = [action for action in actions if isinstance(action, ActionNode)]
        score = self._scorer.score_path(action_nodes) if action_nodes else candidate.score
        score += self._goal_alignment_bonus(candidate)
        score -= self._budget_penalty(candidate, planning_context)
        score -= self._critic_penalty(candidate, planning_context)
        score -= self._policy_penalty(candidate, planning_context)
        return candidate.model_copy(update={"score": round(score, 6)})

    @staticmethod
    def _select_top_k(
        *,
        candidates: Sequence[PlanningCandidate],
        top_k: int,
    ) -> list[PlanningCandidate]:
        """Select top-k scored planning candidates."""

        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]

    def _emit_decisions(
        self,
        *,
        selected: Sequence[PlanningCandidate],
        goal_ref: GraphRef,
        planning_context: PlanningContext,
    ) -> list[DecisionRecord]:
        """Emit planner decisions with rationale, score, and target refs."""

        decisions: list[DecisionRecord] = []
        for rank, candidate in enumerate(selected, start=1):
            decisions.append(
                DecisionRecord(
                    source_agent=self.name,
                    summary=f"Selected planning candidate #{rank} for goal {goal_ref.ref_id}",
                    confidence=max(0.0, min(1.0, candidate.score)),
                    refs=[goal_ref, *candidate.target_refs],
                    payload={
                        "planning_candidate": candidate.model_dump(mode="json"),
                        "planning_context": planning_context.model_dump(mode="json"),
                        "selection_rank": rank,
                    },
                    decision_type="plan_selection",
                    score=candidate.score,
                    target_refs=list(candidate.target_refs),
                    rationale=candidate.rationale,
                )
            )
        return decisions

    # 中文注释：
    # Planner 的 LLM 只允许补充排序建议与解释，不改变执行边界。
    def _apply_llm_advice(
        self,
        *,
        graph: AttackGraph,
        goal_ref: GraphRef,
        candidates: Sequence[PlanningCandidate],
        planning_context: PlanningContext,
        logs: list[str],
    ) -> list[PlanningCandidate]:
        if self._llm_advisor is None:
            return list(candidates)
        llm_output = self._llm_advisor.advise(
            graph=graph,
            goal_ref=goal_ref,
            candidates=candidates,
            planning_context=planning_context,
        )
        if not llm_output:
            return list(candidates)
        if isinstance(llm_output, PlannerLLMDecision):
            return self._apply_llm_strategy_decision(
                goal_ref=goal_ref,
                candidates=candidates,
                planning_context=planning_context,
                planner_decision=llm_output,
                logs=logs,
            )
        advice_list = llm_output
        allowed_candidate_ids = {candidate.candidate_id for candidate in candidates}
        advice_by_id: dict[str, PlannerLLMAdvice] = {}
        rejected_count = 0
        for advice in advice_list:
            decision = advice.decision or self._planner_decision_from_advice(advice)
            validation = self._llm_decision_validator.validate_planner_decision(
                decision,
                allowed_candidate_ids=allowed_candidate_ids,
                policy_context=planning_context.policy_context,
                runtime_summary=planning_context.runtime_summary,
            )
            if not validation.accepted:
                rejected_count += 1
                continue
            advice_by_id[advice.candidate_id] = advice.model_copy(
                update={
                    "decision": decision,
                    "validation": validation,
                    "score_delta": float(validation.sanitized_payload.get("score_delta", 0.0)),
                    "rationale_suffix": validation.sanitized_payload.get("rationale_suffix"),
                    "metadata": validation.sanitized_payload.get("metadata", {}),
                }
            )
        if not advice_by_id:
            logs.append(f"planner llm decision validation accepted=0 rejected={rejected_count}")
            return list(candidates)
        updated: list[PlanningCandidate] = []
        for candidate in candidates:
            advice = advice_by_id.get(candidate.candidate_id)
            if advice is None:
                updated.append(candidate)
                continue
            rationale = candidate.rationale
            if advice.rationale_suffix:
                rationale = f"{rationale}; {advice.rationale_suffix}"
            metadata = dict(candidate.metadata)
            metadata["llm_advice"] = advice.model_dump(mode="json")
            if advice.decision is not None:
                metadata["llm_decision"] = advice.decision.model_dump(mode="json")
            if advice.validation is not None:
                metadata["llm_decision_validation"] = advice.validation.model_dump(mode="json")
            updated.append(
                candidate.model_copy(
                    update={
                        "score": round(candidate.score + advice.score_delta, 6),
                        "rationale": rationale,
                        "metadata": metadata,
                    }
                )
            )
        logs.append(
            f"planner llm decision validation accepted={len(advice_by_id)} rejected={rejected_count}"
        )
        logs.append(f"planner llm advisor adjusted {len(advice_by_id)} candidate(s)")
        return updated

    def _apply_llm_strategy_decision(
        self,
        *,
        goal_ref: GraphRef,
        candidates: Sequence[PlanningCandidate],
        planning_context: PlanningContext,
        planner_decision: PlannerLLMDecision,
        logs: list[str],
    ) -> list[PlanningCandidate]:
        allowed_candidate_ids = {candidate.candidate_id for candidate in candidates}
        decision = planner_decision.decision or self._planner_strategy_envelope(
            goal_ref=goal_ref,
            planner_decision=planner_decision,
        )
        validation = self._llm_decision_validator.validate_planner_strategy_decision(
            decision,
            allowed_candidate_ids=allowed_candidate_ids,
            allowed_goal_ids={goal_ref.ref_id},
            selected_candidate_ids=planner_decision.selected_candidate_ids,
            rank_adjustments=[
                adjustment.model_dump(mode="json")
                for adjustment in planner_decision.rank_adjustments
            ],
            policy_context=planning_context.policy_context,
            runtime_summary=planning_context.runtime_summary,
        )
        if not validation.accepted:
            logs.append(f"planner llm strategy decision rejected: {validation.reason}")
            return list(candidates)

        sanitized = validation.sanitized_payload
        adjustments = {
            str(item["candidate_id"]): item
            for item in sanitized.get("rank_adjustments", [])
            if isinstance(item, dict) and item.get("candidate_id") is not None
        }
        selected_ids = list(sanitized.get("selected_candidate_ids", []))
        selected_rank = {candidate_id: index for index, candidate_id in enumerate(selected_ids)}
        updated: list[PlanningCandidate] = []
        for candidate in candidates:
            adjustment = adjustments.get(candidate.candidate_id)
            metadata = dict(candidate.metadata)
            metadata["llm_decision_summary"] = {
                "adopted": True,
                "reason": validation.reason,
                "selected_candidate_ids": selected_ids,
                "requires_human_review": bool(sanitized.get("requires_human_review", False)),
                "defer_reason": sanitized.get("defer_reason"),
            }
            metadata["llm_planner_decision"] = planner_decision.model_dump(mode="json")
            metadata["llm_decision"] = decision.model_dump(mode="json")
            metadata["llm_decision_validation"] = validation.model_dump(mode="json")
            score_delta = 0.0
            rationale = candidate.rationale
            if adjustment is not None:
                score_delta = float(adjustment.get("score_delta", 0.0))
                if adjustment.get("rationale_suffix"):
                    rationale = f"{rationale}; {adjustment['rationale_suffix']}"
            if candidate.candidate_id in selected_rank:
                # Small deterministic ordering nudge; task payloads remain unchanged.
                score_delta += max(0.0, 0.05 - (selected_rank[candidate.candidate_id] * 0.001))
            updated.append(
                candidate.model_copy(
                    update={
                        "score": round(candidate.score + score_delta, 6),
                        "rationale": rationale,
                        "metadata": metadata,
                    }
                )
            )
        logs.append(
            "planner llm strategy decision accepted "
            f"selected={len(selected_ids)} adjustments={len(adjustments)} "
            f"requires_human_review={bool(sanitized.get('requires_human_review', False))}"
        )
        return sorted(
            updated,
            key=lambda item: (
                selected_rank.get(item.candidate_id, len(selected_rank) + 1),
                -item.score,
            ),
        )

    @staticmethod
    def _planner_decision_from_advice(advice: PlannerLLMAdvice) -> LLMDecision:
        return LLMDecision(
            source=LLMDecisionSource.PLANNER,
            status=LLMDecisionStatus.ACCEPTED,
            decision_type="planner_candidate_advice",
            target_id=advice.candidate_id,
            target_kind="planning_candidate",
            score_delta=advice.score_delta,
            rationale_suffix=advice.rationale_suffix,
            metadata=dict(advice.metadata),
        )

    @staticmethod
    def _planner_strategy_envelope(
        *,
        goal_ref: GraphRef,
        planner_decision: PlannerLLMDecision,
    ) -> LLMDecision:
        return LLMDecision(
            source=LLMDecisionSource.PLANNER,
            status=LLMDecisionStatus.ACCEPTED,
            decision_type="planner_strategy_decision",
            target_id=goal_ref.ref_id,
            target_kind="planner_goal",
            risk_notes=list(planner_decision.risk_notes),
            metadata={
                **dict(planner_decision.metadata),
                "selected_candidate_ids": list(planner_decision.selected_candidate_ids),
                "rank_adjustments": [
                    adjustment.model_dump(mode="json")
                    for adjustment in planner_decision.rank_adjustments
                ],
                "defer_reason": planner_decision.defer_reason,
                "requires_human_review": planner_decision.requires_human_review,
            },
        )

    def _plan_with_existing_modules(
        self,
        *,
        graph: AttackGraph,
        goal_id: str,
        planning_context: PlanningContext,
    ) -> PlanningResult:
        """Bridge to the existing planner/scorer stack."""

        scorer_config = planning_context.scorer_config
        if scorer_config:
            scorer_context = ScoringContext.model_validate(scorer_config)
            self._scorer = HeuristicScorer(context=scorer_context)
            self._planner = AttackGraphPlanner(scorer=self._scorer)
        return self.plan_with_existing_planner(
            graph=graph,
            goal_id=goal_id,
            top_k=planning_context.top_k,
            max_depth=planning_context.max_depth,
        )

    def _resolve_attack_graph(self, agent_input: AgentInput) -> AttackGraph:
        """Resolve an AG snapshot from raw payload."""

        raw_graph = (
            agent_input.raw_payload.get("ag_graph")
            or agent_input.raw_payload.get("attack_graph")
            or agent_input.raw_payload.get("ag_snapshot")
        )
        if isinstance(raw_graph, AttackGraph):
            return raw_graph
        if isinstance(raw_graph, dict):
            return AttackGraph.from_dict(raw_graph)
        raise ValueError("planner input requires raw_payload.ag_graph or equivalent AttackGraph snapshot")

    def _resolve_planning_context(self, agent_input: AgentInput) -> PlanningContext:
        """Resolve and normalize planner configuration inputs."""

        payload = {
            "budget_summary": self._coerce_mapping(agent_input.raw_payload.get("budget")),
            "policy_context": self._coerce_mapping(agent_input.raw_payload.get("policy_context")),
            "runtime_summary": self._coerce_mapping(agent_input.raw_payload.get("runtime_summary")),
            "critic_hints": self._coerce_list_of_mappings(agent_input.raw_payload.get("critic_hints")),
            **self._coerce_mapping(agent_input.raw_payload.get("planning_context")),
        }
        return PlanningContext.model_validate(payload)

    def _resolve_goal_refs(self, agent_input: AgentInput) -> list[GraphRef]:
        """Resolve AG/KG goal refs from the invocation."""

        refs: list[GraphRef] = []
        seen: set[tuple[str, str, str | None]] = set()
        raw_goal_refs = agent_input.raw_payload.get("goal_refs")
        if isinstance(raw_goal_refs, list):
            for raw_ref in raw_goal_refs:
                ref = raw_ref if isinstance(raw_ref, GraphRef) else GraphRef.model_validate(raw_ref)
                key = (ref.graph.value, ref.ref_id, ref.ref_type)
                if key not in seen:
                    seen.add(key)
                    refs.append(ref)

        for ref in agent_input.graph_refs:
            ref_type = (ref.ref_type or "").lower()
            if ref_type in {"goal", "goalnode"}:
                key = (ref.graph.value, ref.ref_id, ref.ref_type)
                if key not in seen:
                    seen.add(key)
                    refs.append(ref)
        return refs

    def _resolve_goal_node_id(self, *, graph: AttackGraph, goal_ref: GraphRef) -> str:
        """Resolve a graph goal node ID from AG or KG-oriented goal refs."""

        if goal_ref.graph == GraphScope.AG:
            node = graph.get_node(goal_ref.ref_id)
            if not isinstance(node, GoalNode):
                raise ValueError(f"AG ref '{goal_ref.ref_id}' is not a GoalNode")
            return node.id

        if goal_ref.graph == GraphScope.KG:
            for node in graph.get_goal_nodes():
                if any(ref.ref_id == goal_ref.ref_id for ref in node.scope_refs):
                    return node.id
        raise ValueError(f"could not resolve goal ref '{goal_ref.ref_id}' into an AG GoalNode")

    def _target_refs_for_actions(self, *, graph: AttackGraph, action_ids: Sequence[str]) -> list[GraphRef]:
        """Collect target refs for one candidate action chain."""

        refs: list[GraphRef] = []
        seen: set[tuple[str, str, str | None]] = set()
        for action_id in action_ids:
            node = graph.get_node(action_id)
            if not isinstance(node, ActionNode):
                continue
            for ref in node.source_refs:
                agent_ref = self._to_agent_ref(ref)
                key = (agent_ref.graph.value, agent_ref.ref_id, agent_ref.ref_type)
                if key in seen:
                    continue
                seen.add(key)
                refs.append(agent_ref)
            for edge in graph.list_edges(AGEdgeType.PRODUCES):
                if edge.source != action_id:
                    continue
                produced = graph.get_node(edge.target)
                if isinstance(produced, StateNode):
                    state_ref = GraphRef(graph=GraphScope.AG, ref_id=produced.id, ref_type="StateNode")
                    key = (state_ref.graph.value, state_ref.ref_id, state_ref.ref_type)
                    if key not in seen:
                        seen.add(key)
                        refs.append(state_ref)
        return refs

    @staticmethod
    def _rationale_for_chain(chain: ActionChainCandidate, task_candidates: Sequence[TaskCandidate]) -> str:
        """Build a concise rationale from chain and task candidate metadata."""

        if chain.goal_reached:
            return "candidate chain reaches a goal-enabling state and exports executable task candidates"
        if task_candidates:
            return "candidate chain is the highest-ranked feasible path under current AG activation conditions"
        return chain.reason

    @staticmethod
    def _goal_alignment_bonus(candidate: PlanningCandidate) -> float:
        """Reward candidates that carry more target refs."""

        return min(0.2, len(candidate.target_refs) * 0.02)

    @staticmethod
    def _budget_penalty(candidate: PlanningCandidate, planning_context: PlanningContext) -> float:
        """Apply a coarse budget penalty based on cost/risk/noise totals."""

        budget = planning_context.budget_summary
        if not budget:
            return 0.0
        total_cost = sum(task.estimated_cost for task in candidate.task_candidates)
        total_risk = sum(task.estimated_risk for task in candidate.task_candidates)
        total_noise = sum(task.estimated_noise for task in candidate.task_candidates)
        penalty = 0.0
        if float(budget.get("max_cost", total_cost or 0.0) or 0.0) < total_cost:
            penalty += 0.15
        if float(budget.get("max_risk", total_risk or 0.0) or 0.0) < total_risk:
            penalty += 0.15
        if float(budget.get("max_noise", total_noise or 0.0) or 0.0) < total_noise:
            penalty += 0.1
        return penalty

    @staticmethod
    def _critic_penalty(candidate: PlanningCandidate, planning_context: PlanningContext) -> float:
        """Apply penalties from optional critic hints."""

        penalty = 0.0
        action_ids = set(candidate.action_ids)
        for hint in planning_context.critic_hints:
            hinted_actions = {str(item) for item in hint.get("action_ids", [])}
            if hinted_actions & action_ids:
                severity = str(hint.get("severity", "medium")).lower()
                penalty += {"low": 0.05, "medium": 0.1, "high": 0.2}.get(severity, 0.1)
        return penalty

    @staticmethod
    def _policy_penalty(candidate: PlanningCandidate, planning_context: PlanningContext) -> float:
        """Apply simple approval and parallelism penalties from policy context."""

        policy = planning_context.policy_context
        if not policy:
            return 0.0
        approval_blocked = bool(policy.get("disallow_approval_required", False)) and any(
            task.approval_required for task in candidate.task_candidates
        )
        serial_only = bool(policy.get("prefer_parallel", False)) and any(
            not task.parallelizable for task in candidate.task_candidates
        )
        penalty = 0.0
        if approval_blocked:
            penalty += 0.2
        if serial_only:
            penalty += 0.05
        return penalty

    @staticmethod
    def _to_agent_ref(ref: AGGraphRef) -> GraphRef:
        """Convert AG-model refs into agent-protocol refs."""

        scope = GraphScope(ref.graph)
        return GraphRef(graph=scope, ref_id=ref.ref_id, ref_type=ref.ref_type)

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any]:
        """Return a shallow mapping copy or an empty mapping."""

        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def _coerce_list_of_mappings(value: Any) -> list[dict[str, Any]]:
        """Normalize critic-like hint payloads into a list of mappings."""

        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [dict(item) for item in items if isinstance(item, dict)]


__all__ = [
    "PlannerAgent",
    "PlannerLLMAdvice",
    "PlannerLLMAdvisor",
    "PlannerLLMDecision",
    "PlannerLLMRankAdjustment",
    "PlanningCandidate",
    "PlanningContext",
]
