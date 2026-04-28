"""Packy-backed planner advisor.

中文注释：
这一层的职责是把底层 LLM 文本结果，转换成 `PlannerAgent` 能消费的
`PlannerLLMAdvice`。它不直接改 planner 打分逻辑，也不执行任何工具。
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_protocol import GraphRef
from src.core.agents.llm_decision import (
    LLMDecision,
    LLMDecisionSource,
    LLMDecisionStatus,
    LLMDecisionValidator,
)
from src.core.agents.packy_llm import PackyLLMClient, PackyLLMConfig, PackyLLMError
from src.core.agents.planner import (
    PlannerLLMDecision,
    PlannerLLMRankAdjustment,
    PlanningCandidate,
    PlanningContext,
)
from src.core.models.ag import AttackGraph


class PackyPlannerAdvisorConfig(BaseModel):
    """Configuration for the Packy-backed planner advisor."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str | None = None
    max_candidates: int = Field(default=5, ge=1, le=10)
    max_abs_score_delta: float = Field(default=0.2, gt=0.0, le=1.0)
    system_prompt: str = (
        "你是 Aegra 的规划建议助手。"
        "你只能对候选任务做排序建议和解释增强，不能生成攻击步骤、工具命令或执行参数。"
        "请只返回 JSON，不要输出额外说明。"
    )


class PackyPlannerAdvisor:
    """Planner advisor backed by `PackyLLMClient`."""

    def __init__(
        self,
        *,
        client: PackyLLMClient,
        config: PackyPlannerAdvisorConfig | None = None,
    ) -> None:
        self._client = client
        self._config = config or PackyPlannerAdvisorConfig()

    @classmethod
    def from_env(
        cls,
        *,
        config: PackyPlannerAdvisorConfig | None = None,
        client_config: PackyLLMConfig | None = None,
    ) -> "PackyPlannerAdvisor":
        """Create an advisor from environment-driven client config."""

        return cls(
            client=PackyLLMClient(client_config or PackyLLMConfig.from_env()),
            config=config,
        )

    def advise(
        self,
        *,
        graph: AttackGraph,
        goal_ref: GraphRef,
        candidates: Sequence[PlanningCandidate],
        planning_context: PlanningContext,
    ) -> PlannerLLMDecision | list:
        if not candidates:
            return []

        limited_candidates = self._limit_candidates(candidates)
        allowed_candidate_ids = {candidate.candidate_id for candidate in limited_candidates}
        user_prompt = self._build_user_prompt(
            graph=graph,
            goal_ref=goal_ref,
            candidates=limited_candidates,
            planning_context=planning_context,
        )

        try:
            response = self._client.complete_chat(
                user_prompt=user_prompt,
                system_prompt=self._config.system_prompt,
                model=self._config.model,
                temperature=0.0,
            )
        except PackyLLMError:
            # 中文注释：
            # advisor 失败时必须安全回退为空建议，不能把 planner 主流程打崩。
            return []

        return self._parse_advice_text(
            response.text,
            allowed_candidate_ids=allowed_candidate_ids,
            goal_id=goal_ref.ref_id,
        )

    def _limit_candidates(self, candidates: Sequence[PlanningCandidate]) -> list[PlanningCandidate]:
        return sorted(candidates, key=lambda item: item.score, reverse=True)[: self._config.max_candidates]

    def _build_user_prompt(
        self,
        *,
        graph: AttackGraph,
        goal_ref: GraphRef,
        candidates: Sequence[PlanningCandidate],
        planning_context: PlanningContext,
    ) -> str:
        prompt_payload = {
            "goal_ref": goal_ref.model_dump(mode="json"),
            "graph_summary": self._graph_summary(graph),
            "planning_context": {
                "top_k": planning_context.top_k,
                "max_depth": planning_context.max_depth,
                "budget_summary": planning_context.budget_summary,
                "policy_context": planning_context.policy_context,
                "critic_hints": planning_context.critic_hints,
            },
            "candidates": [self._candidate_payload(candidate) for candidate in candidates],
            "response_schema": {
                "selected_candidate_ids": ["候选 ID 列表，必须来自输入 candidates"],
                "rank_adjustments": [
                    {
                        "candidate_id": "候选 ID，必须来自输入 candidates",
                        "score_delta": "[-0.2, 0.2] 之间的浮点数，用于微调候选得分",
                        "risk_notes": ["可选风险说明，只能描述已有候选"],
                        "rationale_suffix": "简短中文解释，补在原 rationale 后面",
                        "metadata": {"reason": "建议原因，例如 goal_alignment 或 risk_reduction"},
                    }
                ],
                "risk_notes": ["整体风险说明"],
                "defer_reason": "可选，若建议暂缓选择则说明原因",
                "requires_human_review": "布尔值，是否需要人工复核",
                "metadata": {"reason": "整体建议原因"},
            },
        }
        return (
            "请基于下面的候选列表返回 JSON。"
            "只允许对已有候选做排序和风险建议，不能发明新的 candidate_id，不能生成任务、命令或工具参数。\n\n"
            f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
        )

    @staticmethod
    def _graph_summary(graph: AttackGraph) -> dict[str, Any]:
        try:
            node_count = len(graph.list_nodes())
        except Exception:
            node_count = None
        try:
            edge_count = len(graph.list_edges())
        except Exception:
            edge_count = None
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "version": getattr(graph, "version", None),
        }

    @staticmethod
    def _candidate_payload(candidate: PlanningCandidate) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "score": candidate.score,
            "rationale": candidate.rationale,
            "action_ids": list(candidate.action_ids),
            "target_refs": [ref.model_dump(mode="json") for ref in candidate.target_refs],
            "task_candidates": [
                {
                    "source_action_id": task.source_action_id,
                    "task_type": task.task_type.value,
                    "estimated_cost": task.estimated_cost,
                    "estimated_risk": task.estimated_risk,
                    "estimated_noise": task.estimated_noise,
                    "goal_relevance": task.goal_relevance,
                    "resource_keys": sorted(task.resource_keys),
                }
                for task in candidate.task_candidates
            ],
        }

    def _parse_advice_text(
        self,
        text: str,
        *,
        allowed_candidate_ids: set[str],
        goal_id: str,
    ) -> PlannerLLMDecision | list:
        payload = self._extract_json_payload(text)
        if payload is None:
            return []

        if not isinstance(payload, (dict, list)):
            return []
        if isinstance(payload, dict) and "advice" not in payload:
            return self._parse_strategy_payload(
                payload,
                allowed_candidate_ids=allowed_candidate_ids,
                goal_id=goal_id,
            )

        raw_items = payload.get("advice") if isinstance(payload, dict) else payload

        if not isinstance(raw_items, list):
            return []
        # Backward-compatible legacy shape: convert per-candidate advice into one strategy decision.
        validator = LLMDecisionValidator()
        rank_adjustments: list[PlannerLLMRankAdjustment] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            raw_validation = validator.validate_no_forbidden_payload(raw_item)
            if not raw_validation.accepted:
                continue
            candidate_id = raw_item.get("candidate_id")
            if not isinstance(candidate_id, str) or candidate_id not in allowed_candidate_ids:
                continue
            metadata = raw_item.get("metadata")
            score_delta = self._coerce_score_delta(raw_item.get("score_delta"))
            risk_notes = self._coerce_string_list(raw_item.get("risk_notes"))
            rationale_suffix = raw_item.get("rationale_suffix")
            rank_adjustments.append(
                PlannerLLMRankAdjustment(
                    candidate_id=candidate_id,
                    score_delta=score_delta,
                    rationale_suffix=rationale_suffix if isinstance(rationale_suffix, str) else None,
                    risk_notes=risk_notes,
                    metadata=dict(metadata) if isinstance(metadata, dict) else {},
                )
            )
        if not rank_adjustments:
            return []
        selected_candidate_ids = [item.candidate_id for item in rank_adjustments]
        decision = PlannerLLMDecision(
            selected_candidate_ids=selected_candidate_ids,
            rank_adjustments=rank_adjustments,
            metadata={"source_schema": "legacy_advice"},
        )
        envelope = self._strategy_envelope(
            decision,
            target_id=goal_id,
        )
        validation = validator.validate_planner_strategy_decision(
            envelope,
            allowed_candidate_ids=allowed_candidate_ids,
            allowed_goal_ids={goal_id},
            selected_candidate_ids=decision.selected_candidate_ids,
            rank_adjustments=[
                adjustment.model_dump(mode="json")
                for adjustment in decision.rank_adjustments
            ],
            max_abs_score_delta=self._config.max_abs_score_delta,
        )
        if not validation.accepted:
            return []
        return decision.model_copy(update={"decision": envelope, "validation": validation})

    def _parse_strategy_payload(
        self,
        payload: dict[str, Any],
        *,
        allowed_candidate_ids: set[str],
        goal_id: str,
    ) -> PlannerLLMDecision | list:
        validator = LLMDecisionValidator()
        raw_validation = validator.validate_no_forbidden_payload(payload)
        if not raw_validation.accepted:
            return []
        selected_candidate_ids = self._coerce_candidate_ids(payload.get("selected_candidate_ids"))
        raw_adjustments = payload.get("rank_adjustments")
        rank_adjustments: list[PlannerLLMRankAdjustment] = []
        if isinstance(raw_adjustments, list):
            for raw_item in raw_adjustments:
                if not isinstance(raw_item, dict):
                    continue
                metadata = raw_item.get("metadata")
                rank_adjustments.append(
                    PlannerLLMRankAdjustment(
                        candidate_id=str(raw_item.get("candidate_id", "")),
                        score_delta=self._coerce_raw_score_delta(raw_item.get("score_delta")),
                        rationale_suffix=raw_item.get("rationale_suffix")
                        if isinstance(raw_item.get("rationale_suffix"), str)
                        else None,
                        risk_notes=self._coerce_string_list(raw_item.get("risk_notes")),
                        metadata=dict(metadata) if isinstance(metadata, dict) else {},
                    )
                )
        metadata = payload.get("metadata")
        planner_decision = PlannerLLMDecision(
            selected_candidate_ids=selected_candidate_ids,
            rank_adjustments=rank_adjustments,
            risk_notes=self._coerce_string_list(payload.get("risk_notes")),
            defer_reason=payload.get("defer_reason") if isinstance(payload.get("defer_reason"), str) else None,
            requires_human_review=bool(payload.get("requires_human_review", False)),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )
        envelope = self._strategy_envelope(planner_decision, target_id=goal_id)
        validation = validator.validate_planner_strategy_decision(
            envelope,
            allowed_candidate_ids=allowed_candidate_ids,
            allowed_goal_ids={goal_id},
            selected_candidate_ids=planner_decision.selected_candidate_ids,
            rank_adjustments=[
                adjustment.model_dump(mode="json")
                for adjustment in planner_decision.rank_adjustments
            ],
            max_abs_score_delta=self._config.max_abs_score_delta,
        )
        if not validation.accepted:
            return []
        return planner_decision.model_copy(update={"decision": envelope, "validation": validation})

    def _coerce_score_delta(self, value: Any) -> float:
        try:
            score_delta = float(value)
        except (TypeError, ValueError):
            return 0.0
        limit = self._config.max_abs_score_delta
        if score_delta > limit:
            return limit
        if score_delta < -limit:
            return -limit
        return score_delta

    @staticmethod
    def _coerce_raw_score_delta(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coerce_string_list(value: Any) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [item for item in items if isinstance(item, str) and item.strip()]

    @staticmethod
    def _coerce_candidate_ids(value: Any) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [item for item in items if isinstance(item, str) and item.strip()]

    @staticmethod
    def _strategy_envelope(decision: PlannerLLMDecision, *, target_id: str) -> LLMDecision:
        return LLMDecision(
            source=LLMDecisionSource.PLANNER,
            status=LLMDecisionStatus.ACCEPTED,
            decision_type="planner_strategy_decision",
            target_id=target_id,
            target_kind="planner_goal",
            risk_notes=list(decision.risk_notes),
            metadata={
                **dict(decision.metadata),
                "selected_candidate_ids": list(decision.selected_candidate_ids),
                "rank_adjustments": [
                    adjustment.model_dump(mode="json")
                    for adjustment in decision.rank_adjustments
                ],
                "defer_reason": decision.defer_reason,
                "requires_human_review": decision.requires_human_review,
            },
        )

    @classmethod
    def _extract_json_payload(cls, text: str) -> dict[str, Any] | list[Any] | None:
        stripped = text.strip()
        if not stripped:
            return None
        if stripped.startswith("```"):
            stripped = cls._strip_code_fences(stripped)
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            block = cls._find_first_json_block(stripped)
            if block is None:
                return None
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                return None
        return payload if isinstance(payload, (dict, list)) else None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return text
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _find_first_json_block(text: str) -> str | None:
        # 中文注释：
        # LLM 有时会在 JSON 前后附带少量解释文字，这里只提取第一段平衡的
        # `{...}` 或 `[...]`，尽量把解析容错放在 advisor 内部消化掉。
        start = -1
        opening = ""
        for index, char in enumerate(text):
            if char in "{[":
                start = index
                opening = char
                break
        if start < 0:
            return None

        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return None


__all__ = [
    "PackyPlannerAdvisor",
    "PackyPlannerAdvisorConfig",
]
