"""Supervisor agent skeleton for bounded high-level strategy advice."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_models import DecisionRecord
from src.core.agents.agent_protocol import AgentInput, AgentKind, AgentOutput, BaseAgent, WritePermission
from src.core.agents.llm_decision import (
    LLMDecisionValidationResult,
    contains_forbidden_llm_decision_key,
)


class SupervisorStrategy(str, Enum):
    """Allowed high-level strategy advice emitted by SupervisorAgent."""

    CONTINUE_PLANNING = "continue_planning"
    CONTINUE_EXECUTION = "continue_execution"
    REQUEST_REPLAN = "request_replan"
    PAUSE_FOR_REVIEW = "pause_for_review"
    STOP_WHEN_QUIESCENT = "stop_when_quiescent"


class SupervisorContext(BaseModel):
    """Bounded input summary available to SupervisorAgent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    runtime_summary: dict[str, Any] = Field(default_factory=dict)
    last_control_cycle: dict[str, Any] = Field(default_factory=dict)
    planner_summary: dict[str, Any] = Field(default_factory=dict)
    critic_summary: dict[str, Any] = Field(default_factory=dict)
    budget_summary: dict[str, Any] = Field(default_factory=dict)


class SupervisorDecision(BaseModel):
    """Structured supervisor strategy recommendation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    strategy: SupervisorStrategy
    rationale: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    requires_human_review: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SupervisorLLMAdvisor(Protocol):
    """Protocol implemented by optional LLM-backed supervisor advisors."""

    def advise(self, *, context: SupervisorContext) -> SupervisorDecision | None:
        """Return a bounded strategy recommendation, or None to fall back."""


class SupervisorAgent(BaseAgent):
    """Control-only high-level strategy advisor.

    The agent is deliberately advisory. It never writes KG/AG/TG/Runtime state,
    never emits replan requests, and never dispatches workers or tool commands.
    """

    def __init__(
        self,
        *,
        name: str = "supervisor_agent",
        llm_advisor: SupervisorLLMAdvisor | None = None,
    ) -> None:
        super().__init__(
            name=name,
            kind=AgentKind.SUPERVISOR,
            write_permission=WritePermission(
                scopes=[],
                allow_structural_write=False,
                allow_state_write=False,
                allow_event_emit=True,
            ),
        )
        self._llm_advisor = llm_advisor

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        context = self._resolve_context(agent_input.raw_payload)
        decision = self._fallback_decision(context)
        validation = LLMDecisionValidationResult.rejected_result(reason="no supervisor llm advisor configured")
        adopted = False
        logs = ["supervisor evaluated high-level strategy in advisory mode"]

        if self._llm_advisor is not None:
            try:
                proposed = self._llm_advisor.advise(context=context)
            except Exception as exc:
                proposed = None
                validation = LLMDecisionValidationResult.rejected_result(
                    reason=f"supervisor llm advisor failed: {exc.__class__.__name__}"
                )
            if proposed is not None:
                validation = self._validate_decision(proposed)
                if validation.accepted:
                    decision = proposed
                    adopted = True
                    logs.append("supervisor adopted llm strategy suggestion")
                else:
                    logs.append(f"supervisor rejected llm strategy suggestion: {validation.reason}")

        record = self._decision_record(
            decision,
            adopted=adopted,
            validation=validation,
        )
        return AgentOutput(
            decisions=[record.to_agent_output_fragment()],
            logs=logs,
        )

    @staticmethod
    def _resolve_context(payload: dict[str, Any]) -> SupervisorContext:
        return SupervisorContext(
            runtime_summary=SupervisorAgent._mapping(payload.get("runtime_summary")),
            last_control_cycle=SupervisorAgent._mapping(payload.get("last_control_cycle")),
            planner_summary=SupervisorAgent._mapping(payload.get("planner_summary")),
            critic_summary=SupervisorAgent._mapping(payload.get("critic_summary")),
            budget_summary=SupervisorAgent._mapping(payload.get("budget_summary")),
        )

    @staticmethod
    def _fallback_decision(context: SupervisorContext) -> SupervisorDecision:
        if bool(context.budget_summary.get("requires_human_review")):
            return SupervisorDecision(
                strategy=SupervisorStrategy.PAUSE_FOR_REVIEW,
                rationale="budget summary requires human review",
                confidence=0.6,
                requires_human_review=True,
                metadata={"source": "heuristic"},
            )
        if SupervisorAgent._int_value(context.runtime_summary.get("replan_request_count")) > 0:
            return SupervisorDecision(
                strategy=SupervisorStrategy.REQUEST_REPLAN,
                rationale="runtime summary contains pending replan requests",
                confidence=0.6,
                metadata={"source": "heuristic"},
            )
        if SupervisorAgent._int_value(context.critic_summary.get("finding_count")) > 0:
            return SupervisorDecision(
                strategy=SupervisorStrategy.REQUEST_REPLAN,
                rationale="critic summary contains findings that may need replanning",
                confidence=0.55,
                metadata={"source": "heuristic"},
            )
        if bool(context.last_control_cycle.get("stopped")):
            return SupervisorDecision(
                strategy=SupervisorStrategy.STOP_WHEN_QUIESCENT,
                rationale="last control cycle already reached quiescence",
                confidence=0.65,
                metadata={"source": "heuristic"},
            )
        if SupervisorAgent._int_value(context.planner_summary.get("pending_candidate_count")) > 0:
            return SupervisorDecision(
                strategy=SupervisorStrategy.CONTINUE_PLANNING,
                rationale="planner summary still has pending candidates",
                confidence=0.55,
                metadata={"source": "heuristic"},
            )
        return SupervisorDecision(
            strategy=SupervisorStrategy.CONTINUE_EXECUTION,
            rationale="no blocking review, replan, or quiescence signal is present",
            confidence=0.5,
            metadata={"source": "heuristic"},
        )

    @staticmethod
    def _validate_decision(decision: SupervisorDecision) -> LLMDecisionValidationResult:
        payload = decision.model_dump(mode="json")
        forbidden_key = contains_forbidden_llm_decision_key(payload)
        if forbidden_key is not None:
            return LLMDecisionValidationResult.rejected_result(
                reason=f"supervisor decision contains forbidden field: {forbidden_key}"
            )
        return LLMDecisionValidationResult.accepted_result(
            sanitized_payload={
                "strategy": decision.strategy.value,
                "rationale": decision.rationale,
                "confidence": decision.confidence,
                "requires_human_review": decision.requires_human_review,
                "metadata": SupervisorAgent._safe_metadata(decision.metadata),
            }
        )

    def _decision_record(
        self,
        decision: SupervisorDecision,
        *,
        adopted: bool,
        validation: LLMDecisionValidationResult,
    ) -> DecisionRecord:
        return DecisionRecord(
            source_agent=self.name,
            summary=f"supervisor recommends {decision.strategy.value}",
            confidence=decision.confidence,
            refs=[],
            payload={
                "supervisor_decision": {
                    "strategy": decision.strategy.value,
                    "rationale": decision.rationale,
                    "requires_human_review": decision.requires_human_review,
                    "metadata": self._safe_metadata(decision.metadata),
                },
                "llm_decision_validation": validation.model_dump(mode="json"),
                "llm_adopted": adopted,
                "control_only": True,
            },
            decision_type="supervisor_strategy",
            score=decision.confidence,
            target_refs=[],
            rationale=decision.rationale,
        )

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in metadata.items()
            if "api_key" not in key.lower() and "secret" not in key.lower()
        }

    @staticmethod
    def _int_value(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0


__all__ = [
    "SupervisorAgent",
    "SupervisorContext",
    "SupervisorDecision",
    "SupervisorLLMAdvisor",
    "SupervisorStrategy",
]
