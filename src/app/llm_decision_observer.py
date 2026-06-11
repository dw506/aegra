"""LLM decision observation helpers for orchestration cycles."""

from __future__ import annotations

from typing import Any

from src.app.settings import AppSettings
from src.core.agents.pipeline_results import PipelineCycleResult
from src.core.agents.agent_protocol import AgentKind
from src.core.runtime.llm_history import LLMDecisionHistoryRecord


class LLMDecisionObserver:
    """Extract prompt-free LLM decision history from pipeline outputs."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    def extract(self, *, cycle_index: int, cycle: PipelineCycleResult) -> list[LLMDecisionHistoryRecord]:
        records: list[LLMDecisionHistoryRecord] = []
        seen: set[tuple[str, str, str | None, str | None, bool, str | None]] = set()
        for step in cycle.steps:
            if step.agent_kind not in {AgentKind.PLANNER, AgentKind.CRITIC, AgentKind.SUPERVISOR}:
                continue
            for container in (
                step.agent_output.decisions,
                step.agent_output.replan_requests,
            ):
                for item in container:
                    for payload in self._iter_llm_payloads(item):
                        record = self._history_record_from_payload(
                            cycle_index=cycle_index,
                            agent_kind=step.agent_kind,
                            payload=payload,
                        )
                        if record is None:
                            continue
                        key = self._record_key(record)
                        if key in seen:
                            continue
                        seen.add(key)
                        records.append(record)
            for record in self._history_records_from_logs(
                cycle_index=cycle_index,
                agent_kind=step.agent_kind,
                logs=step.agent_output.logs,
            ):
                key = self._record_key(record)
                if key in seen:
                    continue
                seen.add(key)
                records.append(record)
        return records

    @staticmethod
    def _record_key(
        record: LLMDecisionHistoryRecord,
    ) -> tuple[str, str, str | None, str | None, bool, str | None]:
        return (
            record.agent_kind,
            record.decision_type,
            record.decision_id,
            record.target_id,
            record.accepted,
            record.rejected_reason,
        )

    def _history_record_from_payload(
        self,
        *,
        cycle_index: int,
        agent_kind: AgentKind,
        payload: dict[str, Any],
    ) -> LLMDecisionHistoryRecord | None:
        validation = self._mapping(payload.get("llm_decision_validation"))
        if not validation and isinstance(payload.get("validation"), dict):
            validation = self._mapping(payload.get("validation"))
        if not validation:
            return None
        accepted = bool(validation.get("accepted"))
        reason = str(validation.get("reason") or "") or None
        decision = self._mapping(payload.get("llm_decision")) or self._mapping(payload.get("decision"))
        decision_type = str(
            decision.get("decision_type")
            or payload.get("decision_type")
            or self._default_llm_decision_type(agent_kind)
        )
        decision_metadata = self._mapping(decision.get("metadata"))
        return LLMDecisionHistoryRecord(
            cycle_index=cycle_index,
            agent_kind=agent_kind.value,
            advisor_type=self._advisor_type(agent_kind, observed=True),
            enabled=self._llm_advisor_enabled(agent_kind, observed=True),
            configured=self._settings.to_packy_llm_config() is not None,
            decision_type=decision_type,
            decision_id=self._optional_text(decision.get("decision_id")),
            target_id=self._optional_text(decision.get("target_id")),
            target_kind=self._optional_text(decision.get("target_kind")),
            accepted=accepted,
            rejected_reason=None if accepted else reason,
            model=self._llm_model(),
            usage=self._mapping(decision_metadata.get("llm_usage")) or None,
            cost_usd=self._coerce_optional_float(decision_metadata.get("llm_cost_usd")),
        )

    def _history_records_from_logs(
        self,
        *,
        cycle_index: int,
        agent_kind: AgentKind,
        logs: list[str],
    ) -> list[LLMDecisionHistoryRecord]:
        records: list[LLMDecisionHistoryRecord] = []
        marker = "llm"
        rejected_marker = "rejected:"
        failed_marker = "failed:"
        for log in logs:
            lowered = log.lower()
            if marker not in lowered:
                continue
            if rejected_marker in lowered:
                reason = log.split(rejected_marker, 1)[1].strip()
            elif failed_marker in lowered:
                reason = log.split(failed_marker, 1)[1].strip()
            else:
                continue
            records.append(
                LLMDecisionHistoryRecord(
                    cycle_index=cycle_index,
                    agent_kind=agent_kind.value,
                    advisor_type=self._advisor_type(agent_kind, observed=True),
                    enabled=self._llm_advisor_enabled(agent_kind, observed=True),
                    configured=self._settings.to_packy_llm_config() is not None,
                    decision_type=self._default_llm_decision_type(agent_kind),
                    accepted=False,
                    rejected_reason=reason or "llm decision rejected",
                    model=self._llm_model(),
                )
            )
        return records

    def _llm_advisor_enabled(self, agent_kind: AgentKind, *, observed: bool = False) -> bool:
        if agent_kind == AgentKind.PLANNER:
            return self._planner_llm_enabled() or observed
        if agent_kind == AgentKind.CRITIC:
            return self._settings.enable_critic_llm_advisor or observed
        if agent_kind == AgentKind.SUPERVISOR:
            return self._settings.enable_supervisor_llm_advisor or observed
        return observed

    def _advisor_type(self, agent_kind: AgentKind, *, observed: bool = False) -> str:
        if not self._llm_advisor_enabled(agent_kind, observed=observed):
            return "none"
        return "packy" if self._settings.to_packy_llm_config() is not None else "injected"

    def _planner_llm_enabled(self) -> bool:
        return (
            self._settings.enable_planner_llm_advisor
            or self._settings.enable_planner_rank_llm_advisor
            or self._settings.enable_graph_llm_planner_advisor
        )

    def _llm_model(self) -> str | None:
        config = self._settings.to_packy_llm_config()
        return config.model if config is not None else None

    @staticmethod
    def _default_llm_decision_type(agent_kind: AgentKind) -> str:
        if agent_kind == AgentKind.PLANNER:
            return "planner_strategy_decision"
        if agent_kind == AgentKind.CRITIC:
            return "critic_finding_review"
        if agent_kind == AgentKind.SUPERVISOR:
            return "supervisor_strategy"
        return "llm_decision"

    def _iter_llm_payloads(self, value: Any) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        if isinstance(value, dict):
            if "llm_decision_validation" in value or (
                "validation" in value and ("llm_decision" in value or "decision" in value)
            ):
                payloads.append(value)
            for item in value.values():
                payloads.extend(self._iter_llm_payloads(item))
        elif isinstance(value, list):
            for item in value:
                payloads.extend(self._iter_llm_payloads(item))
        return payloads

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = ["LLMDecisionObserver"]
