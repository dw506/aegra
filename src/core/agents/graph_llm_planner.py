"""Graph-driven LLM planner advisor.

This advisor asks an LLM for declarative graph task proposals based on a
compact `GraphContext`. It never executes commands and returns only proposals
that pass the graph-plan validator.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.graph_context import (
    GraphContext,
    GraphContextAction,
    GraphContextEvidence,
    GraphContextGoal,
    GraphContextRef,
    GraphContextService,
    GraphContextTask,
)
from src.core.agents.graph_llm_models import GraphLLMPlanProposal, GraphLLMPlanValidationResult
from src.core.agents.llm_decision import validate_graph_plan_proposal
from src.core.agents.llm_safety import response_within_limits, sanitize_llm_payload
from src.core.agents.packy_llm import PackyLLMClient, PackyLLMConfig, PackyLLMError
from src.core.models.ag import GraphRef
from src.core.models.tg import TaskType


class GraphLLMPlannerAdvisorConfig(BaseModel):
    """Configuration for graph-driven LLM planning."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str | None = None
    max_context_chars: int = Field(default=12000, ge=1000, le=100000)
    max_response_chars: int = Field(default=20000, ge=1000, le=200000)
    max_response_json_depth: int = Field(default=12, ge=2, le=50)
    max_recent_signals: int = Field(default=12, ge=0, le=50)
    risk_review_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    noise_review_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    system_prompt: str = (
        "You are Aegra's graph planning advisor. Return JSON only. "
        "You may only propose task_proposals, rank_adjustments, and replan_hint fields matching GraphLLMPlanProposal. "
        "Do not output shell commands, command strings, raw payloads, exploit payloads, reverse shells, patches, or graph mutations. "
        "Do not invent graph refs; every target_ref and rank adjustment target_ref must come from visible_refs. "
        "Prefer the next graph tasks that maximize evidence gain while minimizing risk, noise, repetition, and policy cost. "
        "If a task may exceed risk/noise policy, mark requires_human_review true instead of providing executable details."
    )


class GraphLLMPlannerAdvice(BaseModel):
    """Validated graph planner advice returned to downstream planning code."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    proposal: GraphLLMPlanProposal = Field(default_factory=GraphLLMPlanProposal)
    validation: GraphLLMPlanValidationResult
    llm_metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def empty(
        cls,
        *,
        reason: str,
        llm_metadata: dict[str, Any] | None = None,
        rejected_items: list[dict[str, Any]] | None = None,
    ) -> "GraphLLMPlannerAdvice":
        return cls(
            proposal=GraphLLMPlanProposal(metadata={"fallback": "empty"}),
            validation=GraphLLMPlanValidationResult.rejected_result(
                reason=reason,
                rejected_items=rejected_items,
            ),
            llm_metadata=llm_metadata or {},
        )


class GraphLLMPlannerAdvisor:
    """LLM advisor that proposes graph task plans from `GraphContext`."""

    def __init__(
        self,
        *,
        client: PackyLLMClient,
        config: GraphLLMPlannerAdvisorConfig | None = None,
    ) -> None:
        self._client = client
        self._config = config or GraphLLMPlannerAdvisorConfig()
        self.last_failure: dict[str, Any] | None = None

    @classmethod
    def from_env(
        cls,
        *,
        config: GraphLLMPlannerAdvisorConfig | None = None,
        client_config: PackyLLMConfig | None = None,
    ) -> "GraphLLMPlannerAdvisor":
        """Create an advisor from environment-driven Packy/OpenAI config."""

        return cls(
            client=PackyLLMClient(client_config or PackyLLMConfig.from_env()),
            config=config,
        )

    def advise(
        self,
        *,
        graph_context: GraphContext,
        goal_refs: Sequence[GraphRef] | None = None,
        policy_context: dict[str, Any] | None = None,
        recent_signals: Sequence[dict[str, Any]] | None = None,
    ) -> GraphLLMPlannerAdvice:
        visible_refs = self._visible_refs(graph_context, goal_refs or [])
        self.last_failure = None
        user_prompt = self._build_user_prompt(
            graph_context=graph_context,
            goal_refs=goal_refs or [],
            visible_refs=visible_refs,
            policy_context=policy_context or graph_context.policy.model_dump(mode="json"),
            recent_signals=recent_signals or [],
        )

        try:
            response = self._client.complete_chat(
                user_prompt=user_prompt,
                system_prompt=self._config.system_prompt,
                model=self._config.model,
                temperature=0.0,
            )
        except PackyLLMError as exc:
            self.last_failure = {"reason": "llm_call_failed", "error": str(exc)}
            return GraphLLMPlannerAdvice.empty(
                reason="llm call failed",
                llm_metadata=self._llm_metadata(error=str(exc)),
            )

        llm_metadata = self._llm_metadata(
            model=response.model,
            finish_reason=response.finish_reason,
            usage=response.usage,
            cost_usd=response.cost_usd,
        )
        payload = self._extract_json_payload(response.text)
        if payload is None:
            self.last_failure = {"reason": "invalid_graph_llm_planner_json"}
            return GraphLLMPlannerAdvice.empty(
                reason="invalid graph llm planner json",
                llm_metadata=llm_metadata,
            )
        if not response_within_limits(
            payload,
            raw_text=response.text,
            max_chars=self._config.max_response_chars,
            max_depth=self._config.max_response_json_depth,
        ):
            self.last_failure = {"reason": "llm_response_exceeds_limits"}
            return GraphLLMPlannerAdvice.empty(
                reason="llm response exceeds configured limits",
                llm_metadata=llm_metadata,
            )

        validation = validate_graph_plan_proposal(
            payload,
            visible_refs=visible_refs,
            policy_context=policy_context or graph_context.policy.model_dump(mode="json"),
            risk_review_threshold=self._config.risk_review_threshold,
            noise_review_threshold=self._config.noise_review_threshold,
        )
        if not validation.accepted:
            self.last_failure = {"reason": validation.reason}
            return GraphLLMPlannerAdvice.empty(
                reason=validation.reason,
                llm_metadata=llm_metadata,
                rejected_items=validation.rejected_items,
            )

        proposal = GraphLLMPlanProposal.model_validate(validation.sanitized_payload)
        return GraphLLMPlannerAdvice(
            proposal=proposal,
            validation=validation,
            llm_metadata=llm_metadata,
        )

    def _build_user_prompt(
        self,
        *,
        graph_context: GraphContext,
        goal_refs: Sequence[GraphRef],
        visible_refs: set[str],
        policy_context: dict[str, Any],
        recent_signals: Sequence[dict[str, Any]],
    ) -> str:
        payload = {
            "graph_context": self._bounded_context(graph_context),
            "goal_refs": [ref.model_dump(mode="json") for ref in goal_refs],
            "visible_refs": sorted(visible_refs),
            "allowed_task_types": [task_type.value for task_type in TaskType],
            "tool_catalog": self._tool_catalog(graph_context),
            "policy_context": policy_context,
            "recent_signals": list(recent_signals)[: self._config.max_recent_signals],
            "response_schema": {
                "proposal_id": "optional stable id",
                "task_proposals": [
                    {
                        "proposal_id": "optional stable id",
                        "task_type": "one allowed TaskType value",
                        "target_refs": [{"graph": "kg|ag|tg|query", "ref_id": "visible ref id", "ref_type": "optional"}],
                        "rationale": "short reason, no commands",
                        "expected_evidence": ["evidence this task should produce"],
                        "tool_hint": "one tool_catalog tool_hint value, not command syntax",
                        "params": {
                            "safe_structured_hint": "optional",
                            "target_url": "copy from tool_catalog when available",
                            "service_id": "copy from tool_catalog when available",
                            "validator_id": "safe validator id when proposing vulnerability_validation",
                        },
                        "estimated_risk": 0.0,
                        "estimated_noise": 0.0,
                        "priority": 50,
                        "requires_human_review": False,
                        "metadata": {"reason": "evidence_gain"},
                    }
                ],
                "rank_adjustments": [
                    {
                        "target_ref": {"graph": "ag", "ref_id": "visible action/state/task ref"},
                        "score_delta": 0.0,
                        "rationale": "short reason",
                        "metadata": {"reason": "risk_reduction"},
                    }
                ],
                "replan_hint": "optional bounded planning hint",
                "risk_notes": ["optional risk notes"],
                "requires_human_review": False,
                "metadata": {"reason": "overall reason"},
            },
        }
        return (
            "Return a single JSON object matching GraphLLMPlanProposal and response_schema. "
            "Use only refs listed in visible_refs and tool hints from tool_catalog. "
            "Do not include command, shell, payload, reverse_shell, or raw output fields.\n\n"
            f"{json.dumps(sanitize_llm_payload(payload), ensure_ascii=False, indent=2)}"
        )

    def _tool_catalog(self, context: GraphContext) -> dict[str, Any]:
        capabilities = sorted(
            {
                capability
                for action in context.frontier_actions
                for capability in action.required_capabilities
                if capability
            }
        )
        frontier_tools = [
            {
                "action_ref": action.ref.model_dump(mode="json"),
                "action_type": action.action_type,
                "target_refs": [ref.model_dump(mode="json") for ref in action.target_refs],
                "required_capabilities": list(action.required_capabilities),
                "resource_keys": list(action.resource_keys),
            }
            for action in context.frontier_actions
        ]
        return {
            "available_capabilities": capabilities,
            "task_type_tools": [
                {
                    "task_type": TaskType.ASSET_CONFIRMATION.value,
                    "tool_hint": "safe_probe",
                    "params": ["host_id", "target_host"],
                },
                {
                    "task_type": TaskType.SERVICE_VALIDATION.value,
                    "tool_hint": "safe_fingerprint",
                    "params": ["host_id", "service_id", "port", "protocol", "service_name"],
                },
                {
                    "task_type": TaskType.WEB_ENUMERATION.value,
                    "tool_hint": "safe_http_client",
                    "params": ["target_url", "service_id", "port", "protocol"],
                },
                {
                    "task_type": TaskType.VULNERABILITY_VALIDATION.value,
                    "tool_hint": "safe_vulnerability_validator",
                    "params": ["validator_id", "target_url", "service_id", "port", "protocol"],
                },
            ],
            "known_service_params": [self._service_tool_context(service) for service in context.known_services],
            "frontier_tools": frontier_tools,
        }

    @staticmethod
    def _service_tool_context(service: GraphContextService) -> dict[str, Any]:
        return {
            "service_ref": service.ref.model_dump(mode="json"),
            "subject_refs": [ref.model_dump(mode="json") for ref in service.subject_refs],
            "host": service.host,
            "host_id": next((ref.ref_id for ref in service.subject_refs if ref.ref_type == "Host"), service.host),
            "service_id": service.ref.ref_id,
            "port": service.port,
            "protocol": service.protocol,
            "service_name": service.service_name,
            "target_url": service.properties.get("target_url") or service.properties.get("url"),
        }

    def _bounded_context(self, context: GraphContext) -> dict[str, Any]:
        payload = context.model_dump(mode="json")
        serialized = json.dumps(payload, ensure_ascii=False)
        if len(serialized) <= self._config.max_context_chars:
            return payload
        return {
            "operation_id": context.operation_id,
            "graph_versions": context.graph_versions,
            "goals": [goal.model_dump(mode="json") for goal in context.goals],
            "frontier_actions": [
                action.model_dump(mode="json")
                for action in context.frontier_actions[: max(1, self._config.max_context_chars // 2000)]
            ],
            "policy": context.policy.model_dump(mode="json"),
            "context_stats": {
                **context.context_stats,
                "truncated_for_llm": True,
                "original_context_chars": len(serialized),
            },
        }

    @classmethod
    def _visible_refs(cls, context: GraphContext, goal_refs: Sequence[GraphRef]) -> set[str]:
        keys = {ref.key() for ref in goal_refs}
        for goal in context.goals:
            keys.update(cls._refs_from_goal(goal))
        for service in context.known_services:
            keys.update(cls._refs_from_service(service))
        for action in context.frontier_actions:
            keys.update(cls._refs_from_action(action))
        for tasks in context.tasks_by_status.values():
            for task in tasks:
                keys.update(cls._refs_from_task(task))
        for evidence in context.evidence:
            keys.update(cls._refs_from_evidence(evidence))
        return {key for key in keys if key}

    @classmethod
    def _refs_from_goal(cls, goal: GraphContextGoal) -> set[str]:
        refs = [goal.ref, *goal.scope_refs]
        return cls._ref_keys(refs)

    @classmethod
    def _refs_from_service(cls, service: GraphContextService) -> set[str]:
        refs = [service.ref, *service.subject_refs]
        return cls._ref_keys(refs)

    @classmethod
    def _refs_from_action(cls, action: GraphContextAction) -> set[str]:
        refs = [action.ref, *action.target_refs]
        return cls._ref_keys(refs)

    @classmethod
    def _refs_from_task(cls, task: GraphContextTask) -> set[str]:
        refs = [task.ref, *task.target_refs]
        return cls._ref_keys(refs)

    @classmethod
    def _refs_from_evidence(cls, evidence: GraphContextEvidence) -> set[str]:
        return cls._ref_keys([evidence.ref])

    @staticmethod
    def _ref_keys(refs: Sequence[GraphContextRef]) -> set[str]:
        return {f"{ref.graph}:{ref.ref_id}" for ref in refs}

    def _llm_metadata(self, **extra: Any) -> dict[str, Any]:
        base_url = getattr(self._client.config, "base_url", None)
        model = extra.pop("model", None) or self._config.model or getattr(self._client.config, "model", None)
        return {
            "model": model,
            "base_url": base_url,
            **{key: value for key, value in extra.items() if value is not None},
        }

    @classmethod
    def _extract_json_payload(cls, text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if not stripped:
            return None
        if stripped.startswith("```"):
            stripped = cls._strip_code_fences(stripped)
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            block = cls._find_first_json_object(stripped)
            if block is None:
                return None
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _find_first_json_object(text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
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
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return None


__all__ = [
    "GraphLLMPlannerAdvice",
    "GraphLLMPlannerAdvisor",
    "GraphLLMPlannerAdvisorConfig",
]
