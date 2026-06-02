"""Shared wrapper for dedicated StageAgent LLM advisors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.core.agents.packy_llm import PackyLLMClient
from src.core.stage.base_stage_agent import StageAgentDecision
from src.core.stage.llm_stage_advisor import LLMStageAdvisor, LLMStageAdvisorConfig
from src.core.stage.models import StageExecutionRequest, StageName


ContextBuilder = Callable[
    [StageExecutionRequest, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]],
    dict[str, Any],
]


class DedicatedLLMStageAdvisor:
    """Bind one LLM advisor instance to one StageAgent and context builder."""

    def __init__(
        self,
        *,
        agent_name: str,
        stage_type: StageName,
        client: PackyLLMClient,
        context_builder: ContextBuilder,
        config: LLMStageAdvisorConfig | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.stage_type = stage_type
        self._context_builder = context_builder
        self._llm = LLMStageAdvisor(client=client, config=config)

    def build_context(
        self,
        request: StageExecutionRequest,
        graph_context: dict[str, Any],
        runtime_context: dict[str, Any],
        policy_context: dict[str, Any],
        memory: list[dict[str, Any]],
        available_tools: dict[str, Any],
    ) -> dict[str, Any]:
        return self._context_builder(request, graph_context, runtime_context, policy_context, memory, available_tools)

    def decide(
        self,
        *,
        request: StageExecutionRequest,
        graph_context: dict[str, Any],
        runtime_context: dict[str, Any],
        policy_context: dict[str, Any],
        memory: list[dict[str, Any]],
        available_tools: dict[str, Any],
    ) -> StageAgentDecision | dict[str, Any]:
        return self._llm.decide(
            agent_name=self.agent_name,
            stage_type=self.stage_type,
            request=request,
            graph_context=graph_context,
            runtime_context=runtime_context,
            policy_context=policy_context,
            memory=memory,
            available_tools=available_tools,
        )


__all__ = ["DedicatedLLMStageAdvisor"]
