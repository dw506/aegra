"""Registry for resolving Stage Agents by stage type."""

from __future__ import annotations

from collections.abc import Mapping

from src.core.agents.packy_llm import PackyLLMClient
from src.core.execution.mcp_client import MCPClient
from src.core.stage.agents import AccessPivotAgent, ExploitValidationAgent, GoalAgent, ReconAgent, VulnAnalysisAgent
from src.core.stage.base_stage_agent import BaseStageAgent, StageAgentAdvisor
from src.core.stage.models import StageName, normalize_stage_name


class StageAgentRegistry:
    """Resolve the dedicated agent for each stage type."""

    def __init__(self, agents: list[BaseStageAgent] | None = None) -> None:
        self._agents: dict[str, BaseStageAgent] = {}
        self._agents_by_name: dict[str, BaseStageAgent] = {}
        for agent in agents or []:
            self.register(agent)

    @classmethod
    def default(
        cls,
        *,
        llm_client: PackyLLMClient | None = None,
        advisor: StageAgentAdvisor | None = None,
        advisors: Mapping[str, StageAgentAdvisor | None] | None = None,
        mcp_client: MCPClient | None = None,
        default_timeout_seconds: int = 60,
    ) -> "StageAgentRegistry":
        advisor_map = dict(advisors or {})

        def kwargs_for(agent_name: str) -> dict[str, object]:
            return {
                "advisor": advisor_map.get(agent_name, advisor),
                "llm_client": None if advisor_map.get(agent_name, advisor) is not None else llm_client,
                "mcp_client": mcp_client,
                "default_timeout_seconds": default_timeout_seconds,
            }

        return cls(
            [
                ReconAgent(**kwargs_for("recon_agent")),
                VulnAnalysisAgent(**kwargs_for("vuln_analysis_agent")),
                ExploitValidationAgent(**kwargs_for("exploit_validation_agent")),
                AccessPivotAgent(**kwargs_for("access_pivot_agent")),
                GoalAgent(**kwargs_for("goal_agent")),
            ]
        )

    def register(self, agent: BaseStageAgent) -> None:
        self._agents[normalize_stage_name(agent.stage_type)] = agent
        self._agents_by_name[agent.agent_name] = agent

    def resolve(self, stage_type: StageName | str) -> BaseStageAgent:
        resolved = normalize_stage_name(stage_type)
        try:
            return self._agents[resolved]
        except KeyError as exc:
            raise ValueError(f"no StageAgent registered for {resolved}") from exc

    def resolve_agent(self, agent_name: str) -> BaseStageAgent:
        try:
            return self._agents_by_name[agent_name]
        except KeyError as exc:
            raise ValueError(f"no StageAgent registered for agent {agent_name}") from exc

    def validate_assignment(self, *, agent_name: str, stage_type: StageName | str) -> None:
        """Ensure Planner-selected agent/stage pairs match registered capabilities."""

        agent = self.resolve_agent(agent_name)
        requested_stage = normalize_stage_name(stage_type)
        registered_stage = normalize_stage_name(agent.stage_type)
        if requested_stage != registered_stage:
            raise ValueError(
                f"StageAgent {agent_name} is registered for {registered_stage}, "
                f"not {requested_stage}"
            )

    def capability_summary(self) -> list[dict[str, str]]:
        return [
            {
                "agent_name": agent.agent_name,
                "stage_type": normalize_stage_name(agent.stage_type),
                "context_builder": getattr(agent, "context_builder_name", "stage_context_builder"),
            }
            for agent in self._agents_by_name.values()
        ]


__all__ = ["StageAgentRegistry"]
