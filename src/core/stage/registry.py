"""Registry for resolving Stage Agents by stage type."""

from __future__ import annotations

from src.core.execution.mcp_client import MCPClient
from src.core.stage.agents import AccessPivotAgent, ExploitValidationAgent, GoalAgent, ReconAgent, VulnAnalysisAgent
from src.core.stage.base_stage_agent import BaseStageAgent, StageAgentAdvisor
from src.core.stage.models import StageName, normalize_stage_name


class StageAgentRegistry:
    """Resolve the dedicated agent for each stage type."""

    def __init__(self, agents: list[BaseStageAgent] | None = None) -> None:
        self._agents: dict[str, BaseStageAgent] = {}
        for agent in agents or []:
            self.register(agent)

    @classmethod
    def default(
        cls,
        *,
        advisor: StageAgentAdvisor | None = None,
        mcp_client: MCPClient | None = None,
        default_timeout_seconds: int = 60,
    ) -> "StageAgentRegistry":
        kwargs = {
            "advisor": advisor,
            "mcp_client": mcp_client,
            "default_timeout_seconds": default_timeout_seconds,
        }
        return cls(
            [
                ReconAgent(**kwargs),
                VulnAnalysisAgent(**kwargs),
                ExploitValidationAgent(**kwargs),
                AccessPivotAgent(**kwargs),
                GoalAgent(**kwargs),
            ]
        )

    def register(self, agent: BaseStageAgent) -> None:
        self._agents[normalize_stage_name(agent.stage_type)] = agent

    def resolve(self, stage_type: StageName | str) -> BaseStageAgent:
        resolved = normalize_stage_name(stage_type)
        try:
            return self._agents[resolved]
        except KeyError as exc:
            raise ValueError(f"no StageAgent registered for {resolved}") from exc


__all__ = ["StageAgentRegistry"]
