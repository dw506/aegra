"""Registry for resolving the execution agent (P2: single executor)."""

from __future__ import annotations

from src.core.agents.packy_llm import PackyLLMClient
from src.core.execution.mcp_client import MCPClient
from src.core.stage.agents import ExecutionStageAgent
from src.core.stage.llm_driven_stage_agent import LLMDrivenStageAgent
from src.core.stage.models import StageName, normalize_stage_name


class StageAgentRegistry:
    """Resolve the execution agent for a dispatched round.

    Since the 5 stage agents collapsed into a single ``ExecutionStageAgent``, the
    common case is a registry holding exactly one agent: it is returned for any
    stage_type / agent_name. The legacy by-stage map is retained only for tests
    that inject several custom agents.
    """

    def __init__(self, agents: list[LLMDrivenStageAgent] | None = None) -> None:
        self._agents: dict[str, LLMDrivenStageAgent] = {}
        self._agents_by_name: dict[str, LLMDrivenStageAgent] = {}
        for agent in agents or []:
            self.register(agent)

    @classmethod
    def default(
        cls,
        *,
        llm_client: PackyLLMClient | None = None,
        mcp_client: MCPClient | None = None,
        default_timeout_seconds: int = 120,
    ) -> "StageAgentRegistry":
        return cls(
            [
                ExecutionStageAgent(
                    llm_client=llm_client,
                    mcp_client=mcp_client,
                    default_timeout_seconds=default_timeout_seconds,
                )
            ]
        )

    def register(self, agent: LLMDrivenStageAgent) -> None:
        self._agents[normalize_stage_name(agent.stage_type)] = agent
        self._agents_by_name[agent.agent_name] = agent

    @property
    def single_agent(self) -> LLMDrivenStageAgent | None:
        """Return the sole agent when exactly one is registered, else None."""

        if len(self._agents_by_name) == 1:
            return next(iter(self._agents_by_name.values()))
        return None

    def resolve(self, stage_type: StageName | str) -> LLMDrivenStageAgent:
        single = self.single_agent
        if single is not None:
            return single
        resolved = normalize_stage_name(stage_type)
        try:
            return self._agents[resolved]
        except KeyError as exc:
            raise ValueError(f"no StageAgent registered for {resolved}") from exc

    def resolve_agent(self, agent_name: str) -> LLMDrivenStageAgent:
        single = self.single_agent
        if single is not None:
            return single
        try:
            return self._agents_by_name[agent_name]
        except KeyError as exc:
            raise ValueError(f"no StageAgent registered for agent {agent_name}") from exc

    def validate_assignment(self, *, agent_name: str, stage_type: StageName | str) -> None:
        """Ensure Planner-selected agent/stage pairs match registered capabilities.

        With a single executor this is a no-op (it serves every stage); the check
        is only meaningful for legacy multi-agent registries used in tests.
        """

        if self.single_agent is not None:
            return
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
