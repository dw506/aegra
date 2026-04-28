"""Agent registry for shared agent lookup and dispatch."""

from __future__ import annotations

import logging

from src.core.agents.agent_protocol import AgentExecutionResult, AgentInput, AgentKind, BaseAgent


logger = logging.getLogger(__name__)


class AgentRegistrationError(ValueError):
    """Raised when an agent cannot be registered in the registry."""


class AgentNotFoundError(KeyError):
    """Raised when the requested agent is not present in the registry."""


class AgentRegistry:
    """In-memory registry for agent instances.

    The registry is intentionally lightweight and orchestration-agnostic. It
    supports name-based lookup, kind-based filtering and dispatch through the
    standard `BaseAgent.run()` entrypoint.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""

        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> BaseAgent:
        """Register one agent instance by its unique name.

        Raises:
            AgentRegistrationError: If the agent name is already registered.
        """

        if agent.name in self._agents:
            raise AgentRegistrationError(f"agent '{agent.name}' is already registered")
        self._agents[agent.name] = agent
        logger.info("registered agent name=%s kind=%s", agent.name, agent.kind.value)
        return agent

    def unregister(self, agent_name: str) -> BaseAgent:
        """Remove one agent from the registry.

        Raises:
            AgentNotFoundError: If the agent is not registered.
        """

        agent = self._agents.pop(agent_name, None)
        if agent is None:
            raise AgentNotFoundError(f"agent '{agent_name}' is not registered")
        logger.info("unregistered agent name=%s kind=%s", agent.name, agent.kind.value)
        return agent

    def get(self, agent_name: str) -> BaseAgent:
        """Return one agent by name.

        Raises:
            AgentNotFoundError: If the agent is not registered.
        """

        agent = self._agents.get(agent_name)
        if agent is None:
            raise AgentNotFoundError(f"agent '{agent_name}' is not registered")
        return agent

    def list_all(self) -> list[BaseAgent]:
        """Return all registered agents ordered by name."""

        return [self._agents[name] for name in sorted(self._agents)]

    def list_by_kind(self, kind: AgentKind) -> list[BaseAgent]:
        """Return all registered agents matching the given kind."""

        return sorted(
            (agent for agent in self._agents.values() if agent.kind == kind),
            key=lambda item: item.name,
        )

    def dispatch(self, agent_name: str, agent_input: AgentInput) -> AgentExecutionResult:
        """Dispatch one input envelope to the named agent.

        The registry performs a small amount of coordination:
        - resolves the named agent,
        - validates the input through the agent,
        - runs the agent through the standard execution entrypoint.
        """

        agent = self.get(agent_name)
        logger.info(
            "dispatching agent name=%s kind=%s operation_id=%s task_ref=%s",
            agent.name,
            agent.kind.value,
            agent_input.context.operation_id,
            agent_input.task_ref,
        )
        agent.validate_input(agent_input)
        result = agent.run(agent_input)
        logger.info(
            "completed agent name=%s success=%s duration_ms=%s",
            agent.name,
            result.success,
            result.duration_ms,
        )
        return result


__all__ = [
    "AgentNotFoundError",
    "AgentRegistrationError",
    "AgentRegistry",
]
