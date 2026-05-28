"""Canonical adapter for worker execution results."""

from __future__ import annotations

from src.core.models.events import AgentResultAdapter


class WorkerResultAdapter(AgentResultAdapter):
    """Non-agent name for worker output to canonical task-result adaptation."""


__all__ = ["WorkerResultAdapter"]
