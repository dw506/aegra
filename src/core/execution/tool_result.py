"""Canonical result returned by execution adapters."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolExecutionResult(BaseModel):
    """Adapter-neutral tool execution result.

    This model does not write graph state directly. Parser and result-applier
    layers decide how to interpret it.
    """

    model_config = ConfigDict(extra="forbid")

    adapter: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    success: bool
    exit_code: int | str | None = None
    stdout: str = ""
    stderr: str = ""
    command_id: str | None = None
    payload_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["ToolExecutionResult"]
