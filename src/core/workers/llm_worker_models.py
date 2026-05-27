"""Models for LLM-driven MCP worker decisions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LLMWorkerDecision(BaseModel):
    """Structured JSON decision returned by the LLM worker advisor."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    action: Literal["call_mcp_tool", "defer", "failed"]
    server_id: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    summary: str = Field(default="LLM worker decision")
    expected_evidence: list[str] = Field(default_factory=list)
    risk_assessment: str | None = None
    writeback_hints: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_tool_for_call(self) -> "LLMWorkerDecision":
        if self.action == "call_mcp_tool":
            if not self.server_id or not self.tool_name:
                raise ValueError("call_mcp_tool requires server_id and tool_name")
        return self


__all__ = ["LLMWorkerDecision"]
