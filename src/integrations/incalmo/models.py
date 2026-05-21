"""Small Incalmo C2 protocol model subset used by Aegra."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CommandStatus(str, Enum):
    """Command lifecycle states returned by Incalmo-like C2 APIs."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class Agent(BaseModel):
    """Remote C2 agent descriptor."""

    model_config = ConfigDict(extra="allow")

    agent_id: str = Field(min_length=1)
    hostname: str | None = None
    address: str | None = None
    platform: str | None = None
    status: str | None = None
    last_seen: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Command(BaseModel):
    """Command submission descriptor."""

    model_config = ConfigDict(extra="allow")

    command_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    command: str = Field(min_length=1)
    payloads: Any = Field(default_factory=dict)
    status: CommandStatus = CommandStatus.PENDING
    metadata: dict[str, Any] = Field(default_factory=dict)


class CommandResult(BaseModel):
    """Command execution result from Incalmo."""

    model_config = ConfigDict(extra="allow")

    command_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    status: CommandStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMActionPayload(BaseModel):
    """Low-level action payload carried through Incalmo command APIs."""

    model_config = ConfigDict(extra="allow")

    action: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
