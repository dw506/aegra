"""Subprocess-based tool execution helpers for worker adapters."""

from __future__ import annotations

import subprocess
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolExecutionSpec(BaseModel):
    """Execution specification for one external command."""

    model_config = ConfigDict(extra="forbid")

    command: list[str] = Field(min_length=1)
    timeout_sec: int = Field(default=30, ge=1)
    retries: int = Field(default=0, ge=0)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    acceptable_exit_codes: set[int] = Field(default_factory=lambda: {0})


class ToolExecutionResult(BaseModel):
    """Normalized subprocess result returned by the tool runner."""

    model_config = ConfigDict(extra="forbid")

    command: list[str]
    attempts: int = Field(default=1, ge=1)
    success: bool = False
    category: str = Field(min_length=1)
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_sec: float = Field(default=0.0, ge=0.0)
    timed_out: bool = False
    error_message: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload fragment for worker evidence."""

        return self.model_dump(mode="json")


class ToolRunner:
    """Execute subprocess commands with timeout, retries, and error normalization."""

    def run(self, spec: ToolExecutionSpec) -> ToolExecutionResult:
        """Run one command and return the last observed normalized result."""

        last_result: ToolExecutionResult | None = None
        for attempt in range(1, spec.retries + 2):
            started = perf_counter()
            try:
                completed = subprocess.run(
                    spec.command,
                    capture_output=True,
                    text=True,
                    timeout=spec.timeout_sec,
                    cwd=spec.cwd,
                    env=(None if not spec.env else spec.env),
                    check=False,
                )
                duration = perf_counter() - started
                success = completed.returncode in spec.acceptable_exit_codes
                last_result = ToolExecutionResult(
                    command=list(spec.command),
                    attempts=attempt,
                    success=success,
                    category="success" if success else "nonzero_exit",
                    exit_code=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    duration_sec=duration,
                )
            except subprocess.TimeoutExpired as exc:
                duration = perf_counter() - started
                last_result = ToolExecutionResult(
                    command=list(spec.command),
                    attempts=attempt,
                    success=False,
                    category="timeout",
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    duration_sec=duration,
                    timed_out=True,
                    error_message=f"command timed out after {spec.timeout_sec}s",
                )
            except FileNotFoundError as exc:
                duration = perf_counter() - started
                last_result = ToolExecutionResult(
                    command=list(spec.command),
                    attempts=attempt,
                    success=False,
                    category="command_not_found",
                    duration_sec=duration,
                    error_message=str(exc),
                )
            except OSError as exc:
                duration = perf_counter() - started
                last_result = ToolExecutionResult(
                    command=list(spec.command),
                    attempts=attempt,
                    success=False,
                    category="process_error",
                    duration_sec=duration,
                    error_message=str(exc),
                )

            if last_result.success:
                return last_result
        assert last_result is not None
        return last_result


__all__ = ["ToolExecutionResult", "ToolExecutionSpec", "ToolRunner"]
