"""Subprocess-based tool execution helpers for worker adapters."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from time import perf_counter, sleep
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.runtime import RuntimeState
from src.core.runtime.observability import append_audit_log
from src.core.runtime.policy_engine import PolicyEngine


class ToolExecutionSpec(BaseModel):
    """Execution specification for one external command."""

    model_config = ConfigDict(extra="forbid")

    command: list[str] = Field(min_length=1)
    timeout_sec: int = Field(default=30, ge=1)
    retries: int = Field(default=0, ge=0)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    env_allowlist: set[str] = Field(default_factory=set)
    command_allowlist: set[str] = Field(default_factory=set)
    acceptable_exit_codes: set[int] = Field(default_factory=lambda: {0})
    stdout_max_bytes: int = Field(default=262144, ge=0)
    stderr_max_bytes: int = Field(default=65536, ge=0)
    rate_limit_per_sec: float = Field(default=0.0, ge=0.0)
    min_interval_sec: float = Field(default=0.0, ge=0.0)
    rate_limit_key: str | None = None
    policy_metadata: dict[str, Any] = Field(default_factory=dict)
    isolation: dict[str, Any] = Field(default_factory=dict)


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
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    error_message: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload fragment for worker evidence."""

        return self.model_dump(mode="json")


class ToolRunner:
    """Execute subprocess commands with timeout, retries, and error normalization."""

    def __init__(self, *, policy_engine: PolicyEngine | None = None) -> None:
        self._policy_engine = policy_engine
        self._last_started_by_key: dict[str, float] = {}

    def run(self, spec: ToolExecutionSpec, *, runtime_state: RuntimeState | None = None) -> ToolExecutionResult:
        """Run one command and return the last observed normalized result."""

        policy_result = self._check_policy(spec)
        if policy_result is not None:
            self._audit(runtime_state, spec, policy_result)
            return policy_result

        last_result: ToolExecutionResult | None = None
        for attempt in range(1, spec.retries + 2):
            self._apply_rate_limit(spec)
            started = perf_counter()
            try:
                completed = subprocess.run(
                    spec.command,
                    capture_output=True,
                    text=True,
                    timeout=spec.timeout_sec,
                    cwd=spec.cwd,
                    env=self._filtered_env(spec),
                    check=False,
                )
                duration = perf_counter() - started
                stdout, stdout_truncated = _truncate_bytes(completed.stdout, spec.stdout_max_bytes)
                stderr, stderr_truncated = _truncate_bytes(completed.stderr, spec.stderr_max_bytes)
                success = completed.returncode in spec.acceptable_exit_codes
                last_result = ToolExecutionResult(
                    command=list(spec.command),
                    attempts=attempt,
                    success=success,
                    category="success" if success else "nonzero_exit",
                    exit_code=completed.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    duration_sec=duration,
                    stdout_truncated=stdout_truncated,
                    stderr_truncated=stderr_truncated,
                )
            except subprocess.TimeoutExpired as exc:
                duration = perf_counter() - started
                stdout, stdout_truncated = _truncate_bytes(_decode_output(exc.stdout), spec.stdout_max_bytes)
                stderr, stderr_truncated = _truncate_bytes(_decode_output(exc.stderr), spec.stderr_max_bytes)
                last_result = ToolExecutionResult(
                    command=list(spec.command),
                    attempts=attempt,
                    success=False,
                    category="timeout",
                    stdout=stdout,
                    stderr=stderr,
                    duration_sec=duration,
                    timed_out=True,
                    stdout_truncated=stdout_truncated,
                    stderr_truncated=stderr_truncated,
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
                self._audit(runtime_state, spec, last_result)
                return last_result
        assert last_result is not None
        self._audit(runtime_state, spec, last_result)
        return last_result

    def _check_policy(self, spec: ToolExecutionSpec) -> ToolExecutionResult | None:
        command_name = _command_name(spec.command)
        local_allowlist = {item.lower() for item in spec.command_allowlist}
        if local_allowlist and command_name not in local_allowlist:
            return ToolExecutionResult(
                command=list(spec.command),
                success=False,
                category="policy_denied",
                error_message=f"{command_name} outside command allowlist",
            )
        engine = self._policy_engine
        if engine is None:
            return None
        decision = engine.evaluate_tool_policy(
            {
                "kind": spec.policy_metadata.get("kind", command_name),
                "name": spec.policy_metadata.get("name", command_name),
                "operation": spec.policy_metadata.get("operation"),
                "tags": spec.policy_metadata.get("tags", []),
                "command": list(spec.command),
                **spec.policy_metadata,
            }
        )
        if decision.decision == "allow":
            return None
        return ToolExecutionResult(
            command=list(spec.command),
            success=False,
            category="policy_denied",
            error_message=decision.reason,
        )

    def _apply_rate_limit(self, spec: ToolExecutionSpec) -> None:
        min_interval = spec.min_interval_sec
        if spec.rate_limit_per_sec > 0:
            min_interval = max(min_interval, 1.0 / spec.rate_limit_per_sec)
        if min_interval <= 0:
            return
        key = spec.rate_limit_key or _command_name(spec.command)
        now = perf_counter()
        previous = self._last_started_by_key.get(key)
        if previous is not None:
            wait_for = min_interval - (now - previous)
            if wait_for > 0:
                sleep(wait_for)
                now = perf_counter()
        self._last_started_by_key[key] = now

    @staticmethod
    def _filtered_env(spec: ToolExecutionSpec) -> dict[str, str] | None:
        if not spec.env:
            return None
        if not spec.env_allowlist:
            return {str(key): str(value) for key, value in spec.env.items()}
        allowed = {item.upper() for item in spec.env_allowlist}
        return {
            str(key): str(value)
            for key, value in spec.env.items()
            if str(key).upper() in allowed
        }

    @staticmethod
    def _audit(runtime_state: RuntimeState | None, spec: ToolExecutionSpec, result: ToolExecutionResult) -> None:
        if runtime_state is None:
            return
        append_audit_log(
            runtime_state,
            {
                "event_type": "tool_execution",
                "command": list(spec.command),
                "tool": result.to_payload(),
                "isolation": dict(spec.isolation),
            },
        )


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _truncate_bytes(value: str, max_bytes: int) -> tuple[str, bool]:
    raw = value.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return value, False
    truncated = raw[:max_bytes].decode("utf-8", errors="ignore")
    return f"{truncated}...(truncated {len(raw) - max_bytes} bytes)", True


def _command_name(command: Sequence[str]) -> str:
    if not command:
        return ""
    return os.path.basename(str(command[0])).lower()


__all__ = ["ToolExecutionResult", "ToolExecutionSpec", "ToolRunner"]
