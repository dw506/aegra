"""Local shell execution adapter."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from time import perf_counter

from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_result import ToolExecutionResult


class LocalShellAdapter:
    """Execute allowlisted commands on the local host."""

    name = "local_shell"

    def __init__(self, allowed_commands: set[str] | None = None) -> None:
        self._allowed_commands = allowed_commands or {
            "echo",
            "python",
            "python3",
            "py",
            "nmap",
            "masscan",
            "httpx",
            "whatweb",
            "sslscan",
            "nuclei",
        }

    def supports(self, plan: ToolPlan) -> bool:
        return plan.adapter == self.name

    def execute(self, plan: ToolPlan) -> ToolExecutionResult:
        command = self._command_argv(plan)
        if not command:
            return self._result(
                plan,
                command=command,
                success=False,
                stderr="tool plan command is empty",
                exit_code="policy_denied",
                metadata={"category": "policy_denied"},
            )

        executable = Path(command[0]).name.lower()
        if executable.endswith(".exe"):
            executable = executable[:-4]
        allowed_commands = self._allowed_commands_for_plan(plan)
        if allowed_commands and executable not in allowed_commands:
            return self._result(
                plan,
                command=command,
                success=False,
                stderr=f"local shell command '{command[0]}' is not allowed",
                exit_code="policy_denied",
                metadata={"category": "policy_denied"},
            )

        started = perf_counter()
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=plan.timeout_seconds,
                cwd=self._string_arg(plan, "cwd"),
                env=self._filtered_env(plan),
            )
        except subprocess.TimeoutExpired as exc:
            duration = perf_counter() - started
            stdout, stdout_truncated = _truncate_bytes(_decode_output(exc.stdout), self._max_bytes(plan, "stdout_max_bytes", 262144))
            stderr, stderr_truncated = _truncate_bytes(_decode_output(exc.stderr), self._max_bytes(plan, "stderr_max_bytes", 65536))
            return self._result(
                plan,
                command=command,
                success=False,
                stdout=stdout,
                stderr=stderr or f"command timed out after {plan.timeout_seconds}s",
                exit_code="timeout",
                metadata={
                    "category": "timeout",
                    "duration_sec": duration,
                    "timed_out": True,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "error_message": f"command timed out after {plan.timeout_seconds}s",
                },
            )
        except FileNotFoundError as exc:
            return self._result(
                plan,
                command=command,
                success=False,
                stderr=str(exc),
                exit_code="command_not_found",
                metadata={"category": "command_not_found", "duration_sec": perf_counter() - started, "error_message": str(exc)},
            )
        except Exception as exc:
            return self._result(
                plan,
                command=command,
                success=False,
                stderr=str(exc),
                exit_code="process_error",
                metadata={"category": "process_error", "duration_sec": perf_counter() - started, "error_message": str(exc)},
            )

        duration = perf_counter() - started
        stdout, stdout_truncated = _truncate_bytes(completed.stdout, self._max_bytes(plan, "stdout_max_bytes", 262144))
        stderr, stderr_truncated = _truncate_bytes(completed.stderr, self._max_bytes(plan, "stderr_max_bytes", 65536))
        acceptable_exit_codes = self._acceptable_exit_codes(plan)
        success = completed.returncode in acceptable_exit_codes
        return self._result(
            plan,
            command=command,
            success=success,
            stdout=stdout,
            stderr=stderr,
            exit_code=completed.returncode,
            metadata={
                "category": "success" if success else "nonzero_exit",
                "duration_sec": duration,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            },
        )

    def _command_argv(self, plan: ToolPlan) -> list[str]:
        if plan.command is None:
            return []
        if isinstance(plan.args.get("argv"), list):
            return [str(part) for part in plan.args["argv"]]
        return shlex.split(plan.command)

    def _result(
        self,
        plan: ToolPlan,
        *,
        command: list[str],
        success: bool,
        exit_code: int | str | None = None,
        stdout: str = "",
        stderr: str = "",
        metadata: dict[str, object] | None = None,
    ) -> ToolExecutionResult:
        payload = {"tool_plan": plan.model_dump(mode="json"), "command": list(command), "attempts": 1}
        if metadata:
            payload.update(metadata)
        return ToolExecutionResult(
            adapter=self.name,
            tool=plan.tool,
            success=success,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            metadata=payload,
        )

    def _allowed_commands_for_plan(self, plan: ToolPlan) -> set[str]:
        raw = plan.args.get("command_allowlist")
        if isinstance(raw, list | set | tuple):
            return {_normalize_command_name(item) for item in raw}
        return set(self._allowed_commands)

    def _acceptable_exit_codes(self, plan: ToolPlan) -> set[int]:
        raw = plan.args.get("acceptable_exit_codes")
        if isinstance(raw, list | set | tuple):
            values: set[int] = set()
            for item in raw:
                try:
                    values.add(int(item))
                except (TypeError, ValueError):
                    continue
            if values:
                return values
        return {0}

    def _filtered_env(self, plan: ToolPlan) -> dict[str, str] | None:
        raw_env = plan.args.get("env")
        if not isinstance(raw_env, dict) or not raw_env:
            return None
        env = {str(key): str(value) for key, value in raw_env.items()}
        raw_allowlist = plan.args.get("env_allowlist")
        if not isinstance(raw_allowlist, list | set | tuple) or not raw_allowlist:
            return {**os.environ, **env}
        allowed = {str(item).upper() for item in raw_allowlist}
        return {**os.environ, **{key: value for key, value in env.items() if key.upper() in allowed}}

    @staticmethod
    def _string_arg(plan: ToolPlan, key: str) -> str | None:
        value = plan.args.get(key)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _max_bytes(plan: ToolPlan, key: str, default: int) -> int:
        try:
            return int(plan.args.get(key, default))
        except (TypeError, ValueError):
            return default


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


def _normalize_command_name(value: object) -> str:
    executable = Path(str(value)).name.lower()
    if executable.endswith(".exe"):
        executable = executable[:-4]
    return executable


__all__ = ["LocalShellAdapter"]
