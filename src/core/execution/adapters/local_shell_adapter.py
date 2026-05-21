"""Local shell execution adapter."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_result import ToolExecutionResult


class LocalShellAdapter:
    """Execute allowlisted commands on the local host."""

    name = "local_shell"

    def __init__(self, allowed_commands: set[str] | None = None) -> None:
        self._allowed_commands = allowed_commands or {"echo", "python", "python3", "py"}

    def supports(self, plan: ToolPlan) -> bool:
        return plan.adapter == self.name

    def execute(self, plan: ToolPlan) -> ToolExecutionResult:
        command = self._command_argv(plan)
        if not command:
            return self._result(plan, success=False, stderr="tool plan command is empty", exit_code="policy_denied")

        executable = Path(command[0]).name.lower()
        if executable.endswith(".exe"):
            executable = executable[:-4]
        if executable not in self._allowed_commands:
            return self._result(
                plan,
                success=False,
                stderr=f"local shell command '{command[0]}' is not allowed",
                exit_code="policy_denied",
            )

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=plan.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return self._result(
                plan,
                success=False,
                stdout=exc.stdout or "",
                stderr=exc.stderr or f"command timed out after {plan.timeout_seconds}s",
                exit_code="timeout",
                metadata={"timed_out": True},
            )
        except Exception as exc:
            return self._result(plan, success=False, stderr=str(exc), exit_code="process_error")

        return self._result(
            plan,
            success=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
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
        success: bool,
        exit_code: int | str | None = None,
        stdout: str = "",
        stderr: str = "",
        metadata: dict[str, object] | None = None,
    ) -> ToolExecutionResult:
        payload = {"tool_plan": plan.model_dump(mode="json")}
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


__all__ = ["LocalShellAdapter"]
