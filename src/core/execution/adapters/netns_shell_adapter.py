"""Network namespace shell execution adapter."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from src.core.execution.pivot_context import PivotExecutionContextResolver
from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_result import ToolExecutionResult
from src.core.models.runtime import RuntimeState


class NetnsShellAdapter:
    """Execute allowlisted commands inside a Linux network namespace."""

    name = "netns_shell"

    def __init__(self, allowed_commands: set[str] | None = None, runtime_state: RuntimeState | None = None) -> None:
        self._allowed_commands = allowed_commands or {"echo", "python", "python3", "py", "curl"}
        self._runtime_state = runtime_state
        self._resolver = PivotExecutionContextResolver()

    def supports(self, plan: ToolPlan) -> bool:
        return plan.adapter == self.name

    def execute(self, plan: ToolPlan) -> ToolExecutionResult:
        context = self._resolver.resolve(plan, self._runtime_state)
        if context.network_namespace is None:
            return self._result(plan, success=False, stderr="network namespace execution requires network_namespace", exit_code="missing_namespace", context=context.model_dump(mode="json"))
        if os.name == "nt":
            return self._result(plan, success=False, stderr="network namespaces are not supported on this platform", exit_code="unsupported_platform", context=context.model_dump(mode="json"))

        command = self._command_argv(plan)
        if not command:
            return self._result(plan, success=False, stderr="tool plan command is empty", exit_code="policy_denied", context=context.model_dump(mode="json"))
        executable = Path(command[0]).name.lower()
        if executable.endswith(".exe"):
            executable = executable[:-4]
        if executable not in self._allowed_commands:
            return self._result(plan, success=False, stderr=f"netns shell command '{command[0]}' is not allowed", exit_code="policy_denied", context=context.model_dump(mode="json"))

        wrapped = [*context.wrapper, *command]
        try:
            completed = subprocess.run(
                wrapped,
                check=False,
                capture_output=True,
                text=True,
                timeout=plan.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return self._result(plan, success=False, stdout=exc.stdout or "", stderr=exc.stderr or f"command timed out after {plan.timeout_seconds}s", exit_code="timeout", context=context.model_dump(mode="json"))
        except Exception as exc:
            return self._result(plan, success=False, stderr=str(exc), exit_code="process_error", context=context.model_dump(mode="json"))
        return self._result(plan, success=completed.returncode == 0, stdout=completed.stdout, stderr=completed.stderr, exit_code=completed.returncode, context=context.model_dump(mode="json"))

    @staticmethod
    def _command_argv(plan: ToolPlan) -> list[str]:
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
        context: dict[str, object] | None = None,
    ) -> ToolExecutionResult:
        metadata: dict[str, object] = {"tool_plan": plan.model_dump(mode="json")}
        if context is not None:
            metadata["execution_context"] = context
        return ToolExecutionResult(
            adapter=self.name,
            tool=plan.tool,
            success=success,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            metadata=metadata,
        )


__all__ = ["NetnsShellAdapter"]
