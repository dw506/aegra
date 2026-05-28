"""Base implementation for bounded stage-level agents."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.core.execution.mcp_client import MCPClient, MCPToolCallResult, UnavailableMCPClient
from src.core.stage.models import StageResult, StageTask, StageType, ToolTrace


class StageToolCall(BaseModel):
    """One MCP tool call requested by a Stage Agent advisor."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    server_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=60, ge=1)


class StageAgentDecision(BaseModel):
    """Decision returned by a Stage Agent advisor for one loop step."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    action: Literal["call_tool", "finish", "need_replan"]
    rationale: str = ""
    tool_call: StageToolCall | None = None
    finish: dict[str, Any] = Field(default_factory=dict)


class StageAgentAdvisor(Protocol):
    """Planner/LLM decision hook for stage agents."""

    def decide(
        self,
        *,
        agent_name: str,
        stage_type: StageType,
        task: StageTask,
        graph_context: dict[str, Any],
        runtime_context: dict[str, Any],
        memory: list[dict[str, Any]],
        available_tools: dict[str, Any],
    ) -> StageAgentDecision | dict[str, Any]:
        """Return the next bounded-loop decision."""


class BaseStageAgent:
    """Bounded ReAct-style stage executor.

    Stage agents may call tools and reason over their results, but they never
    mutate KG, AG, TG or Runtime directly. The only persisted output is a
    StageResult later adapted into the canonical AgentTaskResult protocol.
    """

    stage_type: StageType
    agent_name: str
    tool_categories: frozenset[str] = frozenset()

    def __init__(
        self,
        *,
        advisor: StageAgentAdvisor | None = None,
        mcp_client: MCPClient | None = None,
        default_timeout_seconds: int = 60,
    ) -> None:
        self._advisor = advisor
        self._mcp_client = mcp_client or UnavailableMCPClient()
        self._default_timeout_seconds = default_timeout_seconds

    def run(
        self,
        *,
        task: StageTask,
        graph_context: dict[str, Any],
        runtime_context: dict[str, Any],
        tool_catalog: dict[str, Any],
    ) -> StageResult:
        if task.stage_type != self.stage_type:
            raise ValueError(f"{self.agent_name} cannot execute stage type {task.stage_type.value}")

        memory: list[dict[str, Any]] = []
        tool_trace: list[ToolTrace] = []
        operation_id = str(runtime_context.get("operation_id") or graph_context.get("operation_id") or "operation")
        available_tools = self.filter_tool_catalog(tool_catalog)
        planned_calls = self._planned_tool_calls(task)

        for step in range(task.max_steps):
            if step < len(planned_calls):
                decision = StageAgentDecision(
                    action="call_tool",
                    rationale="executing explicit stage tool_plan",
                    tool_call=planned_calls[step],
                )
            else:
                decision = self._decide(
                    task=task,
                    graph_context=graph_context,
                    runtime_context=runtime_context,
                    memory=memory,
                    available_tools=available_tools,
                )

            if decision.action == "call_tool":
                if decision.tool_call is None:
                    return self._replan_result(
                        operation_id=operation_id,
                        task=task,
                        summary="stage advisor requested a tool call without tool_call details",
                        memory=memory,
                        tool_trace=tool_trace,
                    )
                trace = self._call_tool(step=step, call=decision.tool_call)
                tool_trace.append(trace)
                memory.append({"decision": decision.model_dump(mode="json"), "tool_trace": trace.model_dump(mode="json")})
                continue

            if decision.action == "finish":
                return self._finish_result(
                    operation_id=operation_id,
                    task=task,
                    decision=decision,
                    memory=memory,
                    tool_trace=tool_trace,
                )

            return self._replan_result(
                operation_id=operation_id,
                task=task,
                summary=decision.rationale or "stage agent requested replanning",
                memory=memory,
                tool_trace=tool_trace,
            )

        return self._finish_result(
            operation_id=operation_id,
            task=task,
            decision=StageAgentDecision(
                action="finish",
                rationale="stage loop reached max_steps",
                finish={
                    "status": "partial" if tool_trace else "needs_replan",
                    "summary": f"{self.agent_name} reached max_steps for {task.task_id}",
                },
            ),
            memory=memory,
            tool_trace=tool_trace,
        )

    def filter_tool_catalog(self, tool_catalog: dict[str, Any]) -> dict[str, Any]:
        if not self.tool_categories:
            return dict(tool_catalog)
        filtered: dict[str, Any] = {}
        for server_id, payload in tool_catalog.items():
            tools = payload.get("tools") if isinstance(payload, dict) else None
            if not isinstance(tools, list):
                filtered[server_id] = payload
                continue
            selected = [
                tool
                for tool in tools
                if any(category in str(tool.get("name") or tool).lower() for category in self.tool_categories)
            ]
            if selected:
                filtered[server_id] = {**payload, "tools": selected}
        return filtered

    def _decide(
        self,
        *,
        task: StageTask,
        graph_context: dict[str, Any],
        runtime_context: dict[str, Any],
        memory: list[dict[str, Any]],
        available_tools: dict[str, Any],
    ) -> StageAgentDecision:
        if self._advisor is None:
            return StageAgentDecision(
                action="finish",
                rationale="no stage advisor configured",
                finish={
                    "status": "partial",
                    "summary": f"{self.agent_name} has no advisor or remaining explicit tool_plan",
                    "observations": memory,
                },
            )
        raw = self._advisor.decide(
            agent_name=self.agent_name,
            stage_type=self.stage_type,
            task=task,
            graph_context=graph_context,
            runtime_context=runtime_context,
            memory=memory,
            available_tools=available_tools,
        )
        return raw if isinstance(raw, StageAgentDecision) else StageAgentDecision.model_validate(raw)

    def _call_tool(self, *, step: int, call: StageToolCall) -> ToolTrace:
        timeout = int(call.timeout_seconds or self._default_timeout_seconds)
        raw = self._mcp_client.call_tool(
            server_id=call.server_id,
            tool_name=call.tool_name,
            arguments=dict(call.arguments),
            timeout_seconds=timeout,
        )
        result = raw if isinstance(raw, MCPToolCallResult) else MCPToolCallResult.model_validate(raw)
        return ToolTrace(
            step=step,
            server_id=call.server_id,
            tool_name=call.tool_name,
            arguments=dict(call.arguments),
            success=result.success,
            summary=result.stderr if not result.success else result.stdout[:200],
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            metadata={**dict(result.metadata), "content": result.content},
        )

    def _finish_result(
        self,
        *,
        operation_id: str,
        task: StageTask,
        decision: StageAgentDecision,
        memory: list[dict[str, Any]],
        tool_trace: list[ToolTrace],
    ) -> StageResult:
        payload = dict(decision.finish)
        status = str(payload.get("status") or ("succeeded" if tool_trace and all(item.success for item in tool_trace) else "partial"))
        return StageResult(
            operation_id=operation_id,
            stage_task_id=task.task_id,
            stage_type=task.stage_type,
            agent_name=self.agent_name,
            status=status,  # type: ignore[arg-type]
            summary=str(payload.get("summary") or decision.rationale or f"{self.agent_name} finished {task.task_id}"),
            observations=list(payload.get("observations") or memory),
            evidence=list(payload.get("evidence") or []),
            findings=list(payload.get("findings") or []),
            discovered_entities=list(payload.get("discovered_entities") or []),
            discovered_relations=list(payload.get("discovered_relations") or []),
            capabilities_gained=list(payload.get("capabilities_gained") or []),
            credentials=list(payload.get("credentials") or []),
            sessions=list(payload.get("sessions") or []),
            pivot_routes=list(payload.get("pivot_routes") or []),
            privilege_contexts=list(payload.get("privilege_contexts") or []),
            next_stage_candidates=list(payload.get("next_stage_candidates") or []),
            failed_hypotheses=list(payload.get("failed_hypotheses") or []),
            tool_trace=list(tool_trace),
            runtime_hints=dict(payload.get("runtime_hints") or {}),
            writeback_hints=dict(payload.get("writeback_hints") or {}),
        )

    def _replan_result(
        self,
        *,
        operation_id: str,
        task: StageTask,
        summary: str,
        memory: list[dict[str, Any]],
        tool_trace: list[ToolTrace],
    ) -> StageResult:
        return StageResult(
            operation_id=operation_id,
            stage_task_id=task.task_id,
            stage_type=task.stage_type,
            agent_name=self.agent_name,
            status="needs_replan",
            summary=summary,
            observations=list(memory),
            tool_trace=list(tool_trace),
        )

    @staticmethod
    def _planned_tool_calls(task: StageTask) -> list[StageToolCall]:
        raw_plan = task.required_context.get("tool_plan")
        if not isinstance(raw_plan, list):
            return []
        calls: list[StageToolCall] = []
        for item in raw_plan:
            if not isinstance(item, dict):
                continue
            calls.append(StageToolCall.model_validate(item))
        return calls


__all__ = ["BaseStageAgent", "StageAgentAdvisor", "StageAgentDecision", "StageToolCall"]
