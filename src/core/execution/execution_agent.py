"""Single public execution module for capability-scoped execution rounds."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.agents.packy_llm import PackyLLMClient, PackyLLMError
from src.core.execution.mcp_client import MCPClient, MCPToolCallResult, UnavailableMCPClient
from src.core.models.graph_common import utc_now
from src.core.runtime.tool_trace_fact_extractor import ToolTraceFactExtractor
from src.core.runtime.txt_trace_logger import TxtTraceLogger
from src.core.execution.models import (
    ExtractedFact,
    RoundDirective,
    RoundResult,
    ExecutionRequest,
    ExecutionResult,
    ToolTrace,
)


class ExecutionAgent:
    """Run directives and execution requests through one bounded tool loop."""

    agent_name = "execution_agent"

    def __init__(
        self,
        agent: Any | None = None,
        *,
        llm_client: PackyLLMClient | None = None,
        mcp_client: MCPClient | None = None,
        default_timeout_seconds: int = 120,
        trace_logger_factory: type[TxtTraceLogger] = TxtTraceLogger,
    ) -> None:
        self._agent = agent or _ExecutionLoop(
            agent_name=self.agent_name,
            llm_client=llm_client,
            mcp_client=mcp_client,
            default_timeout_seconds=default_timeout_seconds,
        )
        self._trace_logger_factory = trace_logger_factory

    @classmethod
    def from_clients(
        cls,
        *,
        llm_client: Any = None,
        mcp_client: Any = None,
        default_timeout_seconds: int = 120,
        trace_logger_factory: type[TxtTraceLogger] = TxtTraceLogger,
    ) -> "ExecutionAgent":
        """Build the default execution agent."""

        return cls(
            llm_client=llm_client,
            mcp_client=mcp_client,
            default_timeout_seconds=default_timeout_seconds,
            trace_logger_factory=trace_logger_factory,
        )

    def capability_summary(self) -> list[dict[str, str]]:
        agent = self._agent
        return [
            {
                "agent_name": agent.agent_name,
                "context_builder": getattr(agent, "context_builder_name", "execution_agent_context"),
            }
        ]

    def run(
        self,
        directive: RoundDirective | ExecutionRequest,
        *,
        graph_summary: dict[str, Any] | None = None,
        graph_history: dict[str, Any] | None = None,
        runtime_context: dict[str, Any] | None = None,
        policy_context: dict[str, Any] | None = None,
        mcp_tool_catalog: dict[str, Any] | None = None,
        pivot_routes: list[dict[str, Any]] | None = None,
        sessions: list[dict[str, Any]] | None = None,
    ) -> RoundResult | ExecutionResult:
        """Execute one capability round through the single execution agent."""

        agent = self._agent
        if isinstance(directive, ExecutionRequest):
            return agent.run(directive)
        # Pass the FULL catalog (every in-scope tool stays callable); the planner's
        # allowed_tools are attached only as a focus hint. The real authorization
        # boundary is scope policy, not this list.
        catalog = dict(mcp_tool_catalog or {})
        if directive.allowed_tools:
            catalog["recommended_tool_names"] = list(directive.allowed_tools)
        request = ExecutionRequest(
            operation_id=directive.operation_id,
            cycle_index=directive.cycle_index,
            agent_name=agent.agent_name,
            capability=directive.capability,
            objective=directive.objective,
            target_refs=list(directive.target_refs),
            required_context={
                **dict(directive.required_context),
                "round_capability": directive.capability,
                "tool_hints": list(directive.tool_hints),
            },
            success_criteria=[directive.success_hint] if directive.success_hint else [],
            risk_level=directive.risk_level,
            max_steps=directive.max_tools,
            graph_summary=dict(graph_summary or {}),
            graph_history=dict(graph_history or {}),
            runtime_context=dict(runtime_context or {}),
            policy_context=dict(policy_context or {}),
            mcp_tool_catalog=catalog,
            allowed_tool_names=list(directive.allowed_tools),
            pivot_routes=list(pivot_routes or []),
            sessions=list(sessions or []),
        )
        execution_result = agent.run(request)
        # Capability is authoritative from the directive; stamp it on the result
        # so the result tier reads it directly.
        execution_result.capability = directive.capability
        round_result = self._round_result(directive=directive, execution_result=execution_result)
        self._write_round_log(directive=directive, execution_result=execution_result, round_result=round_result)
        if round_result.log_ref:
            execution_result.runtime_hints = {
                **dict(execution_result.runtime_hints),
                "round_log_ref": round_result.log_ref,
                "cycle_index": directive.cycle_index,
                "capability": directive.capability,
            }
        return round_result

    @staticmethod
    def _round_result(*, directive: RoundDirective, execution_result: ExecutionResult) -> RoundResult:
        facts: list[ExtractedFact] = []
        extractor = ToolTraceFactExtractor()
        for extraction in extractor.extract_all(execution_result.tool_trace):
            for fact in extraction.facts:
                facts.append(
                    ExtractedFact(
                        fact_id=f"tool-fact-{extraction.trace_id}-{fact.entity_type}-{fact.label[:40]}",
                        entity_type=fact.entity_type,
                        label=fact.label,
                        properties=dict(fact.properties),
                        source_tool=fact.source_tool,
                        trace_id=extraction.trace_id,
                        confidence=fact.confidence,
                    )
                )
        return RoundResult(
            operation_id=execution_result.operation_id,
            cycle_index=directive.cycle_index,
            capability=directive.capability,
            tool_traces=list(execution_result.tool_trace),
            extracted_facts=facts,
            raw_summary=execution_result.summary,
            objective_met=execution_result.status in {"success", "succeeded"},
            execution_result=execution_result,
        )

    def _write_round_log(
        self,
        *,
        directive: RoundDirective,
        execution_result: ExecutionResult,
        round_result: RoundResult,
    ) -> None:
        logger = self._trace_logger_factory(
            directive.operation_id,
            log_dir="var/runtime",
            filename=f"round-{directive.cycle_index}.txt",
            operation_subdir=True,
        )
        logger.write_block(
            "ROUND_DIRECTIVE",
            "execution directive",
            {
                "cycle_index": directive.cycle_index,
                "capability": directive.capability,
                "objective": directive.objective,
                "target_refs": [item.model_dump(mode="json") for item in directive.target_refs],
                "allowed_tools": list(directive.allowed_tools),
                "max_tools": directive.max_tools,
            },
        )
        logger.write_block(
            "ROUND_RESULT",
            "execution result",
            {
                "status": execution_result.status,
                "objective_met": round_result.objective_met,
                "summary": execution_result.summary,
                "tool_traces": [
                    {
                        "trace_id": trace.trace_id,
                        "step": trace.step,
                        "tool_name": trace.tool_name,
                        "success": trace.success,
                        "raw_output_ref": trace.raw_output_ref,
                        "summary": trace.summary,
                    }
                    for trace in execution_result.tool_trace
                ],
                "extracted_fact_count": len(round_result.extracted_facts),
            },
        )
        round_result.log_ref = str(logger.path)


__all__ = [
    "ExecutionAgent",
]


class _ExecutionToolCall(BaseModel):
    """One MCP tool call selected by the execution loop."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    server_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=120, ge=1)


class _ExecutionLoop:
    """Autonomous bounded execution loop used by the single ExecutionAgent.

    Capability-agnostic: the round capability is a per-request tag
    (``ExecutionRequest.capability``), not a stage binding on the agent.
    """

    agent_name: str = "execution_agent"
    role_prompt: str = ""
    context_builder_name: str = "execution_agent_context"

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        role_prompt: str | None = None,
        llm_client: PackyLLMClient | None = None,
        mcp_client: MCPClient | None = None,
        operation_logger: TxtTraceLogger | None = None,
        default_timeout_seconds: int = 120,
        **_: Any,
    ) -> None:
        resolved_name = agent_name or getattr(self, "agent_name", None)
        if resolved_name is None:
            raise ValueError("execution loop requires agent_name")
        self.agent_name = str(resolved_name)
        if role_prompt is not None:
            self.role_prompt = role_prompt
        self._llm_client = llm_client
        self._mcp_client = mcp_client or UnavailableMCPClient()
        self._operation_logger = operation_logger
        self._default_timeout_seconds = default_timeout_seconds

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        logger = self._logger(request.operation_id)
        logger.write_header(
            f"Operation: {request.operation_id}",
            {"cycle_index": request.cycle_index, "agent_name": request.agent_name, "capability": request.capability},
        )
        logger.write_block(
            "CYCLE_START",
            "execution round started",
            {
                "cycle_index": request.cycle_index,
                "agent_name": request.agent_name,
                "capability": request.capability,
                "objective": request.objective,
                "task_brief": request.task_brief,
                "max_steps": request.max_steps,
            },
        )

        if self._llm_client is None:
            result = self._replan_result(request, "llm_client unavailable", [], [])
            self._log_execution_finish(logger, request, result)
            return result

        memory: list[dict[str, Any]] = []
        tool_traces: list[ToolTrace] = []
        messages = self._build_messages(request, memory)
        for step in range(1, request.max_steps + 1):
            try:
                raw_text = self._call_llm(messages)
            except PackyLLMError as exc:
                logger.write_block("ERROR", "llm call failed", {"phase": "llm_decision", "type": type(exc).__name__, "message": str(exc)})
                result = self._replan_result(request, f"llm call failed: {exc}", memory, tool_traces)
                self._log_execution_finish(logger, request, result)
                return result

            decision = self._extract_json_object(raw_text)
            if decision is None:
                logger.write_block("LLM_DECISION", "invalid json", {"step_index": step, "raw_response": raw_text[:2000]})
                result = self._replan_result(request, "LLM decision JSON parse failed", memory, tool_traces)
                self._log_execution_finish(logger, request, result)
                return result

            action = str(decision.get("action") or "")
            logger.write_block(
                "LLM_DECISION",
                "execution agent decision",
                {
                    "cycle_index": request.cycle_index,
                    "agent_name": request.agent_name,
                    "capability": request.capability,
                    "step_index": step,
                    "action": action,
                    "reasoning_summary": decision.get("reasoning_summary") or decision.get("rationale"),
                    "decision_json": decision,
                },
            )

            if action == "call_mcp_tool":
                trace = self._call_mcp_tool(step=step, request=request, decision=decision, logger=logger)
                tool_traces.append(trace)
                memory.append({"decision": decision, "tool_trace": trace.model_dump(mode="json")})
                messages = self._build_messages(request, memory)
                continue

            if action == "finish":
                result = self._finish_result(request, decision, memory, tool_traces)
                self._log_execution_finish(logger, request, result)
                return result

            if action == "need_replan":
                summary = str(decision.get("summary") or decision.get("replan_reason") or "execution agent requested replanning")
                result = self._replan_result(request, summary, memory, tool_traces, decision=decision)
                self._log_execution_finish(logger, request, result)
                return result

            result = self._replan_result(request, f"unsupported LLM action: {action}", memory, tool_traces)
            self._log_execution_finish(logger, request, result)
            return result

        result = self._partial_result_from_tool_memory(request, memory, tool_traces)
        self._log_execution_finish(logger, request, result)
        return result

    def _partial_result_from_tool_memory(
        self,
        request: ExecutionRequest,
        memory: list[dict[str, Any]],
        tool_traces: list[ToolTrace],
    ) -> ExecutionResult:
        evidence = self._tool_trace_evidence(tool_traces)
        observations = list(memory)
        if tool_traces:
            observations.append(
                {
                    "category": "execution_tool_trace_summary",
                    "summary": f"{self.agent_name} reached max_steps after {len(tool_traces)} tool call(s); preserving tool evidence for writeback.",
                    "tool_count": len(tool_traces),
                    "successful_tool_count": sum(1 for trace in tool_traces if trace.success),
                    "parsed_outputs": [trace.parsed_output for trace in tool_traces if trace.parsed_output],
                }
            )
        return ExecutionResult(
            operation_id=request.operation_id,
            execution_id=self._request_id(request),
            capability=request.capability,
            agent_name=self.agent_name,
            status="partial",
            summary=f"{self.agent_name} reached max_steps for {self._request_id(request)}",
            observations=observations,
            evidence=evidence,
            evidence_refs=[item["payload_ref"] for item in evidence if item.get("payload_ref")],
            tool_trace=list(tool_traces),
            tool_traces=list(tool_traces),
            runtime_hints={"cycle_index": request.cycle_index, "max_steps_exhausted": True},
        )

    @staticmethod
    def _tool_trace_evidence(tool_traces: list[ToolTrace]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for trace in tool_traces:
            if not trace.success:
                continue
            payload_ref = trace.raw_output_ref or f"runtime://tool-trace/{trace.trace_id}"
            evidence.append(
                {
                    "evidence_id": f"evidence::{trace.trace_id}",
                    "kind": "execution_tool_trace",
                    "summary": trace.summary or f"{trace.server_id}.{trace.tool_name}",
                    "payload_ref": payload_ref,
                    "tool_output_ref": payload_ref,
                    "tool": trace.tool_name,
                    "server_id": trace.server_id,
                    "parsed_output": trace.parsed_output,
                    "confidence": 0.8,
                }
            )
        return evidence

    def _call_mcp_tool(
        self,
        *,
        step: int,
        request: ExecutionRequest,
        decision: dict[str, Any],
        logger: TxtTraceLogger,
    ) -> ToolTrace:
        arguments = dict(decision.get("arguments") or decision.get("input") or {})
        call = _ExecutionToolCall(
            server_id=str(decision.get("server_id") or decision.get("server") or self._default_server_id(request.mcp_tool_catalog)),
            tool_name=str(decision.get("tool_name") or decision.get("tool") or ""),
            arguments=arguments,
            timeout_seconds=int(arguments.get("timeout_seconds") or self._default_timeout_seconds),
        )
        call = self._normalize_tool_call_arguments(call=call, request=request)
        call = self._with_trace_arguments(call=call, request=request, step=step)
        server_metadata = self._lookup_server(request.mcp_tool_catalog, server_id=call.server_id)
        tool_metadata = self._lookup_tool(request.mcp_tool_catalog, server_id=call.server_id, tool_name=call.tool_name)
        logger.write_block(
            "TOOL_CALL",
            "mcp tool call",
            {
                "cycle_index": request.cycle_index,
                "agent_name": request.agent_name,
                "capability": request.capability,
                "step_index": step,
                "server_id": call.server_id,
                "tool_name": call.tool_name,
                "arguments": call.arguments,
                "timeout_seconds": call.timeout_seconds,
            },
        )
        if server_metadata and server_metadata.get("available") is False:
            trace = ToolTrace(
                step=step,
                server_id=call.server_id,
                tool_name=call.tool_name or "unknown_tool",
                arguments=dict(call.arguments),
                success=False,
                summary="MCP tool server is unavailable",
                stderr=str(server_metadata.get("error") or "MCP tool server is unavailable"),
                exit_code="tool_server_unavailable",
                ended_at=utc_now().isoformat(),
                policy_check={
                    "allowed": False,
                    "reason": "MCP tool server is unavailable",
                    "metadata": {"catalog_enforced": True, "server_available": False},
                },
                metadata={"content": {"success": False, "exit_code": "tool_server_unavailable"}},
            )
            self._log_tool_result(logger, trace)
            return trace
        if not tool_metadata:
            trace = ToolTrace(
                step=step,
                server_id=call.server_id,
                tool_name=call.tool_name or "unknown_tool",
                arguments=dict(call.arguments),
                success=False,
                summary="tool is not present in the supplied MCP tool catalog",
                stderr="tool is not present in the supplied MCP tool catalog",
                exit_code="tool_not_in_catalog",
                ended_at=utc_now().isoformat(),
                policy_check={
                    "allowed": False,
                    "reason": "tool is not present in the supplied MCP tool catalog",
                    "metadata": {"catalog_enforced": True},
                },
                metadata={"content": {"success": False, "exit_code": "tool_not_in_catalog"}},
            )
            self._log_tool_result(logger, trace)
            return trace

        # Hard scope gate: reject any tool call that targets a denied host
        # (control-plane infrastructure / out of authorized scope). Unlike the
        # advisory policy hints, this is enforced before the tool ever runs.
        blocked_hosts = [str(h).strip() for h in (request.policy_context.get("blocked_hosts") or []) if str(h).strip()]
        if blocked_hosts:
            arg_blob = " ".join(str(value) for value in call.arguments.values())
            blocked_hit = next((host for host in blocked_hosts if host in arg_blob), None)
            if blocked_hit is not None:
                trace = ToolTrace(
                    step=step,
                    server_id=call.server_id,
                    tool_name=call.tool_name or "unknown_tool",
                    arguments=dict(call.arguments),
                    success=False,
                    summary=f"target host {blocked_hit} is out of authorized scope (blocked_hosts)",
                    stderr=f"blocked_host: {blocked_hit}",
                    exit_code="target_out_of_scope",
                    ended_at=utc_now().isoformat(),
                    policy_check={
                        "allowed": False,
                        "reason": f"target host {blocked_hit} is in policy.blocked_hosts (out of scope)",
                        "metadata": {"scope_enforced": True, "blocked_host": blocked_hit},
                    },
                    metadata={"content": {"success": False, "exit_code": "target_out_of_scope"}},
                )
                self._log_tool_result(logger, trace)
                return trace

        raw = self._mcp_client.call_tool(
            server_id=call.server_id,
            tool_name=call.tool_name,
            arguments=dict(call.arguments),
            timeout_seconds=call.timeout_seconds,
        )
        result = raw if isinstance(raw, MCPToolCallResult) else MCPToolCallResult.model_validate(raw)
        original_policy = self._original_policy_decision(call=call, policy_context=request.policy_context)
        parsed_output = dict(result.metadata.get("parsed_output") or {})
        if not parsed_output and isinstance(result.content, dict) and isinstance(result.content.get("parsed"), dict):
            parsed_output = dict(result.content["parsed"])
        raw_output_ref = str(
            result.metadata.get("raw_output_ref")
            or (result.content.get("raw_output_ref") if isinstance(result.content, dict) else "")
            or ""
        )
        trace = ToolTrace(
            step=step,
            server_id=call.server_id,
            tool_name=call.tool_name,
            tool_category=str(tool_metadata.get("category") or ""),
            input_summary=self._summarize_arguments(call.arguments),
            arguments=dict(call.arguments),
            success=result.success,
            summary=result.stderr if not result.success else result.stdout[:200],
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            ended_at=utc_now().isoformat(),
            policy_check={
                "allowed": True,
                "reason": "MCP policy is audit-only in authorized lab mode; execution continued.",
                "metadata": {
                    "policy_audit_only": True,
                    "original_allowed": original_policy["allowed"],
                    "original_reason": original_policy["reason"],
                    "mcp_server_id": call.server_id,
                    "mcp_tool_name": call.tool_name,
                    "adapter": "mcp",
                },
            },
            raw_output_ref=raw_output_ref,
            parsed_output=parsed_output,
            metadata={**dict(result.metadata), "content": result.content},
        )
        self._log_tool_result(logger, trace)
        return trace

    def _build_messages(self, request: ExecutionRequest, memory: list[dict[str, Any]]) -> list[dict[str, str]]:
        system_prompt = (
            "You are Aegra's bounded ExecutionAgent for an authorized assessment. "
            "Return strict JSON only. Allowed actions are call_mcp_tool, finish, need_replan. "
            "Call only tools present in mcp_tool_catalog (every tool listed there is in-scope and callable). "
            "Prefer the tools in recommended_tool_names for this round's capability, but you MAY call any "
            "other catalog tool when it advances the objective — the authorization boundary is scope policy, "
            "not the tool menu. Prefer argv for run_command. "
            "Do not invent facts; base findings on KG/Runtime/evidence/tool results."
        )
        context = {
            "agent_name": request.agent_name,
            "capability": request.capability,
            "role_prompt": self.role_prompt,
            "planner_objective": request.objective,
            "task_brief": request.task_brief,
            "target_refs": [ref.model_dump(mode="json") for ref in request.target_refs],
            "success_criteria": request.success_criteria,
            "graph_summary": request.graph_summary,
            "graph_history": request.graph_history,
            "runtime_context": request.runtime_context,
            "policy_context": request.policy_context,
            "mcp_tool_catalog": request.mcp_tool_catalog,
            "stage_memory": memory[-10:],
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
        ]

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        system_prompt = "\n".join(item["content"] for item in messages if item.get("role") == "system")
        user_prompt = "\n\n".join(item["content"] for item in messages if item.get("role") != "system")
        response = self._llm_client.complete_chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)  # type: ignore[union-attr]
        return response.text

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        candidates: list[str] = []
        if stripped.startswith("{"):
            candidates.append(stripped)
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            candidates.append(match.group(1))
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            candidates.append(match.group(1))
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    payload, _ = json.JSONDecoder().raw_decode(candidate)
                except json.JSONDecodeError:
                    continue
            if isinstance(payload, dict):
                return payload
        return None

    def _finish_result(
        self,
        request: ExecutionRequest,
        payload: dict[str, Any],
        memory: list[dict[str, Any]],
        tool_traces: list[ToolTrace],
    ) -> ExecutionResult:
        payload = self._normalized_finish_payload(payload)
        observations = list(payload.get("observations") or memory)
        observations.extend(self._structured_finish_observations(payload))
        findings = self._normalized_findings(payload)
        status = self._normalized_status(payload.get("status"))
        if self._empty_success_needs_replan(
            status=status,
            summary=payload.get("summary"),
            observations=observations,
            findings=findings,
            evidence=payload.get("evidence"),
            evidence_refs=payload.get("evidence_refs"),
            tool_traces=tool_traces,
        ):
            status = "needs_replan"
            payload.setdefault("replan_recommendation", "stage returned no tool results, evidence, findings, or structured output")
        try:
            return self._build_execution_result_from_finish(
                request=request,
                payload=payload,
                observations=observations,
                findings=findings,
                status=status,
                tool_traces=tool_traces,
            )
        except (TypeError, ValueError, ValidationError) as exc:
            repaired = self._repair_finish_payload(request=request, payload=payload, validation_error=str(exc))
            if repaired is not None:
                repaired = self._normalized_finish_payload(repaired)
                repaired_observations = list(repaired.get("observations") or observations or memory)
                repaired_observations.extend(self._structured_finish_observations(repaired))
                repaired_findings = self._normalized_findings(repaired)
                try:
                    return self._build_execution_result_from_finish(
                        request=request,
                        payload=repaired,
                        observations=repaired_observations,
                        findings=repaired_findings,
                        status=self._normalized_status(repaired.get("status") or status),
                        tool_traces=tool_traces,
                    )
                except (TypeError, ValueError, ValidationError):
                    pass
            return self._replan_result(request, f"invalid finish payload after structure repair: {exc}", memory, tool_traces)

    def _build_execution_result_from_finish(
        self,
        *,
        request: ExecutionRequest,
        payload: dict[str, Any],
        observations: list[dict[str, Any]],
        findings: list[dict[str, Any]],
        status: str,
        tool_traces: list[ToolTrace],
    ) -> ExecutionResult:
        runtime_hints = dict(payload.get("runtime_hints") or {})
        runtime_hints.setdefault("cycle_index", request.cycle_index)
        return ExecutionResult(
            operation_id=request.operation_id,
            execution_id=self._request_id(request),
            capability=request.capability,
            agent_name=self.agent_name,
            status=status,  # type: ignore[arg-type]
            summary=str(payload.get("summary") or f"{self.agent_name} finished"),
            observations=observations,
            evidence=self._normalized_evidence(payload),
            findings=findings,
            discovered_entities=self._normalized_dict_list(payload.get("discovered_entities"), default_kind="entity"),
            discovered_relations=self._normalized_dict_list(payload.get("discovered_relations"), default_kind="relation"),
            capabilities_gained=list(payload.get("capabilities_gained") or []),
            credentials=list(payload.get("credentials") or []),
            sessions=list(payload.get("sessions") or []),
            pivot_routes=list(payload.get("pivot_routes") or []),
            next_capability_candidates=list(payload.get("next_capability_candidates") or []),
            failed_hypotheses=list(payload.get("failed_hypotheses") or []),
            handoff_suggestion=payload.get("handoff_suggestion"),
            next_capability_suggestion=payload.get("next_capability_suggestion"),
            evidence_refs=self._normalized_evidence_refs(payload.get("evidence_refs")),
            tool_trace=list(tool_traces),
            tool_traces=list(tool_traces),
            confidence=float(payload.get("confidence") or 0.5),
            replan_recommendation=payload.get("replan_recommendation"),
            retry_recommendation=payload.get("retry_recommendation"),
            runtime_hints=runtime_hints,
            writeback_hints=dict(payload.get("writeback_hints") or {}),
        )

    def _repair_finish_payload(
        self,
        *,
        request: ExecutionRequest,
        payload: dict[str, Any],
        validation_error: str,
    ) -> dict[str, Any] | None:
        if self._llm_client is None:
            return None
        repair_prompt = (
            "Repair this ExecutionResult finish payload into strict JSON only. "
            "Preserve facts; do not add new findings. Acceptable wrappers are result, execution_result, or direct fields. "
            "If observations are strings, convert each to {\"type\":\"note\",\"detail\":\"...\"}.\n\n"
            f"operation_id={request.operation_id}\n"
            f"execution_id={self._request_id(request)}\n"
            f"capability={request.capability}\n"
            f"agent_name={self.agent_name}\n"
            f"Validation error: {validation_error}\n\n"
            f"Payload:\n{json.dumps(payload, ensure_ascii=False, default=str)[:6000]}"
        )
        try:
            response = self._llm_client.complete_chat(
                system_prompt="Return repaired ExecutionResult finish JSON only.",
                user_prompt=repair_prompt,
                temperature=0.0,
            )
        except PackyLLMError:
            return None
        repaired = self._extract_json_object(response.text)
        return repaired if isinstance(repaired, dict) else None

    @staticmethod
    def _normalized_finish_payload(payload: dict[str, Any]) -> dict[str, Any]:
        nested = payload.get("result") or payload.get("execution_result") or payload.get("data") or payload.get("output")
        if not isinstance(nested, dict):
            merged = dict(payload)
        else:
            merged = {key: value for key, value in payload.items() if key not in {"result", "execution_result", "data", "output"}}
            merged.update(nested)
        summary = merged.get("summary")
        if isinstance(summary, str):
            stripped = summary.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    for key, value in parsed.items():
                        merged.setdefault(key, value)
                    if not isinstance(parsed.get("summary"), str):
                        merged["summary"] = parsed.get("status") or "structured stage output"
        return merged

    @classmethod
    def _normalized_findings(cls, payload: dict[str, Any]) -> list[dict[str, Any]]:
        findings = cls._normalized_dict_list(payload.get("findings"), default_kind="stage_finding")
        for key in ("candidate_findings", "findings_candidates"):
            for item in cls._normalized_dict_list(payload.get(key), default_kind="candidate_finding"):
                normalized = dict(item)
                normalized.setdefault("kind", normalized.get("type") or "candidate_finding")
                normalized.setdefault(
                    "summary",
                    normalized.get("statement") or normalized.get("title") or normalized.get("description") or str(item),
                )
                findings.append(normalized)
        analysis = payload.get("analysis")
        if isinstance(analysis, dict):
            for item in cls._normalized_dict_list(analysis.get("candidate_findings"), default_kind="candidate_finding"):
                normalized = dict(item)
                normalized.setdefault("kind", normalized.get("type") or "candidate_finding")
                normalized.setdefault(
                    "summary",
                    normalized.get("statement") or normalized.get("title") or normalized.get("description") or str(item),
                )
                findings.append(normalized)
        return findings

    @staticmethod
    def _empty_success_needs_replan(
        *,
        status: str,
        summary: Any,
        observations: list[dict[str, Any]],
        findings: list[dict[str, Any]],
        evidence: Any,
        evidence_refs: Any,
        tool_traces: list[ToolTrace],
    ) -> bool:
        if status != "succeeded":
            return False
        if tool_traces or observations or findings:
            return False
        if evidence not in (None, "", [], {}) or evidence_refs not in (None, "", [], {}):
            return False
        text = str(summary or "").strip().lower()
        return not text or text.endswith(" finished")

    @staticmethod
    def _structured_finish_observations(payload: dict[str, Any]) -> list[dict[str, Any]]:
        known_fields = {
            "action",
            "status",
            "summary",
            "observations",
            "evidence",
            "evidence_refs",
            "findings",
            "discovered_entities",
            "discovered_relations",
            "capabilities_gained",
            "credentials",
            "sessions",
            "pivot_routes",
            "next_capability_candidates",
            "handoff_suggestion",
            "confidence",
            "replan_recommendation",
            "result",
            "execution_result",
            "data",
            "output",
            "runtime_hints",
            "writeback_hints",
        }
        structured = {
            key: value
            for key, value in payload.items()
            if key not in known_fields and value not in (None, "", [], {})
        }
        if not structured:
            return []
        return [
            {
                "category": "stage_structured_output",
                "summary": str(payload.get("summary") or "Structured stage output"),
                **structured,
            }
        ]

    @staticmethod
    def _normalized_status(value: Any) -> str:
        status = str(value or "succeeded").strip().lower()
        return {
            "completed": "succeeded",
            "complete": "succeeded",
            "ok": "succeeded",
            "needs_replan": "needs_replan",
            "need_replan": "needs_replan",
        }.get(status, status)

    @classmethod
    def _normalized_evidence(cls, payload: dict[str, Any]) -> list[dict[str, Any]]:
        evidence = cls._normalized_dict_list(payload.get("evidence"), default_kind="stage_evidence")
        if evidence:
            return evidence
        refs = payload.get("evidence_refs")
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(refs if isinstance(refs, list) else []):
            if isinstance(item, dict):
                normalized.append(
                    {
                        "evidence_id": str(item.get("evidence_id") or item.get("id") or f"stage-evidence-{index}"),
                        "kind": str(item.get("kind") or "stage_evidence"),
                        "summary": str(item.get("summary") or item.get("description") or item),
                        "payload_ref": str(item.get("payload_ref") or item.get("raw_output_ref") or item.get("evidence_id") or ""),
                    }
                )
            elif item:
                normalized.append(
                    {
                        "evidence_id": f"stage-evidence-{index}",
                        "kind": "stage_evidence_ref",
                        "summary": str(item),
                        "payload_ref": str(item),
                    }
                )
        return normalized

    @staticmethod
    def _normalized_dict_list(value: Any, *, default_kind: str) -> list[dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [dict(value)]
        if isinstance(value, list):
            normalized: list[dict[str, Any]] = []
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    normalized.append(dict(item))
                elif item:
                    normalized.append(
                        {
                            "type": default_kind,
                            "summary": str(item),
                            "value": item,
                            "index": index,
                        }
                    )
            return normalized
        return [{"type": default_kind, "summary": str(value), "value": value}]

    @staticmethod
    def _normalized_evidence_refs(value: Any) -> list[str]:
        refs: list[str] = []
        for item in value if isinstance(value, list) else []:
            if isinstance(item, dict):
                ref = item.get("payload_ref") or item.get("raw_output_ref") or item.get("evidence_id") or item.get("id")
                if ref:
                    refs.append(str(ref))
            elif item:
                refs.append(str(item))
        return refs

    def _replan_result(
        self,
        request: ExecutionRequest,
        summary: str,
        memory: list[dict[str, Any]],
        tool_traces: list[ToolTrace],
        *,
        decision: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Return needs_replan unless successful ToolTraces exist, in which case return partial.

        If any tool trace has success=True, the tool facts must be preserved even
        when LLM post-processing failed. In this case we return status="partial"
        so ResultApplier can still extract and write back confirmed facts.
        """
        evidence = self._tool_trace_evidence(tool_traces)
        observations = list(memory)
        successful_traces = [t for t in tool_traces if t.success]
        has_successful_tool = len(successful_traces) > 0

        if tool_traces:
            observations.append(
                {
                    "category": "execution_tool_trace_summary",
                    "summary": f"{self.agent_name} {'has successful tools, preserving facts' if has_successful_tool else 'requested replanning'} after {len(tool_traces)} tool call(s).",
                    "tool_count": len(tool_traces),
                    "successful_tool_count": len(successful_traces),
                    "parsed_outputs": [trace.parsed_output for trace in tool_traces if trace.parsed_output],
                }
            )

        # If tools succeeded but LLM post-processing failed, return partial (not needs_replan)
        # so that deterministic fact extraction can still run.
        if has_successful_tool:
            return ExecutionResult(
                operation_id=request.operation_id,
                execution_id=self._request_id(request),
                capability=request.capability,
                agent_name=self.agent_name,
                status="partial",
                summary=f"{self.agent_name}: tools succeeded, LLM postprocess failed - {summary}",
                observations=observations,
                evidence=evidence,
                evidence_refs=[item["payload_ref"] for item in evidence if item.get("payload_ref")],
                tool_trace=list(tool_traces),
                tool_traces=list(tool_traces),
                runtime_hints={
                    "cycle_index": request.cycle_index,
                    "llm_postprocess_failed": True,
                    "recoverable": True,
                    "missing_context": (decision or {}).get("missing_context") or [],
                    "successful_tool_count": len(successful_traces),
                },
            )

        return ExecutionResult(
            operation_id=request.operation_id,
            execution_id=self._request_id(request),
            capability=request.capability,
            agent_name=self.agent_name,
            status="needs_replan",
            summary=summary,
            observations=observations,
            evidence=evidence,
            evidence_refs=[item["payload_ref"] for item in evidence if item.get("payload_ref")],
            tool_trace=list(tool_traces),
            tool_traces=list(tool_traces),
            replan_recommendation=str((decision or {}).get("replan_reason") or summary),
            runtime_hints={"cycle_index": request.cycle_index, "missing_context": (decision or {}).get("missing_context") or []},
        )

    def _logger(self, operation_id: str) -> TxtTraceLogger:
        return self._operation_logger or TxtTraceLogger.operation_trace(operation_id)

    @staticmethod
    def _log_tool_result(logger: TxtTraceLogger, trace: ToolTrace) -> None:
        logger.write_block(
            "TOOL_RESULT",
            "mcp tool result",
            {
                "step_index": trace.step,
                "server_id": trace.server_id,
                "tool_name": trace.tool_name,
                "success": trace.success,
                "exit_code": trace.exit_code,
                "stdout_excerpt": trace.stdout[:2000],
                "stderr_excerpt": trace.stderr[:2000],
                "raw_output_ref": trace.raw_output_ref,
            },
        )

    @staticmethod
    def _log_execution_finish(logger: TxtTraceLogger, request: ExecutionRequest, result: ExecutionResult) -> None:
        logger.write_block(
            "EXECUTION_FINISH",
            "execution finished",
            {
                "cycle_index": request.cycle_index,
                "agent_name": request.agent_name,
                "capability": request.capability,
                "status": result.status,
                "summary": result.summary,
                "findings_count": len(result.findings),
                "evidence_count": len(result.evidence),
                "handoff_suggestion": result.handoff_suggestion,
                "replan_recommendation": result.replan_recommendation,
            },
        )

    @staticmethod
    def _lookup_server(catalog: dict[str, Any], *, server_id: str) -> dict[str, Any]:
        server = catalog.get(server_id)
        return dict(server) if isinstance(server, dict) else {}

    @staticmethod
    def _lookup_tool(catalog: dict[str, Any], *, server_id: str, tool_name: str) -> dict[str, Any]:
        server = catalog.get(server_id)
        if not isinstance(server, dict):
            return {}
        for item in server.get("tools", []) if isinstance(server.get("tools"), list) else []:
            if isinstance(item, dict) and str(item.get("name") or item.get("tool_name")) == tool_name:
                return dict(item)
        return {}

    @staticmethod
    def _default_server_id(catalog: dict[str, Any]) -> str:
        if isinstance(catalog.get("pentest-tools"), dict):
            return "pentest-tools"
        for server_id, server in catalog.items():
            if isinstance(server, dict):
                return str(server_id)
        return "pentest-tools"

    @staticmethod
    def _with_trace_arguments(*, call: _ExecutionToolCall, request: ExecutionRequest, step: int) -> _ExecutionToolCall:
        arguments = dict(call.arguments)
        arguments.setdefault("operation_id", request.operation_id)
        arguments.setdefault("trace_id", f"{request.cycle_index}-{request.agent_name}-{step}-{call.tool_name}")
        return call.model_copy(update={"arguments": arguments})

    @classmethod
    def _normalize_tool_call_arguments(cls, *, call: _ExecutionToolCall, request: ExecutionRequest) -> _ExecutionToolCall:
        if call.tool_name in {"http_probe", "web_fingerprint", "whatweb_fingerprint", "nuclei_scan"}:
            arguments = dict(call.arguments)
            url = cls._url_from_ref(arguments.get("url")) or cls._url_from_http_arguments(arguments)
            if not url:
                url = cls._infer_target_url(request)
            if url:
                arguments["url"] = url
                arguments.pop("target", None)
                return call.model_copy(update={"arguments": arguments})
            return call
        if call.tool_name in {"validation_precheck", "safe_vuln_validate"} and not call.arguments.get("target_url"):
            inferred = cls._infer_target_url(request)
            if inferred:
                arguments = dict(call.arguments)
                arguments["target_url"] = inferred
                return call.model_copy(update={"arguments": arguments})
        return call

    @classmethod
    def _infer_target_url(cls, request: ExecutionRequest) -> str | None:
        for candidate in (request.required_context, request.runtime_context, request.graph_summary, request.graph_history):
            found = cls._find_url_value(candidate)
            if found:
                return found
        for ref in request.target_refs:
            found = cls._url_from_ref(ref.ref_id) or cls._url_from_ref(ref.label)
            if found:
                return found
        return None

    @classmethod
    def _find_url_value(cls, value: Any) -> str | None:
        if isinstance(value, str):
            return cls._url_from_ref(value)
        if isinstance(value, dict):
            for key in ("target_url", "canonical_target_url", "url", "endpoint", "base_url"):
                found = cls._url_from_ref(value.get(key))
                if found:
                    return found
            for item in value.values():
                found = cls._find_url_value(item)
                if found:
                    return found
        if isinstance(value, list):
            for item in value:
                found = cls._find_url_value(item)
                if found:
                    return found
        return None

    @staticmethod
    def _url_from_ref(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        parsed = urlparse(text)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return text
        match = re.search(r"(?P<host>(?:\d{1,3}\.){3}\d{1,3}):(?P<port>\d{1,5})(?:/tcp)?", text)
        if not match:
            return None
        port = int(match.group("port"))
        scheme = "https" if port == 443 else "http"
        return f"{scheme}://{match.group('host')}:{port}/"

    @staticmethod
    def _url_from_http_arguments(arguments: dict[str, Any]) -> str | None:
        target = str(arguments.get("target") or arguments.get("host") or "").strip()
        if not target:
            return None
        parsed = urlparse(target)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return target
        if "/" in target and not re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", target):
            return None
        scheme = str(arguments.get("scheme") or "http").strip() or "http"
        port = arguments.get("port")
        path = str(arguments.get("path") or "/").strip() or "/"
        if not path.startswith("/"):
            path = "/" + path
        netloc = target
        if port is not None and ":" not in target:
            netloc = f"{target}:{int(port)}"
        return f"{scheme}://{netloc}{path}"

    @staticmethod
    def _summarize_arguments(arguments: dict[str, Any]) -> str:
        keys = ", ".join(sorted(str(key) for key in arguments)[:8])
        return f"arguments: {keys}" if keys else "no arguments"

    @staticmethod
    def _request_id(request: ExecutionRequest) -> str:
        return f"execution-{request.operation_id}-{request.cycle_index}-{request.agent_name}"

    @staticmethod
    def _original_policy_decision(*, call: _ExecutionToolCall, policy_context: dict[str, Any]) -> dict[str, Any]:
        deny_tools = {str(item) for item in policy_context.get("mcp_tool_denylist", [])}
        if call.tool_name in deny_tools:
            return {"allowed": False, "reason": f"MCP tool '{call.tool_name}' is denied by policy"}
        allow_tools = {str(item) for item in policy_context.get("mcp_tool_allowlist", [])}
        if allow_tools and call.tool_name not in allow_tools:
            return {"allowed": False, "reason": f"MCP tool '{call.tool_name}' is not allowlisted by policy"}
        deny_servers = {str(item) for item in policy_context.get("mcp_server_denylist", [])}
        if call.server_id in deny_servers:
            return {"allowed": False, "reason": f"MCP server '{call.server_id}' is denied by policy"}
        allow_servers = {str(item) for item in policy_context.get("mcp_server_allowlist", [])}
        if allow_servers and call.server_id not in allow_servers:
            return {"allowed": False, "reason": f"MCP server '{call.server_id}' is not allowlisted by policy"}
        return {"allowed": True, "reason": "policy allowed MCP tool call"}


__all__ = ["ExecutionAgent"]
