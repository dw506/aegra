"""LLM-driven shared execution kernel for concrete Stage Agents."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.agents.packy_llm import PackyLLMClient, PackyLLMError
from src.core.execution.mcp_client import MCPClient, MCPToolCallResult, UnavailableMCPClient
from src.core.models.events import utc_now
from src.core.runtime.txt_trace_logger import TxtTraceLogger
from src.core.stage.models import StageExecutionRequest, StageName, StageResult, ToolTrace, normalize_stage_name


class LLMDrivenToolCall(BaseModel):
    """One MCP tool call selected by an LLM-driven StageAgent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    server_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=120, ge=1)


class LLMDrivenStageAgent:
    """Autonomous bounded stage loop shared by the five execution agents."""

    stage_type: StageName
    agent_name: str
    role_prompt: str = ""
    context_builder_name: str = "llm_driven_stage_context_builder"

    def __init__(
        self,
        *,
        llm_client: PackyLLMClient | None = None,
        mcp_client: MCPClient | None = None,
        operation_logger: TxtTraceLogger | None = None,
        default_timeout_seconds: int = 120,
        **_: Any,
    ) -> None:
        self._llm_client = llm_client
        self._mcp_client = mcp_client or UnavailableMCPClient()
        self._operation_logger = operation_logger
        self._default_timeout_seconds = default_timeout_seconds

    def run(self, request: StageExecutionRequest) -> StageResult:
        if normalize_stage_name(request.stage_type) != normalize_stage_name(self.stage_type):
            raise ValueError(f"{self.agent_name} cannot execute stage type {request.stage_type}")
        if request.agent_name != self.agent_name:
            raise ValueError(f"{self.agent_name} cannot execute request for {request.agent_name}")

        logger = self._logger(request.operation_id)
        logger.write_header(
            f"Operation: {request.operation_id}",
            {"cycle_index": request.cycle_index, "agent_name": request.agent_name, "stage_type": request.stage_type},
        )
        logger.write_block(
            "CYCLE_START",
            "stage execution started",
            {
                "cycle_index": request.cycle_index,
                "agent_name": request.agent_name,
                "stage_type": request.stage_type,
                "objective": request.objective,
                "task_brief": request.task_brief,
                "max_steps": request.max_steps,
            },
        )

        if self._llm_client is None:
            result = self._replan_result(request, "llm_client unavailable", [], [])
            self._log_stage_finish(logger, request, result)
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
                self._log_stage_finish(logger, request, result)
                return result

            decision = self._extract_json_object(raw_text)
            if decision is None:
                logger.write_block("LLM_DECISION", "invalid json", {"step_index": step, "raw_response": raw_text[:2000]})
                result = self._replan_result(request, "LLM decision JSON parse failed", memory, tool_traces)
                self._log_stage_finish(logger, request, result)
                return result

            action = str(decision.get("action") or "")
            logger.write_block(
                "LLM_DECISION",
                "stage agent decision",
                {
                    "cycle_index": request.cycle_index,
                    "agent_name": request.agent_name,
                    "stage_type": request.stage_type,
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
                self._log_stage_finish(logger, request, result)
                return result

            if action == "need_replan":
                summary = str(decision.get("summary") or decision.get("replan_reason") or "stage agent requested replanning")
                result = self._replan_result(request, summary, memory, tool_traces, decision=decision)
                self._log_stage_finish(logger, request, result)
                return result

            result = self._replan_result(request, f"unsupported LLM action: {action}", memory, tool_traces)
            self._log_stage_finish(logger, request, result)
            return result

        result = StageResult(
            operation_id=request.operation_id,
            stage_task_id=self._request_id(request),
            stage_type=request.stage_type,
            agent_name=self.agent_name,
            status="partial",
            summary=f"{self.agent_name} reached max_steps for {self._request_id(request)}",
            observations=list(memory),
            tool_trace=list(tool_traces),
            tool_traces=list(tool_traces),
            runtime_hints={"cycle_index": request.cycle_index},
        )
        self._log_stage_finish(logger, request, result)
        return result

    def _call_mcp_tool(
        self,
        *,
        step: int,
        request: StageExecutionRequest,
        decision: dict[str, Any],
        logger: TxtTraceLogger,
    ) -> ToolTrace:
        arguments = dict(decision.get("arguments") or decision.get("input") or {})
        call = LLMDrivenToolCall(
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
                "stage_type": request.stage_type,
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

        raw = self._mcp_client.call_tool(
            server_id=call.server_id,
            tool_name=call.tool_name,
            arguments=dict(call.arguments),
            timeout_seconds=call.timeout_seconds,
        )
        result = raw if isinstance(raw, MCPToolCallResult) else MCPToolCallResult.model_validate(raw)
        original_policy = self._original_policy_decision(call=call, policy_context=request.policy_context)
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
            raw_output_ref=str(result.metadata.get("raw_output_ref") or ""),
            parsed_output=dict(result.metadata.get("parsed_output") or {}),
            metadata={**dict(result.metadata), "content": result.content},
        )
        self._log_tool_result(logger, trace)
        return trace

    def _build_messages(self, request: StageExecutionRequest, memory: list[dict[str, Any]]) -> list[dict[str, str]]:
        system_prompt = (
            "You are an LLM-driven StageAgent for Aegra, an authorized local lab automation framework. "
            "Return strict JSON only. Allowed actions are call_mcp_tool, finish, need_replan. "
            "Call only tools present in mcp_tool_catalog. Prefer argv for run_command. "
            "Do not invent facts; base findings on KG/Runtime/evidence/tool results."
        )
        context = {
            "agent_name": request.agent_name,
            "stage_type": request.stage_type,
            "role_prompt": self.role_prompt,
            "planner_objective": request.objective,
            "task_brief": request.task_brief,
            "target_refs": [ref.model_dump(mode="json") for ref in request.target_refs],
            "success_criteria": request.success_criteria,
            "kg_snapshot": request.kg_snapshot,
            "ag_process_history": request.ag_process_history,
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
        request: StageExecutionRequest,
        payload: dict[str, Any],
        memory: list[dict[str, Any]],
        tool_traces: list[ToolTrace],
    ) -> StageResult:
        payload = self._normalized_finish_payload(payload)
        try:
            return StageResult(
                operation_id=request.operation_id,
                stage_task_id=self._request_id(request),
                stage_type=request.stage_type,
                agent_name=self.agent_name,
                status=self._normalized_status(payload.get("status")),  # type: ignore[arg-type]
                summary=str(payload.get("summary") or f"{self.agent_name} finished"),
                observations=list(payload.get("observations") or memory),
                evidence=self._normalized_evidence(payload),
                findings=self._normalized_dict_list(payload.get("findings"), default_kind="stage_finding"),
                discovered_entities=self._normalized_dict_list(payload.get("discovered_entities"), default_kind="entity"),
                discovered_relations=self._normalized_dict_list(payload.get("discovered_relations"), default_kind="relation"),
                capabilities_gained=list(payload.get("capabilities_gained") or []),
                credentials=list(payload.get("credentials") or []),
                sessions=list(payload.get("sessions") or []),
                pivot_routes=list(payload.get("pivot_routes") or []),
                next_stage_candidates=list(payload.get("next_stage_candidates") or []),
                handoff_suggestion=payload.get("handoff_suggestion"),
                evidence_refs=self._normalized_evidence_refs(payload.get("evidence_refs")),
                tool_trace=list(tool_traces),
                tool_traces=list(tool_traces),
                confidence=float(payload.get("confidence") or 0.5),
                replan_recommendation=payload.get("replan_recommendation"),
                runtime_hints={"cycle_index": request.cycle_index},
            )
        except (TypeError, ValueError, ValidationError) as exc:
            return self._replan_result(request, f"invalid finish payload: {exc}", memory, tool_traces)

    @staticmethod
    def _normalized_finish_payload(payload: dict[str, Any]) -> dict[str, Any]:
        nested = payload.get("result") or payload.get("data") or payload.get("output")
        if not isinstance(nested, dict):
            return dict(payload)
        merged = {key: value for key, value in payload.items() if key not in {"result", "data", "output"}}
        merged.update(nested)
        return merged

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
        request: StageExecutionRequest,
        summary: str,
        memory: list[dict[str, Any]],
        tool_traces: list[ToolTrace],
        *,
        decision: dict[str, Any] | None = None,
    ) -> StageResult:
        return StageResult(
            operation_id=request.operation_id,
            stage_task_id=self._request_id(request),
            stage_type=request.stage_type,
            agent_name=self.agent_name,
            status="needs_replan",
            summary=summary,
            observations=list(memory),
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
    def _log_stage_finish(logger: TxtTraceLogger, request: StageExecutionRequest, result: StageResult) -> None:
        logger.write_block(
            "STAGE_FINISH",
            "stage finished",
            {
                "cycle_index": request.cycle_index,
                "agent_name": request.agent_name,
                "stage_type": request.stage_type,
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
    def _with_trace_arguments(*, call: LLMDrivenToolCall, request: StageExecutionRequest, step: int) -> LLMDrivenToolCall:
        arguments = dict(call.arguments)
        arguments.setdefault("operation_id", request.operation_id)
        arguments.setdefault("trace_id", f"{request.cycle_index}-{request.agent_name}-{step}-{call.tool_name}")
        return call.model_copy(update={"arguments": arguments})

    @classmethod
    def _normalize_tool_call_arguments(cls, *, call: LLMDrivenToolCall, request: StageExecutionRequest) -> LLMDrivenToolCall:
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
    def _infer_target_url(cls, request: StageExecutionRequest) -> str | None:
        for candidate in (request.required_context, request.runtime_context, request.kg_snapshot, request.ag_process_history):
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
    def _request_id(request: StageExecutionRequest) -> str:
        return f"stage-{request.operation_id}-{request.cycle_index}-{request.agent_name}"

    @staticmethod
    def _original_policy_decision(*, call: LLMDrivenToolCall, policy_context: dict[str, Any]) -> dict[str, Any]:
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


__all__ = ["LLMDrivenStageAgent"]
