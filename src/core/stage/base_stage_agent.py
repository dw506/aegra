"""Base implementation for bounded stage-level agents."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.core.execution.mcp_client import MCPClient, MCPToolCallResult, UnavailableMCPClient
from src.core.models.events import utc_now
from src.core.runtime.txt_trace_logger import TxtTraceLogger
from src.core.stage.models import StageExecutionRequest, StageName, StageResult, ToolTrace, normalize_stage_name


_AGENT_TOOL_ALLOWLISTS: dict[str, frozenset[str]] = {
    "recon_agent": frozenset(
        {
            "nmap_scan",
            "http_probe",
            "web_fingerprint",
            "web_discover",
            "dns_lookup",
            "tls_probe",
            "tcp_connect_probe",
        }
    ),
    "vuln_analysis_agent": frozenset(
        {
            "vuln_profile_match",
            "validation_precheck",
            "whatweb_fingerprint",
            "nuclei_scan",
            "http_probe",
        }
    ),
    "exploit_validation_agent": frozenset(
        {
            "validation_precheck",
            "safe_vuln_validate",
            "http_probe",
            "artifact_store",
            "nuclei_scan",
        }
    ),
    "access_pivot_agent": frozenset(
        {
            "credential_check",
            "session_probe",
            "session_open_lab",
            "identity_context_probe",
            "privilege_context_probe",
            "pivot_route_probe",
            "internal_service_discover",
            "tcp_connect_probe",
            "http_probe",
        }
    ),
    "goal_agent": frozenset(
        {
            "goal_check",
            "chain_goal_check",
            "internal_service_discover",
            "http_probe",
            "artifact_store",
        }
    ),
}

_AGENT_TOOL_DENYLISTS: dict[str, frozenset[str]] = {
    "recon_agent": frozenset(
        {
            "run_command",
            "safe_vuln_validate",
            "credential_check",
            "session_open_lab",
            "pivot_route_probe",
        }
    ),
    "vuln_analysis_agent": frozenset(
        {
            "run_command",
            "safe_vuln_validate",
            "credential_check",
            "session_open_lab",
            "pivot_route_probe",
        }
    ),
    "exploit_validation_agent": frozenset(
        {
            "run_command",
            "credential_check",
            "session_open_lab",
            "pivot_route_probe",
            "internal_service_discover",
        }
    ),
    "access_pivot_agent": frozenset({"safe_vuln_validate", "nuclei_scan", "run_command"}),
    "goal_agent": frozenset(
        {
            "safe_vuln_validate",
            "credential_check",
            "session_open_lab",
            "pivot_route_probe",
            "run_command",
        }
    ),
}


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


class BaseStageAdvisor(Protocol):
    """Stage-specific advisor hook for one concrete StageAgent."""

    def build_context(
        self,
        request: StageExecutionRequest,
        graph_context: dict[str, Any],
        runtime_context: dict[str, Any],
        policy_context: dict[str, Any],
        memory: list[dict[str, Any]],
        available_tools: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the agent-specific context supplied to decide()."""

    def decide(
        self,
        *,
        request: StageExecutionRequest,
        graph_context: dict[str, Any],
        runtime_context: dict[str, Any],
        policy_context: dict[str, Any],
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

    stage_type: StageName
    agent_name: str
    allowed_tool_names: frozenset[str] = frozenset()
    denied_tool_names: frozenset[str] = frozenset()
    tool_categories: frozenset[str] = frozenset()
    context_builder_name: str = "base_stage_context_builder"

    def __init__(
        self,
        *,
        advisor: BaseStageAdvisor | None = None,
        mcp_client: MCPClient | None = None,
        default_timeout_seconds: int = 60,
    ) -> None:
        self._advisor = advisor
        self._mcp_client = mcp_client or UnavailableMCPClient()
        self._default_timeout_seconds = default_timeout_seconds

    def run(
        self,
        request: StageExecutionRequest,
    ) -> StageResult:
        if normalize_stage_name(request.stage_type) != normalize_stage_name(self.stage_type):
            raise ValueError(f"{self.agent_name} cannot execute stage type {request.stage_type}")
        if request.agent_name != self.agent_name:
            raise ValueError(f"{self.agent_name} cannot execute request for {request.agent_name}")

        memory: list[dict[str, Any]] = []
        tool_trace: list[ToolTrace] = []
        operation_id = request.operation_id
        policy = dict(request.policy_context)
        graph_context = self.build_graph_context(request)
        runtime_context = dict(request.runtime_context)
        available_tools = self.filter_tool_catalog(request.mcp_tool_catalog)
        if self._advisor is not None and hasattr(self._advisor, "build_context"):
            graph_context = self._advisor.build_context(
                request,
                graph_context,
                runtime_context,
                policy,
                memory,
                available_tools,
            )

        for step in range(request.max_steps):
            decision = self._decide(
                request=request,
                graph_context=graph_context,
                runtime_context=runtime_context,
                policy_context=policy,
                memory=memory,
                available_tools=available_tools,
            )
            TxtTraceLogger(operation_id).write_block(
                "LLM_DECISION",
                "stage agent decision",
                {
                    "agent": request.agent_name,
                    "stage": request.stage_type,
                    "selected_action": decision.action,
                    "tool": decision.tool_call.tool_name if decision.tool_call is not None else None,
                    "rationale_summary": decision.rationale,
                    "confidence": self._decision_metadata(decision).get("confidence"),
                    "assumptions": self._decision_metadata(decision).get("assumptions", []),
                    "evidence_refs": self._decision_metadata(decision).get("evidence_refs", []),
                },
            )

            if decision.action == "call_tool":
                if decision.tool_call is None:
                    return self._replan_result(
                        operation_id=operation_id,
                        request=request,
                        summary="stage advisor requested a tool call without tool_call details",
                        memory=memory,
                        tool_trace=tool_trace,
                    )
                trace = self._call_tool(
                    step=step,
                    call=decision.tool_call,
                    available_tools=available_tools,
                    policy_context=policy,
                    request=request,
                )
                tool_trace.append(trace)
                memory.append({"decision": decision.model_dump(mode="json"), "tool_trace": trace.model_dump(mode="json")})
                continue

            if decision.action == "finish":
                return self._finish_result(
                    operation_id=operation_id,
                    request=request,
                    decision=decision,
                    memory=memory,
                    tool_trace=tool_trace,
                )

            return self._replan_result(
                operation_id=operation_id,
                request=request,
                summary=decision.rationale or "stage agent requested replanning",
                memory=memory,
                tool_trace=tool_trace,
            )

        return self._finish_result(
            operation_id=operation_id,
            request=request,
            decision=StageAgentDecision(
                action="finish",
                rationale="stage loop reached max_steps",
                finish={
                    "status": "partial" if tool_trace else "need_more_info",
                    "summary": f"{self.agent_name} reached max_steps for {self._request_id(request)}",
                },
            ),
            memory=memory,
            tool_trace=tool_trace,
        )

    def filter_tool_catalog(self, tool_catalog: dict[str, Any]) -> dict[str, Any]:
        if not self.allowed_tool_names and not self.tool_categories:
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
                if self._tool_matches_stage(tool)
            ]
            if selected:
                filtered[server_id] = {**payload, "tools": selected}
        return filtered

    def build_graph_context(self, request: StageExecutionRequest) -> dict[str, Any]:
        """Build the read-only context passed to this StageAgent's advisor.

        Subclasses own the stage-specific shape. The default keeps all
        Planner-provided graph/runtime/policy inputs but marks which builder
        produced the context for auditability.
        """

        return {
            **self._graph_context_from_request(request),
            "stage_context_builder": self.context_builder_name,
            "stage_agent": self.agent_name,
            "stage_type": normalize_stage_name(self.stage_type),
        }

    def _decide(
        self,
        *,
        request: StageExecutionRequest,
        graph_context: dict[str, Any],
        runtime_context: dict[str, Any],
        policy_context: dict[str, Any],
        memory: list[dict[str, Any]],
        available_tools: dict[str, Any],
    ) -> StageAgentDecision:
        if self._advisor is None:
            return StageAgentDecision(
                action="need_replan",
                rationale=f"{self.agent_name} requires an LLM advisor; no hard-coded stage execution is available",
            )
        try:
            raw = self._advisor.decide(
                request=request,
                graph_context=graph_context,
                runtime_context=runtime_context,
                policy_context=policy_context,
                memory=memory,
                available_tools=available_tools,
            )
        except TypeError as exc:
            if "agent_name" not in str(exc) and "stage_type" not in str(exc):
                raise
            raw = self._advisor.decide(
                agent_name=self.agent_name,
                stage_type=self.stage_type,
                request=request,
                graph_context=graph_context,
                runtime_context=runtime_context,
                policy_context=policy_context,
                memory=memory,
                available_tools=available_tools,
            )
        return raw if isinstance(raw, StageAgentDecision) else StageAgentDecision.model_validate(raw)

    def _call_tool(
        self,
        *,
        step: int,
        call: StageToolCall,
        available_tools: dict[str, Any],
        policy_context: dict[str, Any],
        request: StageExecutionRequest,
    ) -> ToolTrace:
        timeout = int(call.timeout_seconds or self._default_timeout_seconds)
        tool_metadata = self._lookup_tool(available_tools, server_id=call.server_id, tool_name=call.tool_name)
        policy_check = self._enforce_policy(
            call=call,
            tool_metadata=tool_metadata,
            policy_context=policy_context,
            request=request,
        )
        TxtTraceLogger(request.operation_id).write_block(
            "AGENT_ACTION",
            "agent action",
            {
                "agent": request.agent_name,
                "action": "tool_call",
                "tool": call.tool_name,
                "target": call.arguments.get("target") or call.arguments.get("url") or call.arguments.get("host"),
            },
        )
        TxtTraceLogger(request.operation_id).write_block(
            "POLICY_DECISION",
            "policy evaluated but not enforced",
            {
                "agent": request.agent_name,
                "stage": request.stage_type,
                "server": call.server_id,
                "tool": call.tool_name,
                "target": call.arguments.get("target") or call.arguments.get("url") or call.arguments.get("host"),
                "original_allowed": policy_check.get("metadata", {}).get("original_allowed", policy_check.get("allowed")),
                "original_reason": policy_check.get("metadata", {}).get("original_reason", policy_check.get("reason")),
                "original_risk_level": policy_context.get("risk_level") or request.risk_level,
                "original_tags": tool_metadata.get("policy_tags", []),
                "original_policy_name": "stage_agent_tool_policy",
                "final_allowed": policy_check.get("allowed"),
                "final_reason": policy_check.get("reason"),
            },
        )
        TxtTraceLogger(request.operation_id).write_block(
            "TOOL_CALL",
            "mcp tool call",
            {
                "server": call.server_id,
                "tool": call.tool_name,
                "arguments": dict(call.arguments),
            },
        )
        raw = self._mcp_client.call_tool(
            server_id=call.server_id,
            tool_name=call.tool_name,
            arguments=dict(call.arguments),
            timeout_seconds=timeout,
        )
        result = raw if isinstance(raw, MCPToolCallResult) else MCPToolCallResult.model_validate(raw)
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
            policy_check=policy_check,
            raw_output_ref=str(result.metadata.get("raw_output_ref") or ""),
            parsed_output=dict(result.metadata.get("parsed_output") or {}),
            metadata={**dict(result.metadata), "content": result.content},
        )
        TxtTraceLogger(request.operation_id).write_block(
            "TOOL_RESULT",
            "mcp tool result",
            {
                "tool": call.tool_name,
                "success": result.success,
                "exit_code": result.exit_code,
                "summary": result.stderr if not result.success else result.stdout[:200],
                "stdout_excerpt": result.stdout[:2000] if result.stdout else "",
                "stderr_excerpt": result.stderr[:2000] if result.stderr else "",
            },
        )
        return trace

    @staticmethod
    def _decision_metadata(decision: StageAgentDecision) -> dict[str, Any]:
        metadata = getattr(decision, "metadata", {})
        return dict(metadata) if isinstance(metadata, dict) else {}

    def _finish_result(
        self,
        *,
        operation_id: str,
        request: StageExecutionRequest,
        decision: StageAgentDecision,
        memory: list[dict[str, Any]],
        tool_trace: list[ToolTrace],
    ) -> StageResult:
        payload = dict(decision.finish)
        status = str(payload.get("status") or ("succeeded" if tool_trace and all(item.success for item in tool_trace) else "partial"))
        return StageResult(
            operation_id=operation_id,
            stage_task_id=self._request_id(request),
            stage_type=request.stage_type,
            agent_name=self.agent_name,
            status=status,  # type: ignore[arg-type]
            summary=str(payload.get("summary") or decision.rationale or f"{self.agent_name} finished {self._request_id(request)}"),
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
            tool_traces=list(tool_trace),
            graph_update_intents=list(payload.get("graph_update_intents") or []),
            evidence_refs=[str(item) for item in payload.get("evidence_refs", []) if item is not None],
            confidence=float(payload.get("confidence") or 0.5),
            risk_level=str(payload.get("risk_level") or request.risk_level),  # type: ignore[arg-type]
            policy_notes=[str(item) for item in payload.get("policy_notes", []) if item is not None],
            retry_recommendation=payload.get("retry_recommendation"),
            replan_recommendation=payload.get("replan_recommendation"),
            next_stage_suggestion=payload.get("next_stage_suggestion"),
            handoff_suggestion=payload.get("handoff_suggestion"),
            runtime_hints={**dict(payload.get("runtime_hints") or {}), "cycle_index": request.cycle_index},
            writeback_hints=dict(payload.get("writeback_hints") or {}),
        )

    def _replan_result(
        self,
        *,
        operation_id: str,
        request: StageExecutionRequest,
        summary: str,
        memory: list[dict[str, Any]],
        tool_trace: list[ToolTrace],
    ) -> StageResult:
        return StageResult(
            operation_id=operation_id,
            stage_task_id=self._request_id(request),
            stage_type=request.stage_type,
            agent_name=self.agent_name,
            status="needs_replan",
            summary=summary,
            observations=list(memory),
            tool_trace=list(tool_trace),
            tool_traces=list(tool_trace),
            replan_recommendation=summary,
            runtime_hints={"cycle_index": request.cycle_index},
        )

    @staticmethod
    def _summarize_arguments(arguments: dict[str, Any]) -> str:
        keys = ", ".join(sorted(str(key) for key in arguments)[:8])
        return f"arguments: {keys}" if keys else "no arguments"

    def _tool_matches_stage(self, tool: Any) -> bool:
        if not isinstance(tool, dict):
            return any(category in str(tool).lower() for category in self.tool_categories)
        category = str(tool.get("category") or "").lower()
        name = str(tool.get("name") or tool.get("tool_name") or "").lower()
        if name in self.denied_tool_names:
            return False
        if self.allowed_tool_names:
            return name in self.allowed_tool_names
        tags = " ".join(str(item).lower() for item in tool.get("policy_tags", []) if item is not None)
        return any(item in {category} or item in name or item in tags for item in self.tool_categories)

    @staticmethod
    def _lookup_tool(available_tools: dict[str, Any], *, server_id: str, tool_name: str) -> dict[str, Any]:
        server = available_tools.get(server_id)
        if not isinstance(server, dict):
            return {}
        for item in server.get("tools", []) if isinstance(server.get("tools"), list) else []:
            if not isinstance(item, dict):
                continue
            if str(item.get("name") or item.get("tool_name")) == tool_name:
                return dict(item)
        return {}

    @classmethod
    def _enforce_policy(
        cls,
        *,
        call: StageToolCall,
        tool_metadata: dict[str, Any],
        policy_context: dict[str, Any],
        request: StageExecutionRequest,
    ) -> dict[str, Any]:
        original = cls._original_policy_decision(
            call=call,
            tool_metadata=tool_metadata,
            policy_context=policy_context,
            request=request,
        )
        return {
            "allowed": True,
            "reason": "PolicyGate is audit-only; execution continued.",
            "metadata": {
                "policy_audit_only": True,
                "original_allowed": bool(original.get("allowed")),
                "original_reason": str(original.get("reason") or ""),
                "original_risk_level": policy_context.get("risk_level") or request.risk_level,
                "original_tags": list(tool_metadata.get("policy_tags", []) or []),
                "original_policy_name": "stage_agent_tool_policy",
                "original_policy_check": original,
            },
        }

    @staticmethod
    def _original_policy_decision(
        *,
        call: StageToolCall,
        tool_metadata: dict[str, Any],
        policy_context: dict[str, Any],
        request: StageExecutionRequest,
    ) -> dict[str, Any]:
        category = str(tool_metadata.get("category") or "").lower()
        name = call.tool_name
        if not tool_metadata:
            return {"allowed": False, "reason": f"tool {call.server_id}.{name} is not in the supplied MCP catalog"}
        agent_allowlist = _AGENT_TOOL_ALLOWLISTS.get(request.agent_name, frozenset())
        agent_denylist = _AGENT_TOOL_DENYLISTS.get(request.agent_name, frozenset())
        if name in agent_denylist:
            return {"allowed": False, "reason": f"MCP tool '{name}' is denied for {request.agent_name}"}
        if agent_allowlist and name not in agent_allowlist:
            return {"allowed": False, "reason": f"MCP tool '{name}' is not allowlisted for {request.agent_name}"}
        deny_tools = {str(item) for item in policy_context.get("mcp_tool_denylist", [])}
        deny_tools.update(str(item) for item in policy_context.get("disabled_tools", []))
        if name in deny_tools:
            return {"allowed": False, "reason": f"MCP tool '{name}' is denied by policy"}
        allow_tools = {str(item) for item in policy_context.get("mcp_tool_allowlist", [])}
        if allow_tools and name not in allow_tools:
            return {"allowed": False, "reason": f"MCP tool '{name}' is not allowlisted by policy"}
        deny_servers = {str(item) for item in policy_context.get("mcp_server_denylist", [])}
        if call.server_id in deny_servers:
            return {"allowed": False, "reason": f"MCP server '{call.server_id}' is denied by policy"}
        allow_servers = {str(item) for item in policy_context.get("mcp_server_allowlist", [])}
        if allow_servers and call.server_id not in allow_servers:
            return {"allowed": False, "reason": f"MCP server '{call.server_id}' is not allowlisted by policy"}
        denied_risk = BaseStageAgent._denied_risk_pattern(
            name=name,
            category=category,
            tool_metadata=tool_metadata,
            arguments=call.arguments,
        )
        if denied_risk is not None:
            return {"allowed": False, "reason": f"high-risk action is not allowed for StageAgent execution: {denied_risk}"}
        strict = category in {"exploit", "access", "credential", "pivot"} or normalize_stage_name(request.stage_type) in {
            "EXPLOIT_STAGE",
            "ACCESS_PIVOT_STAGE",
        }
        if strict and bool(tool_metadata.get("requires_authorization", True)):
            authorized = bool(policy_context.get("authorized") or policy_context.get("authorization_confirmed"))
            authorized = authorized or bool(policy_context.get("authorized_hosts") or policy_context.get("engagement"))
            if not authorized:
                return {"allowed": False, "reason": f"{category or 'sensitive'} tool requires explicit authorization context"}
        stage = normalize_stage_name(request.stage_type)
        if stage == "EXPLOIT_STAGE" and not BaseStageAgent._is_safe_validation_tool(tool_metadata):
            return {"allowed": False, "reason": "ExploitValidationAgent can only call safe validation tools"}
        if stage == "ACCESS_PIVOT_STAGE" and not BaseStageAgent._is_access_pivot_validation_tool(tool_metadata):
            return {"allowed": False, "reason": "AccessPivotAgent can only validate authorized session, credential, pivot route, or reachability state"}
        return {"allowed": True, "reason": "policy allowed MCP tool call"}

    @staticmethod
    def _denied_risk_pattern(
        *,
        name: str,
        category: str,
        tool_metadata: dict[str, Any],
        arguments: dict[str, Any],
    ) -> str | None:
        text = " ".join(
            [
                name,
                category,
                str(tool_metadata.get("description") or ""),
                " ".join(str(item) for item in tool_metadata.get("policy_tags", []) if item is not None),
                " ".join(str(key) for key in arguments.keys()),
            ]
        ).lower()
        denied_terms = {
            "reverse_shell": "reverse shell",
            "reverse shell": "reverse shell",
            "bind_shell": "bind shell",
            "bind shell": "bind shell",
            "persistence": "persistence",
            "persist": "persistence",
            "evasion": "evasion",
            "bypass": "bypass",
            "stealth": "stealth",
            "bruteforce": "credential brute force",
            "brute_force": "credential brute force",
            "brute force": "credential brute force",
            "password_spray": "credential spraying",
            "spray": "credential spraying",
            "destructive": "destructive action",
            "delete": "destructive action",
            "wipe": "destructive action",
            "ransom": "destructive action",
        }
        for term, reason in denied_terms.items():
            if term in text:
                return reason
        return None

    @staticmethod
    def _is_safe_validation_tool(tool_metadata: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(tool_metadata.get("name") or tool_metadata.get("tool_name") or ""),
                str(tool_metadata.get("category") or ""),
                str(tool_metadata.get("description") or ""),
                " ".join(str(item) for item in tool_metadata.get("policy_tags", []) if item is not None),
            ]
        ).lower()
        return any(term in text for term in ("safe", "validate", "validation", "precheck", "fingerprint", "probe", "artifact"))

    @staticmethod
    def _is_access_pivot_validation_tool(tool_metadata: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(tool_metadata.get("name") or tool_metadata.get("tool_name") or ""),
                str(tool_metadata.get("category") or ""),
                str(tool_metadata.get("description") or ""),
                " ".join(str(item) for item in tool_metadata.get("policy_tags", []) if item is not None),
            ]
        ).lower()
        return any(term in text for term in ("validate", "validation", "session", "credential", "pivot", "route", "reachability", "identity", "privilege"))

    @staticmethod
    def _graph_context_from_request(request: StageExecutionRequest) -> dict[str, Any]:
        two_graph_keys = {
            "kg_summary",
            "ag_process_summary",
            "runtime_summary",
            "policy_summary",
            "recent_evidence",
            "known_assets",
            "known_services",
            "active_sessions",
            "recent_attack_process_nodes",
            "recent_handoff_suggestions",
            "recent_failures",
            "current_goal",
        }
        if two_graph_keys.issubset(set(request.kg_snapshot.keys())):
            return {key: request.kg_snapshot.get(key) for key in sorted(two_graph_keys)}
        return {
            "operation_id": request.operation_id,
            "cycle_index": request.cycle_index,
            "kg_snapshot": dict(request.kg_snapshot),
            "ag_process_history": dict(request.ag_process_history),
            "runtime_context": dict(request.runtime_context),
            "policy_context": dict(request.policy_context),
            "target_refs": [ref.model_dump(mode="json") for ref in request.target_refs],
            "required_context": dict(request.required_context),
            "success_criteria": list(request.success_criteria),
        }

    @staticmethod
    def _request_id(request: StageExecutionRequest) -> str:
        return f"stage-{request.operation_id}-{request.cycle_index}-{request.agent_name}"


StageAgentAdvisor = BaseStageAdvisor


__all__ = ["BaseStageAgent", "BaseStageAdvisor", "StageAgentAdvisor", "StageAgentDecision", "StageToolCall"]
