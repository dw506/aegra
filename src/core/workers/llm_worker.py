"""Single experimental worker agent that lets an LLM call MCP tools directly."""

from __future__ import annotations

from typing import Any

from src.core.agents.agent_protocol import AgentInput, AgentOutput
from src.core.execution.mcp_client import MCPClient, MCPToolCallResult, UnavailableMCPClient
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec
from src.core.workers.llm_worker_advisor import LLMWorkerAdvisor, LLMWorkerAdvisorProtocol
from src.core.workers.llm_worker_models import LLMWorkerDecision


class LLMWorkerAgent(BaseWorkerAgent):
    """Experimental all-task worker that delegates MCP tool selection to an LLM."""

    capabilities = frozenset(
        {
            WorkerCapability.RECON,
            WorkerCapability.PORT_SCAN,
            WorkerCapability.FINGERPRINT,
            WorkerCapability.INTERNAL_SERVICE_FINGERPRINT,
            WorkerCapability.WEB_ENUMERATION,
            WorkerCapability.WEB_DISCOVERY,
            WorkerCapability.CONTEXT_VALIDATION,
            WorkerCapability.VULNERABILITY_VALIDATION,
            WorkerCapability.ACCESS_VALIDATION,
            WorkerCapability.CREDENTIAL_VALIDATION,
            WorkerCapability.CREDENTIAL_REUSE_VALIDATION,
            WorkerCapability.LATERAL_REACHABILITY_VALIDATION,
            WorkerCapability.PRIVILEGE_VALIDATION,
            WorkerCapability.GOAL_VALIDATION,
        }
    )

    def __init__(
        self,
        name: str = "llm_worker_agent",
        *,
        advisor: LLMWorkerAdvisorProtocol | None = None,
        mcp_client: MCPClient | None = None,
        default_timeout_seconds: int = 60,
    ) -> None:
        super().__init__(name=name)
        self._advisor = advisor
        self._mcp_client = mcp_client or UnavailableMCPClient()
        self._default_timeout_seconds = default_timeout_seconds

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        del task_spec
        return True

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        advisor = self._resolve_advisor()
        catalog = self._mcp_tool_catalog(agent_input)
        decision = advisor.advise(task_spec=task_spec, agent_input=agent_input, mcp_tool_catalog=catalog)
        if decision.action != "call_mcp_tool":
            return self._decision_only_output(task_spec=task_spec, decision=decision)

        timeout = int(task_spec.timeout_seconds or self._default_timeout_seconds)
        raw_result = self._mcp_client.call_tool(
            server_id=str(decision.server_id),
            tool_name=str(decision.tool_name),
            arguments=dict(decision.arguments),
            timeout_seconds=timeout,
        )
        result = self._coerce_mcp_result(raw_result)
        return self._tool_output(task_spec=task_spec, decision=decision, result=result)

    def _resolve_advisor(self) -> LLMWorkerAdvisorProtocol:
        if self._advisor is not None:
            return self._advisor
        self._advisor = LLMWorkerAdvisor.from_env()
        return self._advisor

    def _mcp_tool_catalog(self, agent_input: AgentInput) -> dict[str, Any]:
        explicit = agent_input.raw_payload.get("mcp_tool_catalog")
        if isinstance(explicit, dict):
            return dict(explicit)
        list_tools = getattr(self._mcp_client, "list_tools", None)
        if callable(list_tools):
            try:
                catalog = list_tools()
            except Exception as exc:
                return {"available": self._mcp_client.is_available(None), "error": str(exc)}
            return dict(catalog) if isinstance(catalog, dict) else {"tools": catalog}
        return {"available": self._mcp_client.is_available(None)}

    def _decision_only_output(self, *, task_spec: WorkerTaskSpec, decision: LLMWorkerDecision) -> AgentOutput:
        success = decision.action == "defer"
        summary = decision.summary or f"LLM worker {decision.action}"
        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type=task_spec.task_type,
            success=False,
            summary=summary,
            confidence=0.0,
            refs=task_spec.target_refs,
            payload={
                "llm_mcp_decision": decision.model_dump(mode="json"),
                "deferred": decision.action == "defer",
                "failed": decision.action == "failed",
            },
        ).to_agent_output_fragment()
        evidence = self.build_raw_result(
            task_id=task_spec.task_id,
            result_type=f"{task_spec.task_type}_llm_worker_decision",
            summary=summary,
            payload_ref=f"runtime://worker-results/llm-worker/{task_spec.task_id}",
            refs=task_spec.target_refs,
            extra={
                "llm_mcp_decision": decision.model_dump(mode="json"),
                "parsed": {
                    "summary": summary,
                    "success": success,
                    "writeback_hints": dict(decision.writeback_hints),
                },
            },
        )
        return AgentOutput(
            outcomes=[outcome],
            evidence=[evidence],
            logs=[f"worker={self.name}", summary],
            errors=[summary],
        )

    def _tool_output(
        self,
        *,
        task_spec: WorkerTaskSpec,
        decision: LLMWorkerDecision,
        result: MCPToolCallResult,
    ) -> AgentOutput:
        summary = decision.summary or f"MCP tool {decision.tool_name} executed"
        tool_execution = {
            "adapter": "mcp_direct",
            "tool": str(decision.tool_name),
            "success": result.success,
            "stdout": result.stdout or self._content_to_stdout(result.content),
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "metadata": {
                **dict(result.metadata),
                "server_id": decision.server_id,
            },
        }
        structured_payload = self._structured_result_payload(result.content)
        parsed_payload = self._parsed_payload(
            structured_payload=structured_payload,
            decision=decision,
            result=result,
            stdout=tool_execution["stdout"],
            summary=summary,
        )
        payload = {
            "tool_execution": tool_execution,
            "llm_mcp_decision": decision.model_dump(mode="json"),
            "expected_evidence": list(decision.expected_evidence),
            "risk_assessment": decision.risk_assessment,
            "writeback_hints": dict(decision.writeback_hints),
            "parsed": parsed_payload,
        }
        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type=task_spec.task_type,
            success=result.success,
            summary=summary,
            raw_result_ref=f"runtime://worker-results/llm-worker/{task_spec.task_id}",
            confidence=0.75 if result.success else 0.0,
            refs=task_spec.target_refs,
            payload=payload,
        ).to_agent_output_fragment()
        outcome["tool_execution"] = tool_execution
        outcome["llm_mcp_decision"] = decision.model_dump(mode="json")
        evidence = self.build_raw_result(
            task_id=task_spec.task_id,
            result_type=f"{task_spec.task_type}_result",
            summary=summary,
            payload_ref=f"runtime://worker-results/llm-worker/{task_spec.task_id}",
            refs=task_spec.target_refs,
            extra={
                "tool_execution": tool_execution,
                "llm_mcp_decision": decision.model_dump(mode="json"),
                "parsed": parsed_payload,
                "mcp_payload": structured_payload,
            },
        )
        return AgentOutput(
            outcomes=[outcome],
            evidence=[evidence],
            logs=[
                f"worker={self.name}",
                f"mcp_server={decision.server_id}",
                f"mcp_tool={decision.tool_name}",
                summary,
            ],
            errors=[] if result.success else [result.stderr or f"MCP tool {decision.tool_name} failed"],
        )

    @staticmethod
    def _coerce_mcp_result(value: MCPToolCallResult | dict[str, Any]) -> MCPToolCallResult:
        if isinstance(value, MCPToolCallResult):
            return value
        if isinstance(value, dict):
            return MCPToolCallResult.model_validate(value)
        if hasattr(value, "model_dump"):
            return MCPToolCallResult.model_validate(value.model_dump(mode="json"))
        raise TypeError(f"unsupported MCP result type: {type(value).__name__}")

    @staticmethod
    def _content_to_stdout(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        import json

        return json.dumps(content, ensure_ascii=True, sort_keys=True)

    @staticmethod
    def _structured_result_payload(content: Any) -> dict[str, Any]:
        if isinstance(content, dict):
            return dict(content)
        return {}

    @staticmethod
    def _parsed_payload(
        *,
        structured_payload: dict[str, Any],
        decision: LLMWorkerDecision,
        result: MCPToolCallResult,
        stdout: str,
        summary: str,
    ) -> dict[str, Any]:
        parsed = structured_payload.get("parsed")
        if not isinstance(parsed, dict):
            parsed = {}
        normalized = {
            "summary": summary,
            "success": result.success,
            "raw_output": stdout,
            "stderr": result.stderr,
            "entities": list(parsed.get("entities", [])) if isinstance(parsed.get("entities"), list) else [],
            "relations": list(parsed.get("relations", [])) if isinstance(parsed.get("relations"), list) else [],
            "findings": list(parsed.get("findings", [])) if isinstance(parsed.get("findings"), list) else [],
            "runtime_hints": dict(parsed.get("runtime_hints", {})) if isinstance(parsed.get("runtime_hints"), dict) else {},
            "writeback_hints": dict(decision.writeback_hints),
        }
        if isinstance(parsed.get("writeback_hints"), dict):
            normalized["writeback_hints"].update(parsed["writeback_hints"])
        return normalized


__all__ = ["LLMWorkerAgent"]
