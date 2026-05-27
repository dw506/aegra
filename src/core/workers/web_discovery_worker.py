"""Controlled directory and API discovery worker."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urljoin, urlparse

from src.core.agents.agent_protocol import AgentInput, AgentOutput
from src.core.execution.adapters.http_request_adapter import HttpRequestExecutionAdapter
from src.core.execution.executor import ExecutionExecutor
from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_result import ToolExecutionResult
from src.core.models.tg import TaskType
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec


class WebDiscoveryWorker(BaseWorkerAgent):
    """Discover same-origin web paths with bounded GET/HEAD requests."""

    capabilities = frozenset({WorkerCapability.WEB_DISCOVERY, WorkerCapability.WEB_ENUMERATION})
    supported_task_types = frozenset({TaskType.WEB_DISCOVERY.value, "web_discovery", "directory_discovery", "api_discovery"})
    safe_methods = frozenset({"GET", "HEAD"})

    def __init__(self, name: str = "web_discovery_worker", *, executor: ExecutionExecutor | None = None) -> None:
        super().__init__(name=name)
        self._executor = executor or ExecutionExecutor([HttpRequestExecutionAdapter()])

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        return task_spec.task_type in self.supported_task_types

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        target_url = self._target_url(task_spec, agent_input)
        if target_url is None:
            return self._result(task_spec, success=False, summary="web discovery requires target_url", endpoints=[], blocked=True)
        method = str(task_spec.input_bindings.get("method") or task_spec.constraints.get("method") or "HEAD").upper()
        if method not in self.safe_methods:
            return self._result(
                task_spec,
                success=False,
                summary=f"web discovery blocked unsupported method {method}",
                endpoints=[],
                blocked=True,
                extra={"blocked_on": "method", "method": method},
            )

        provided = task_spec.input_bindings.get("discovery_results") or agent_input.raw_payload.get("discovery_results")
        if isinstance(provided, list):
            endpoints = [dict(item) for item in provided if isinstance(item, dict)]
        else:
            endpoints = self._probe_paths(target_url=target_url, method=method, task_spec=task_spec)
        return self._result(
            task_spec,
            success=True,
            summary=f"web discovery found {len(endpoints)} endpoint(s) for {target_url}",
            endpoints=endpoints,
            extra={"target_url": target_url, "method": method},
        )

    def _probe_paths(self, *, target_url: str, method: str, task_spec: WorkerTaskSpec) -> list[dict[str, Any]]:
        paths = task_spec.input_bindings.get("paths") or task_spec.input_bindings.get("wordlist") or ["/"]
        if not isinstance(paths, list):
            paths = [paths]
        max_paths = int(task_spec.input_bindings.get("max_paths") or task_spec.constraints.get("max_paths") or 20)
        timeout = float(task_spec.input_bindings.get("timeout_seconds") or task_spec.timeout_seconds or 5)
        parsed_base = urlparse(target_url)
        endpoints: list[dict[str, Any]] = []
        for raw_path in paths[:max_paths]:
            url = urljoin(target_url.rstrip("/") + "/", str(raw_path).lstrip("/"))
            parsed = urlparse(url)
            if parsed.scheme != parsed_base.scheme or parsed.netloc != parsed_base.netloc:
                endpoints.append({"url": url, "blocked": True, "blocked_reason": "cross_origin"})
                continue
            result = self._executor.execute(
                ToolPlan(
                    task_id=task_spec.task_id,
                    tool="http_request",
                    adapter="http_request",
                    target=url,
                    args={"method": method, "timeout_seconds": timeout, "same_origin": target_url},
                    timeout_seconds=max(1, int(timeout)),
                    metadata={"task_type": task_spec.task_type},
                )
            )
            endpoints.append(self._endpoint_from_execution(url=url, path=parsed.path or "/", result=result))
        return endpoints

    @staticmethod
    def _endpoint_from_execution(*, url: str, path: str, result: ToolExecutionResult) -> dict[str, Any]:
        payload = dict(result.metadata)
        if result.stdout.strip():
            try:
                decoded = json.loads(result.stdout)
            except json.JSONDecodeError:
                decoded = {}
            if isinstance(decoded, dict):
                payload.update(decoded)
        if payload.get("blocked_reason"):
            return {"url": url, "path": path, "blocked": True, "blocked_reason": payload.get("blocked_reason")}
        endpoint = {
            "url": url,
            "path": path,
            "status_code": payload.get("status_code"),
            "content_type": payload.get("content_type"),
            "auth_required": bool(payload.get("auth_required", False)),
            "reachable": bool(payload.get("reachable", result.success)),
        }
        if not endpoint["reachable"]:
            endpoint["failure_reason"] = str(payload.get("error_message") or result.stderr or "request failed")
        return endpoint

    def _result(
        self,
        task_spec: WorkerTaskSpec,
        *,
        success: bool,
        summary: str,
        endpoints: list[dict[str, Any]],
        blocked: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> AgentOutput:
        entities = [
            {
                "id": item.get("url") or item.get("path"),
                "type": "WebEndpoint",
                **item,
            }
            for item in endpoints
            if item.get("url") or item.get("path")
        ]
        payload = {
            "task_type": TaskType.WEB_DISCOVERY.value,
            "blocked": blocked,
            "endpoint_count": len(endpoints),
            "endpoints": endpoints,
            "entities": entities,
            **dict(extra or {}),
        }
        evidence = self.build_raw_result(
            task_id=task_spec.task_id,
            result_type="web_discovery_result",
            summary=summary,
            payload_ref=f"runtime://worker-results/web-discovery/{task_spec.task_id}",
            refs=task_spec.target_refs,
            extra={
                "parsed": {
                    "summary": summary,
                    "success": success,
                    "blocked": blocked,
                    "entities": entities,
                    "relations": [],
                    "runtime_hints": {"endpoint_count": len(endpoints)},
                    "evidence": payload,
                }
            },
        )
        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type="web_discovery",
            success=success and not blocked,
            summary=summary,
            raw_result_ref=evidence["payload_ref"],
            confidence=0.75 if success and not blocked else 0.0,
            refs=task_spec.target_refs,
            payload=payload,
        )
        return AgentOutput(
            outcomes=[outcome.to_agent_output_fragment()],
            evidence=[evidence],
            logs=[f"worker={self.name}", summary],
            errors=([summary] if blocked else []),
        )

    @staticmethod
    def _target_url(task_spec: WorkerTaskSpec, agent_input: AgentInput) -> str | None:
        for source in (task_spec.input_bindings, task_spec.constraints, agent_input.raw_payload):
            value = source.get("target_url")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None


__all__ = ["WebDiscoveryWorker"]
