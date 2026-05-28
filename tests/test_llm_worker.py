from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.core.agents.agent_pipeline import AgentPipeline
from src.core.agents.agent_protocol import AgentContext, GraphRef, GraphScope
from src.core.agents.scheduler_agent import SchedulerAgent
from src.core.execution.configured_mcp_client import ConfiguredMCPClient
from src.core.execution.mcp_client import MCPToolCallResult
from src.core.models.tg import TaskGraph, TaskNode, TaskStatus, TaskType
from src.core.perception.tool_execution_parser import ToolExecutionParser
from src.core.workers.base import WorkerTaskSpec
from src.core.workers.llm_worker import LLMWorkerAgent
from src.core.workers.llm_worker_advisor import LLMWorkerAdvisor
from src.core.workers.llm_worker_models import LLMWorkerDecision
from src.core.workers.recon_worker import ReconWorker


class FakeLLMClient:
    def __init__(self, text: str) -> None:
        self.text = text

    def complete_chat(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(text=self.text, model="fake", usage=None, cost_usd=None, finish_reason="stop")


class FakeAdvisor:
    def __init__(self, decision: LLMWorkerDecision) -> None:
        self.decision = decision

    def advise(self, **kwargs: Any) -> LLMWorkerDecision:
        return self.decision


class FakeMCPClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def is_available(self, server_id: str | None = None) -> bool:
        return True

    def list_tools(self) -> dict[str, Any]:
        return {"pentest-tools": {"tools": [{"name": "run_command"}]}}

    def call_tool(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_seconds: int,
    ) -> MCPToolCallResult:
        self.calls.append(
            {
                "server_id": server_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "timeout_seconds": timeout_seconds,
            }
        )
        return MCPToolCallResult(
            success=True,
            stdout="open 80/tcp",
            exit_code=0,
            metadata={"fixture": True},
        )


def test_llm_worker_advisor_extracts_json_decision() -> None:
    advisor = LLMWorkerAdvisor(
        client=FakeLLMClient(
            """```json
{"action":"call_mcp_tool","server_id":"pentest-tools","tool_name":"run_command","arguments":{"command":"whoami"},"summary":"run command","expected_evidence":["stdout"],"risk_assessment":"lab","writeback_hints":{"category":"command"}}
```"""
        )  # type: ignore[arg-type]
    )

    decision = advisor.advise(
        task_spec=WorkerTaskSpec(task_id="task-1", task_type="port_scan"),
        agent_input=_agent_input(),
        mcp_tool_catalog={"pentest-tools": {"tools": [{"name": "run_command"}]}},
    )

    assert decision.action == "call_mcp_tool"
    assert decision.server_id == "pentest-tools"
    assert decision.arguments["command"] == "whoami"


def test_llm_worker_calls_mcp_and_outputs_tool_execution() -> None:
    mcp = FakeMCPClient()
    worker = LLMWorkerAgent(
        advisor=FakeAdvisor(
            LLMWorkerDecision(
                action="call_mcp_tool",
                server_id="pentest-tools",
                tool_name="run_command",
                arguments={"command": "nmap -p 80 10.0.0.5"},
                summary="execute lab scan",
                writeback_hints={"observation_category": "service_discovery"},
            )
        ),
        mcp_client=mcp,
    )

    output = worker.execute_task(WorkerTaskSpec(task_id="task-1", task_type="port_scan"), _agent_input())

    assert mcp.calls[0]["server_id"] == "pentest-tools"
    assert output.outcomes[0]["tool_execution"]["adapter"] == "mcp_direct"
    assert output.outcomes[0]["tool_execution"]["success"] is True
    assert output.evidence[0]["extra"]["tool_execution"]["stdout"] == "open 80/tcp"


def test_llm_worker_preserves_mcp_structured_parsed_payload() -> None:
    class StructuredMCPClient(FakeMCPClient):
        def call_tool(
            self,
            *,
            server_id: str,
            tool_name: str,
            arguments: dict[str, Any],
            timeout_seconds: int,
        ) -> MCPToolCallResult:
            self.calls.append({"server_id": server_id, "tool_name": tool_name})
            return MCPToolCallResult(
                success=True,
                content={
                    "stdout": "80/tcp open http",
                    "stderr": "",
                    "exit_code": 0,
                    "parsed": {
                        "entities": [{"type": "service", "port": 80, "service": "http"}],
                        "relations": [{"type": "HOSTS", "source": "10.0.0.5", "target": "10.0.0.5:80"}],
                        "findings": [{"kind": "open_service", "port": 80}],
                        "writeback_hints": {"observation_category": "service_discovery"},
                    },
                },
                stdout="80/tcp open http",
                exit_code=0,
            )

    worker = LLMWorkerAgent(
        advisor=FakeAdvisor(
            LLMWorkerDecision(
                action="call_mcp_tool",
                server_id="pentest-tools",
                tool_name="nmap_scan",
                arguments={"target": "10.0.0.5", "ports": "80"},
                summary="scan target",
            )
        ),
        mcp_client=StructuredMCPClient(),
    )

    output = worker.execute_task(WorkerTaskSpec(task_id="task-1", task_type="port_scan"), _agent_input())

    parsed = output.evidence[0]["extra"]["parsed"]
    assert parsed["entities"][0]["service"] == "http"
    assert parsed["relations"][0]["type"] == "HOSTS"
    assert parsed["findings"][0]["kind"] == "open_service"
    assert parsed["writeback_hints"]["observation_category"] == "service_discovery"


def test_llm_worker_defer_does_not_call_mcp() -> None:
    mcp = FakeMCPClient()
    worker = LLMWorkerAgent(
        advisor=FakeAdvisor(LLMWorkerDecision(action="defer", summary="not enough context")),
        mcp_client=mcp,
    )

    output = worker.execute_task(WorkerTaskSpec(task_id="task-1", task_type="unknown"), _agent_input())

    assert mcp.calls == []
    assert output.errors == ["not enough context"]
    assert output.outcomes[0]["payload"]["deferred"] is True


def test_pipeline_scheduler_routes_accepted_task_to_single_llm_worker() -> None:
    mcp = FakeMCPClient()
    worker = LLMWorkerAgent(
        advisor=FakeAdvisor(
            LLMWorkerDecision(
                action="call_mcp_tool",
                server_id="pentest-tools",
                tool_name="run_command",
                arguments={"command": "echo ok"},
                summary="execute scheduled task",
            )
        ),
        mcp_client=mcp,
    )
    pipeline = AgentPipeline(agents=[SchedulerAgent(), worker])
    graph = TaskGraph()
    graph.add_node(
        TaskNode(
            id="task-1",
            label="scan",
            task_type=TaskType.PORT_SCAN,
            status=TaskStatus.READY,
            input_bindings={"target_host": "10.0.0.5"},
        )
    )

    result = pipeline.run_execution_cycle(
        operation_id="op-llm-worker",
        graph_refs=[GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph")],
        scheduler_payload={
            "tg_graph": graph.to_dict(),
            "available_workers": [{"worker_id": "runtime-worker-1", "status": "idle"}],
        },
    )

    worker_steps = [step for step in result.steps if step.agent_name == "llm_worker_agent"]
    assert result.success is True
    assert len(worker_steps) == 1
    assert mcp.calls[0]["tool_name"] == "run_command"


def test_pipeline_prefers_concrete_worker_before_llm_fallback() -> None:
    pipeline = AgentPipeline(agents=[LLMWorkerAgent(), ReconWorker()])
    worker_input = pipeline._build_input(
        "op-worker-selection",
        [GraphRef(graph=GraphScope.KG, ref_id="kg-host::abc", ref_type="Host")],
        {"task_id": "task-1", "task_type": TaskType.ASSET_CONFIRMATION.value},
        task_ref="task-1",
    )

    worker = pipeline._select_worker(worker_input)

    assert isinstance(worker, ReconWorker)


def test_tool_execution_parser_accepts_llm_worker_output() -> None:
    tool_execution = {
        "adapter": "mcp_direct",
        "tool": "run_command",
        "success": True,
        "stdout": "ok",
        "stderr": "",
        "exit_code": 0,
        "metadata": {"server_id": "pentest-tools"},
    }
    outcome = {
        "summary": "done",
        "confidence": 0.8,
        "payload": {"tool_execution": tool_execution},
    }

    parsed = ToolExecutionParser().parse({"tool_execution": tool_execution}, SimpleNamespace(payload=outcome["payload"], confidence=0.8))

    assert parsed.observations[0]["payload"]["adapter"] == "mcp_direct"
    assert parsed.evidence[0]["payload"]["tool"] == "run_command"


def test_configured_mcp_client_from_config_json_reports_missing_server() -> None:
    client = ConfiguredMCPClient.from_sources(config_json={"servers": {"pentest-tools": {"transport": "http", "url": "http://127.0.0.1/mcp"}}})

    result = client.call_tool(server_id="missing", tool_name="run_command", arguments={}, timeout_seconds=1)

    assert client.is_available("pentest-tools") is True
    assert result.success is False
    assert result.exit_code == "mcp_server_not_configured"


def test_configured_mcp_client_from_config_path(tmp_path) -> None:
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        '{"servers":{"pentest-tools":{"transport":"http","url":"http://127.0.0.1/mcp"}}}',
        encoding="utf-8",
    )

    client = ConfiguredMCPClient.from_sources(config_path=config_path)

    assert client.is_available("pentest-tools") is True
    assert client.is_available("missing") is False


def _agent_input():
    from src.core.agents.agent_protocol import AgentInput

    return AgentInput(
        graph_refs=[GraphRef(graph=GraphScope.TG, ref_id="task-1", ref_type="Task")],
        task_ref="task-1",
        context=AgentContext(operation_id="op-llm-worker"),
        raw_payload={"task_id": "task-1", "task_type": "port_scan", "input_bindings": {"target_host": "10.0.0.5"}},
    )
