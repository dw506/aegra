from __future__ import annotations

from src.app.orchestrator import AppOrchestrator, TargetHost
from src.app.settings import AppSettings
from src.core.planning.models import PlannerOutcome
from src.core.execution.models import RoundDirective, ExecutionRequest, ExecutionResult, ToolTrace
from src.core.execution.execution_agent import ExecutionAgent


class FixedPlanner:
    def decide(self, **kwargs):
        return PlannerOutcome(
            operation_id="operation",
            cycle_index=0,
            action="execute",
            directive=RoundDirective(
                operation_id="operation",
                cycle_index=0,
                objective="collect environment facts",
                max_tools=2,
                risk_level="low",
            ),
            confidence=0.9,
        )


class FixedReconAgent:
    agent_name = "recon_agent"

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        return ExecutionResult(
            operation_id=request.operation_id,
            execution_id=f"execution-{request.operation_id}-{request.cycle_index}-recon_agent",
            agent_name=self.agent_name,
            status="succeeded",
            summary="host discovered",
            tool_trace=[ToolTrace(tool_name="safe_probe", success=True, summary="probe ok", raw_output_ref="runtime://tool-output/probe-1")],
        )


def test_two_graph_runtime_flow_creates_operation_imports_target_and_publishes_ag_kg_without_tg(tmp_path) -> None:
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    orchestrator = AppOrchestrator(settings=settings)
    orchestrator.planner = FixedPlanner()  # type: ignore[assignment]
    orchestrator.execution_agent = ExecutionAgent(FixedReconAgent())  # type: ignore[arg-type]

    orchestrator.create_operation("op-flow")
    orchestrator.import_targets("op-flow", [TargetHost(address="127.0.0.1")])

    result = orchestrator.run_operation_cycle(
        "op-flow",
        graph_refs=[],
        planner_payload={"goal": "collect environment facts"},
    )

    state = result.runtime_state
    assert "task_graph" not in state.execution.metadata
    # KG facts derive solely from tool_trace now: the probe mints a tool-evidence node.
    kg = orchestrator.graph_memory_store.load_kg("op-flow")
    assert any(
        node.type.value == "Evidence" and node.properties.get("tool_name") == "safe_probe"
        for node in kg.list_nodes()
    )
    assert orchestrator.graph_memory_store.load_ag("op-flow").find_process_nodes()
