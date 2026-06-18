from __future__ import annotations

from src.app.orchestrator import AppOrchestrator, TargetHost
from src.app.settings import AppSettings
from src.core.planning.models import PlannerOutcome
from src.core.stage.models import RoundDirective, StageExecutionRequest, StageResult, StageType, ToolTrace
from src.core.stage.registry import StageAgentRegistry


class FixedPlanner:
    def decide(self, **kwargs):
        return PlannerOutcome(
            operation_id="operation",
            cycle_index=0,
            action="execute",
            directive=RoundDirective(
                operation_id="operation",
                cycle_index=0,
                capability="recon",
                objective="collect environment facts",
                max_tools=2,
                risk_level="low",
            ),
            confidence=0.9,
        )


class FixedReconAgent:
    agent_name = "recon_agent"
    stage_type = StageType.RECON_STAGE

    def run(self, request: StageExecutionRequest) -> StageResult:
        return StageResult(
            operation_id=request.operation_id,
            stage_task_id=f"stage-{request.operation_id}-{request.cycle_index}-recon_agent",
            stage_type=request.stage_type,
            agent_name=self.agent_name,
            status="succeeded",
            summary="host discovered",
            discovered_entities=[
                {"id": "host-1", "type": "Host", "summary": "127.0.0.1", "address": "127.0.0.1", "confidence": 0.9}
            ],
            tool_trace=[ToolTrace(tool_name="safe_probe", success=True, summary="probe ok")],
        )


def test_two_graph_runtime_flow_creates_operation_imports_target_and_publishes_ag_kg_without_tg(tmp_path) -> None:
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    orchestrator = AppOrchestrator(settings=settings)
    orchestrator.mission_planner = FixedPlanner()  # type: ignore[assignment]
    orchestrator.stage_registry = StageAgentRegistry([FixedReconAgent()])  # type: ignore[list-item]

    orchestrator.create_operation("op-flow")
    orchestrator.import_targets("op-flow", [TargetHost(address="127.0.0.1")])

    result = orchestrator.run_operation_cycle(
        "op-flow",
        graph_refs=[],
        planner_payload={"goal": "collect environment facts"},
    )

    state = result.runtime_state
    graphs = [delta.graph for applied in result.apply_results for delta in applied.visual_graph_deltas]
    assert "ag" in graphs
    assert "kg" in graphs
    assert "tg" not in graphs
    assert "task_graph" not in state.execution.metadata
    assert orchestrator.graph_memory_store.load_kg("op-flow").get_node("host-1") is not None
    assert orchestrator.graph_memory_store.load_ag("op-flow").find_process_nodes()
