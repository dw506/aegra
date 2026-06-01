from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.models.runtime import OutcomeCacheEntry
from src.core.models.tg import TaskGraph, TaskStatus
from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.kg_store import KnowledgeGraph
from src.core.planning.mission_planner_agent import MissionPlannerAgent, MissionPlannerResult
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.stage.agents import ExploitAgent, ReconAgent
from src.core.stage.base_stage_agent import StageAgentDecision, StageToolCall
from src.core.stage.models import PlannerResult, StageResult, StageTask, StageType


class PlannerAdvisor:
    def __init__(self) -> None:
        self.calls = 0

    def propose_stage_tasks(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        assert kwargs["policy_context"]["authorized"] is True
        task = {
            "task_id": "stage-recon-1",
            "stage_type": "recon",
            "objective": "Discover authorized surface",
            "priority": 80,
        }
        return {
            "operation_id": "op-arch",
            "reasoning_summary": "LLM selected recon based on missing service evidence",
            "new_stage_tasks": [task],
            "selected_next_task": task,
            "confidence": 0.8,
            "summary": "planned one stage",
        }


class RecordingPlannerAdvisor:
    def __init__(self) -> None:
        self.recent_stage_results: list[dict[str, Any]] = []

    def propose_stage_tasks(self, **kwargs: Any) -> dict[str, Any]:
        self.recent_stage_results = list(kwargs["recent_stage_results"])
        return {
            "operation_id": "op-replan",
            "reasoning_summary": "read recent stage result before planning",
            "new_stage_tasks": [],
            "selected_next_task": None,
            "confidence": 0.8,
            "summary": "planner read previous stage output",
        }


class StageAdvisor:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, **kwargs: Any) -> StageAgentDecision:
        self.calls += 1
        assert kwargs["policy_context"]["authorized"] is True
        if kwargs["memory"]:
            return StageAgentDecision(
                action="finish",
                rationale="tool evidence is sufficient",
                finish={
                    "status": "success",
                    "summary": "recon completed",
                    "observations": [{"summary": "safe probe returned service evidence"}],
                    "confidence": 0.9,
                    "graph_update_intents": [
                        {
                            "target_graph": "KG",
                            "operation": "add",
                            "entity_type": "Service",
                            "entity_ref": "svc-1",
                            "payload": {"source": "tool"},
                            "confidence": 0.8,
                        }
                    ],
                },
            )
        return StageAgentDecision(
            action="call_tool",
            rationale="probe authorized target",
            tool_call=StageToolCall(server_id="mcp", tool_name="safe_recon_probe", arguments={"target": "127.0.0.1"}),
        )


class MCP:
    def is_available(self, server_id: str | None = None) -> bool:
        return True

    def list_tools(self) -> dict[str, Any]:
        return {
            "mcp": {
                "tools": [
                    {
                        "name": "safe_recon_probe",
                        "category": "recon",
                        "description": "safe probe",
                        "requires_authorization": False,
                    }
                ]
            }
        }

    def call_tool(self, **kwargs: Any) -> dict[str, Any]:
        return {"success": True, "stdout": "http service", "metadata": {"parsed_output": {"service": "http"}}}


def test_planner_agent_is_llm_owned_and_result_applier_writes_tg() -> None:
    advisor = PlannerAdvisor()
    planner = MissionPlannerAgent(advisor=advisor)
    result = planner.run(
        goal="validate goal",
        graph_context={"operation_id": "op-arch", "kg_summary": {}, "tg_summary": {}},
        policy_context={"authorized": True},
    )

    assert advisor.calls == 1
    assert isinstance(result, PlannerResult)
    assert result.selected_next_task is not None

    state = RuntimeState(operation_id="op-arch", execution=OperationRuntime(operation_id="op-arch"))
    graph = TaskGraph()
    applied = PhaseTwoResultApplier().apply(result, state, task_graph=graph)

    assert applied.tg_created_task_ids == ["stage-recon-1"]
    assert graph.get_node("stage-recon-1").status == TaskStatus.READY
    assert state.execution.metadata["stage_planning"]["reasoning_summary"]


def test_stage_agent_uses_llm_decision_and_records_tool_trace() -> None:
    advisor = StageAdvisor()
    agent = ReconAgent(advisor=advisor, mcp_client=MCP())
    result = agent.run(
        task=StageTask(task_id="stage-recon-1", stage_type=StageType.RECON, objective="Discover service"),
        graph_context={"operation_id": "op-arch"},
        runtime_context={"operation_id": "op-arch"},
        policy_context={"authorized": True},
        tool_catalog=MCP().list_tools(),
    )

    assert advisor.calls == 2
    assert isinstance(result, StageResult)
    assert result.status == "success"
    assert result.tool_traces[0].tool_name == "safe_recon_probe"
    assert result.graph_update_intents[0].target_graph == "KG"


def test_orchestrator_passes_stage_output_back_to_planner() -> None:
    advisor = RecordingPlannerAdvisor()
    settings = AppSettings(runtime_store_backend="memory")
    orchestrator = AppOrchestrator(settings=settings)
    orchestrator.mission_planner = MissionPlannerAgent(advisor=advisor)
    state = orchestrator.create_operation("op-replan")
    state.record_outcome(
        OutcomeCacheEntry(
            outcome_id="outcome::stage-recon-1",
            task_id="stage-recon-1",
            outcome_type=StageType.RECON_STAGE.value,
            summary="Recon found an HTTP service",
            payload_ref="runtime://stage-results/stage-recon-1",
            metadata={
                "status": "succeeded",
                "outcome_payload": {
                    "stage_result": {
                        "stage_task_id": "stage-recon-1",
                        "stage_type": StageType.RECON_STAGE.value,
                        "status": "succeeded",
                        "summary": "Recon found an HTTP service",
                    },
                    "task_candidates": [
                        {
                            "task_type": StageType.VULN_ANALYSIS_STAGE.value,
                            "input_bindings": {"objective": "Analyze discovered HTTP service"},
                        }
                    ],
                },
            },
        )
    )
    orchestrator.runtime_store.save_state(state)

    orchestrator._run_stage_planning_phase(  # noqa: SLF001
        operation_id="op-replan",
        graph_refs=[],
        planner_payload={"mission_goal": "Continue from graph state", "policy_context": {"authorized": True}},
        kg=KnowledgeGraph(),
        ag=AttackGraphProjector().project(KnowledgeGraph()),
        task_graph=TaskGraph(),
        runtime_state=state,
        context=None,
    )

    assert advisor.recent_stage_results
    assert advisor.recent_stage_results[0]["stage_result"]["summary"] == "Recon found an HTTP service"
    assert advisor.recent_stage_results[0]["task_candidates"][0]["input_bindings"]["objective"] == "Analyze discovered HTTP service"


def test_sensitive_tool_policy_denial_becomes_blocked_tool_trace() -> None:
    advisor = StageAgentDecision(
        action="call_tool",
        rationale="verify exploit",
        tool_call=StageToolCall(server_id="mcp", tool_name="controlled_exploit", arguments={"target": "10.0.0.5"}),
    )

    class StaticAdvisor:
        def decide(self, **_: Any) -> StageAgentDecision:
            return advisor

    tool_catalog = {
        "mcp": {
            "tools": [
                {
                    "name": "controlled_exploit",
                    "category": "exploit",
                    "requires_authorization": True,
                }
            ]
        }
    }
    result = ExploitAgent(advisor=StaticAdvisor(), mcp_client=MCP()).run(
        task=StageTask(task_id="stage-exploit-1", stage_type=StageType.EXPLOIT, objective="Verify exploit"),
        graph_context={"operation_id": "op-arch"},
        runtime_context={"operation_id": "op-arch"},
        policy_context={},
        tool_catalog=tool_catalog,
    )

    assert result.status == "blocked"
    assert result.tool_traces[0].policy_check["allowed"] is False
    assert result.tool_traces[0].exit_code == "policy_denied"


def test_planner_and_stage_layers_do_not_import_graph_mutation_owners() -> None:
    banned = (
        "src.core.graph.kg_store",
        "src.core.graph.ag_projector",
        "src.core.runtime.result_applier",
    )
    for root in (Path("src/core/planning"), Path("src/core/stage")):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = ""
                if isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                if module:
                    assert not module.startswith(banned), f"{path} imports {module}"
