from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.models.ag import GraphRef
from src.core.stage.agents import ReconAgent
from src.core.stage.base_stage_agent import StageAgentDecision
from src.core.stage.models import StageExecutionRequest, StageResult, StageType
from src.core.stage.registry import StageAgentRegistry


class FinishAdvisor:
    def __init__(self) -> None:
        self.requests: list[StageExecutionRequest] = []

    def decide(self, **kwargs: Any) -> StageAgentDecision:
        request = kwargs["request"]
        self.requests.append(request)
        return StageAgentDecision(
            action="finish",
            rationale="request satisfied",
            finish={
                "status": "succeeded",
                "summary": f"finished {request.objective}",
                "confidence": 0.9,
                "handoff_suggestion": {
                    "suggested_agent": "vuln_analysis_agent",
                    "suggested_stage": "VULN_ANALYSIS_STAGE",
                    "reason": "service evidence is ready for vulnerability analysis",
                    "confidence": 0.8,
                    "required_context_refs": ["kg:svc-1"],
                },
            },
        )


def test_stage_agent_registry_still_registers_five_agents() -> None:
    registry = StageAgentRegistry.default()

    assert registry.resolve(StageType.RECON_STAGE).agent_name == "recon_agent"
    assert registry.resolve(StageType.VULN_ANALYSIS_STAGE).agent_name == "vuln_analysis_agent"
    assert registry.resolve(StageType.EXPLOIT_STAGE).agent_name == "exploit_agent"
    assert registry.resolve(StageType.ACCESS_PIVOT_STAGE).agent_name == "access_pivot_agent"
    assert registry.resolve(StageType.GOAL_STAGE).agent_name == "goal_agent"


def test_base_stage_agent_runs_stage_execution_request_main_path() -> None:
    advisor = FinishAdvisor()
    request = StageExecutionRequest(
        operation_id="op-1",
        cycle_index=2,
        agent_name="recon_agent",
        stage_type=StageType.RECON_STAGE,
        objective="Collect service evidence",
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        required_context={"scope": "authorized"},
        success_criteria=["service evidence recorded"],
        risk_level="low",
        max_steps=3,
        kg_snapshot={"hosts": ["host-1"]},
        ag_process_history={"recent": []},
        runtime_context={"operation_id": "op-1"},
        policy_context={"authorized": True},
        mcp_tool_catalog={},
    )

    result = ReconAgent(advisor=advisor).run(request)

    assert isinstance(result, StageResult)
    assert result.stage_task_id == "stage-op-1-2-recon_agent"
    assert result.summary == "finished Collect service evidence"
    assert advisor.requests == [request]
    assert result.handoff_suggestion is not None
    assert result.handoff_suggestion.suggested_agent == "vuln_analysis_agent"
    assert result.handoff_suggestion.required_context_refs == ["kg:svc-1"]


def test_base_stage_agent_main_path_has_no_task_graph_dependency() -> None:
    source = Path("src/core/stage/base_stage_agent.py").read_text(encoding="utf-8")

    assert "TaskGraph" not in source
    assert "src.core.models.tg" not in source
