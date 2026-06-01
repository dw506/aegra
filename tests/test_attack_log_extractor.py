from __future__ import annotations

from src.core.models.attack_process import AttackProcessNodeType
from src.core.planning.models import PlannerDecision
from src.core.runtime.attack_log_extractor import AttackLogExtractor
from src.core.stage.models import StageResult, StageType, ToolTrace


def test_attack_log_extractor_extracts_structured_stage_cycle_without_raw_output() -> None:
    decision = PlannerDecision(
        operation_id="op-log",
        cycle_index=2,
        decision="dispatch_agent",
        selected_agent="recon_agent",
        selected_stage="RECON_STAGE",
        objective="Collect service evidence",
        risk_level="low",
        max_steps=3,
        confidence=0.8,
    )
    stage_result = StageResult(
        result_id="result-1",
        operation_id="op-log",
        stage_task_id="stage-op-log-2-recon_agent",
        stage_type=StageType.RECON_STAGE,
        agent_name="recon_agent",
        status="succeeded",
        summary="two probes completed",
        evidence_refs=["evidence:stage"],
        tool_trace=[
            ToolTrace(
                step=0,
                tool_name="probe_a",
                success=True,
                summary="probe a found service",
                stdout="SECRET RAW STDOUT A",
                stderr="",
                raw_output_ref="artifact://probe-a",
                evidence_refs=["evidence:a"],
            ),
            ToolTrace(
                step=1,
                tool_name="probe_b",
                success=True,
                stdout="SECRET RAW STDOUT B",
                stderr="SECRET RAW STDERR B",
                raw_output_ref="artifact://probe-b",
                evidence_refs=["evidence:b"],
            ),
        ],
    )

    extraction = AttackLogExtractor().extract(decision, stage_result, stage_result.tool_trace, [], [])
    nodes_by_type = {}
    for node in extraction.ag_nodes:
        nodes_by_type.setdefault(node.node_type, []).append(node)

    assert extraction.operation_id == "op-log"
    assert extraction.cycle_index == 2
    assert len(nodes_by_type[AttackProcessNodeType.ATTACK_CYCLE]) == 1
    assert len(nodes_by_type[AttackProcessNodeType.PLANNER_DECISION]) == 1
    assert len(nodes_by_type[AttackProcessNodeType.AGENT_EXECUTION]) == 1
    assert len(nodes_by_type[AttackProcessNodeType.TOOL_CALL]) == 2
    assert len(nodes_by_type[AttackProcessNodeType.STAGE_RESULT]) == 1
    assert {"artifact://probe-a", "artifact://probe-b", "evidence:a", "evidence:b", "evidence:stage"}.issubset(
        set(extraction.evidence_refs)
    )

    dumped = str([node.model_dump(mode="json") for node in extraction.ag_nodes])
    assert "SECRET RAW STDOUT" not in dumped
    assert "SECRET RAW STDERR" not in dumped
    assert "raw_output_ref" in dumped


def test_attack_log_extractor_uses_stable_ids_for_same_input() -> None:
    decision = PlannerDecision(
        operation_id="op-stable",
        cycle_index=1,
        decision="dispatch_agent",
        selected_agent="recon_agent",
        selected_stage="RECON_STAGE",
        objective="Collect service evidence",
        risk_level="low",
        max_steps=3,
        confidence=0.8,
    )
    stage_result = StageResult(
        result_id="result-stable",
        operation_id="op-stable",
        stage_task_id="stage-op-stable-1-recon_agent",
        stage_type=StageType.RECON_STAGE,
        agent_name="recon_agent",
        status="succeeded",
        summary="probe completed",
        tool_trace=[ToolTrace(step=0, tool_name="probe", success=True, raw_output_ref="artifact://probe")],
    )

    first = AttackLogExtractor().extract(decision, stage_result, stage_result.tool_trace, [], [])
    second = AttackLogExtractor().extract(decision, stage_result, stage_result.tool_trace, [], [])

    assert [node.id for node in first.ag_nodes] == [node.id for node in second.ag_nodes]
    assert [edge.id for edge in first.ag_edges] == [edge.id for edge in second.ag_edges]
