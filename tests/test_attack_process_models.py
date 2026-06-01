from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.models.attack_process import (
    AttackProcessEdge,
    AttackProcessEdgeType,
    AttackProcessNode,
    AttackProcessNodeType,
    GraphRef,
    stable_node_id,
)


def test_attack_process_node_records_process_event() -> None:
    node_id = stable_node_id(
        "ap-node",
        {
            "operation_id": "op-1",
            "cycle_index": 1,
            "node_type": AttackProcessNodeType.PLANNER_DECISION.value,
        },
    )

    node = AttackProcessNode(
        id=node_id,
        node_type=AttackProcessNodeType.PLANNER_DECISION,
        label="Planner selected recon agent",
        operation_id="op-1",
        cycle_index=1,
        agent_name="PlannerAgent",
        stage_type="recon",
        status="selected",
        summary="Planner selected the next authorized stage.",
        refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        evidence_refs=["evidence-1"],
        properties={"selected_agent": "ReconAgent"},
    )

    assert node.id.startswith("ap-node::")
    assert node.node_type == AttackProcessNodeType.PLANNER_DECISION
    assert node.refs[0].key() == "kg:host-1"
    assert node.evidence_refs == ["evidence-1"]
    assert node.properties["selected_agent"] == "ReconAgent"
    assert node.created_at.tzinfo is not None


def test_attack_process_edge_connects_process_nodes() -> None:
    edge = AttackProcessEdge(
        id="ap-edge-1",
        edge_type=AttackProcessEdgeType.DISPATCHED_TO,
        source="planner-decision-1",
        target="agent-execution-1",
        label="dispatched to",
        properties={"selected_agent": "ReconAgent"},
    )

    serialized = edge.model_dump(mode="json")

    assert edge.edge_type == AttackProcessEdgeType.DISPATCHED_TO
    assert serialized["edge_type"] == "DISPATCHED_TO"
    assert serialized["source"] == "planner-decision-1"
    assert edge.created_at.tzinfo is not None


def test_attack_process_models_forbid_extra_fields() -> None:
    try:
        AttackProcessNode(
            id="node-1",
            node_type=AttackProcessNodeType.BLOCKED_REASON,
            label="Blocked by policy",
            operation_id="op-1",
            unexpected=True,
        )
    except ValidationError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("AttackProcessNode accepted an unexpected field")


def test_attack_process_model_has_no_task_model_imports() -> None:
    source = Path("src/core/models/attack_process.py").read_text(encoding="utf-8")

    assert "from src.core.models.tg" not in source
    assert "import src.core.models.tg" not in source
    assert "TaskGraph" not in source
    assert "TaskNode" not in source
    assert "TaskStatus" not in source
    assert "TaskType" not in source
