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
    AttackStepNode,
    stable_node_id,
)


def test_attack_process_node_records_attack_step_result() -> None:
    node_id = stable_node_id(
        "ap-node",
        {
            "operation_id": "op-1",
            "cycle_index": 1,
            "node_type": AttackProcessNodeType.ATTACK_STEP.value,
        },
    )

    node = AttackStepNode(
        id=node_id,
        label="Recon completed",
        operation_id="op-1",
        cycle_index=1,
        agent_name="execution_agent",
        status="succeeded",
        summary="Recon found an exposed service.",
        evidence_refs=["evidence-1"],
        properties={"execution_id": "exec-1"},
    )

    assert node.id.startswith("ap-node::")
    assert node.node_type == AttackProcessNodeType.ATTACK_STEP
    assert node.evidence_refs == ["evidence-1"]
    assert node.properties["execution_id"] == "exec-1"
    assert node.created_at.tzinfo is not None


def test_attack_process_edge_connects_attack_steps() -> None:
    edge = AttackProcessEdge(
        id="ap-edge-1",
        edge_type=AttackProcessEdgeType.NEXT,
        source="attack-step-1",
        target="attack-step-2",
        label="next",
    )

    serialized = edge.model_dump(mode="json")

    assert edge.edge_type == AttackProcessEdgeType.NEXT
    assert serialized["edge_type"] == "NEXT"
    assert serialized["source"] == "attack-step-1"
    assert edge.created_at.tzinfo is not None


def test_attack_process_models_forbid_extra_fields() -> None:
    try:
        AttackProcessNode(
            id="node-1",
            node_type=AttackProcessNodeType.ATTACK_STEP,
            label="Attack step",
            operation_id="op-1",
            unexpected=True,
        )
    except ValidationError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("AttackProcessNode accepted an unexpected field")


def test_attack_process_model_has_no_task_model_imports() -> None:
    source = Path("src/core/models/attack_process.py").read_text(encoding="utf-8")
    legacy_module = "src.core.models." + "t" + "g"
    legacy_prefix = "Task"

    assert f"from {legacy_module}" not in source
    assert f"import {legacy_module}" not in source
    assert legacy_prefix + "Graph" not in source
    assert legacy_prefix + "Node" not in source
    assert legacy_prefix + "Status" not in source
    assert legacy_prefix + "Type" not in source
