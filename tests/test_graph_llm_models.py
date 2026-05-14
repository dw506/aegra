from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.agents.graph_llm_models import (
    GraphLLMPlanProposal,
    GraphLLMPlanValidationResult,
    GraphLLMRankAdjustment,
    GraphLLMTaskProposal,
)
from src.core.agents.llm_decision import validate_graph_plan_proposal
from src.core.models.ag import GraphRef
from src.core.runtime.policy import RuntimePolicy


def _host_ref(ref_id: str = "host-1") -> GraphRef:
    return GraphRef(graph="kg", ref_id=ref_id, ref_type="Host", label=ref_id)


def test_graph_llm_models_accept_bounded_plan_payload() -> None:
    host_ref = _host_ref()
    proposal = GraphLLMPlanProposal(
        task_proposals=[
            GraphLLMTaskProposal(
                task_type="SERVICE_VALIDATION",
                target_refs=[host_ref],
                rationale="confirm exposed service",
                expected_evidence=["service banner"],
                estimated_risk=0.1,
                estimated_noise=0.1,
            )
        ],
        rank_adjustments=[
            GraphLLMRankAdjustment(
                target_ref=host_ref,
                score_delta=0.1,
                rationale="higher evidence gain",
            )
        ],
    )

    assert proposal.task_proposals[0].target_refs == [host_ref]
    assert proposal.rank_adjustments[0].score_delta == 0.1


def test_graph_llm_models_reject_extra_command_field() -> None:
    with pytest.raises(ValidationError):
        GraphLLMTaskProposal.model_validate(
            {
                "task_type": "SERVICE_VALIDATION",
                "target_refs": [_host_ref().model_dump(mode="json")],
                "command": "nmap -sV host-1",
            }
        )


def test_graph_llm_validation_result_helpers_are_structured() -> None:
    accepted = GraphLLMPlanValidationResult.accepted_result(
        sanitized_payload={"proposal_id": "proposal-1"},
        requires_human_review=True,
    )
    rejected = GraphLLMPlanValidationResult.rejected_result(reason="unknown ref")

    assert accepted.accepted is True
    assert accepted.requires_human_review is True
    assert rejected.accepted is False


def test_validate_graph_plan_proposal_accepts_and_sanitizes() -> None:
    host_ref = _host_ref()
    result = validate_graph_plan_proposal(
        {
            "proposal_id": "proposal-1",
            "task_proposals": [
                {
                    "proposal_id": "task-1",
                    "task_type": "SERVICE_VALIDATION",
                    "target_refs": [host_ref.model_dump(mode="json")],
                    "rationale": "confirm service",
                    "expected_evidence": ["service metadata"],
                    "estimated_risk": 0.1,
                    "estimated_noise": 0.1,
                    "metadata": {"api_key_hint": "drop", "reason": "evidence_gain"},
                }
            ],
        },
        visible_refs={host_ref},
    )

    assert result.accepted is True
    task = result.sanitized_payload["task_proposals"][0]
    assert task["task_type"] == "SERVICE_VALIDATION"
    assert task["target_refs"] == [host_ref.model_dump(mode="json")]
    assert "api_key_hint" not in task["metadata"]


def test_validate_graph_plan_proposal_rejects_unknown_ref() -> None:
    result = validate_graph_plan_proposal(
        GraphLLMPlanProposal(
            task_proposals=[
                GraphLLMTaskProposal(
                    task_type="SERVICE_VALIDATION",
                    target_refs=[_host_ref("host-2")],
                )
            ]
        ),
        visible_refs={_host_ref("host-1")},
    )

    assert result.accepted is False
    assert "unknown ref" in result.reason


def test_validate_graph_plan_proposal_rejects_forbidden_command_payloads() -> None:
    host_ref = _host_ref()
    for key in ["command", "shell", "payload", "reverse_shell"]:
        result = validate_graph_plan_proposal(
            {
                "task_proposals": [
                    {
                        "task_type": "SERVICE_VALIDATION",
                        "target_refs": [host_ref.model_dump(mode="json")],
                        key: "unsafe",
                    }
                ]
            },
            visible_refs={host_ref},
        )

        assert result.accepted is False
        assert key in result.reason


def test_validate_graph_plan_proposal_rejects_invalid_task_type() -> None:
    host_ref = _host_ref()
    result = validate_graph_plan_proposal(
        GraphLLMPlanProposal(
            task_proposals=[
                GraphLLMTaskProposal(
                    task_type="RUN_SHELL",
                    target_refs=[host_ref],
                )
            ]
        ),
        visible_refs={host_ref},
    )

    assert result.accepted is False
    assert "task_type" in result.reason


def test_validate_graph_plan_proposal_marks_high_risk_for_human_review() -> None:
    host_ref = _host_ref()
    result = validate_graph_plan_proposal(
        GraphLLMPlanProposal(
            task_proposals=[
                GraphLLMTaskProposal(
                    task_type="VULNERABILITY_VALIDATION",
                    target_refs=[host_ref],
                    estimated_risk=0.8,
                    estimated_noise=0.2,
                )
            ]
        ),
        visible_refs={host_ref},
    )

    assert result.accepted is True
    assert result.requires_human_review is True
    assert result.sanitized_payload["task_proposals"][0]["requires_human_review"] is True


def test_validate_graph_plan_proposal_rejects_runtime_policy_scope_violation() -> None:
    host_ref = _host_ref("10.0.0.2")
    result = validate_graph_plan_proposal(
        GraphLLMPlanProposal(
            task_proposals=[
                GraphLLMTaskProposal(
                    task_type="SERVICE_VALIDATION",
                    target_refs=[host_ref],
                )
            ]
        ),
        visible_refs={host_ref},
        policy_context=RuntimePolicy(authorized_hosts=["10.0.0.1"]),
    )

    assert result.accepted is False
    assert "policy scope" in result.reason
