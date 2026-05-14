from __future__ import annotations

import pytest

from src.core.workers.probe_adapters import NucleiSafeTemplateAdapter, ProbeAdapterUnavailable
from src.core.workers.tool_runner import ToolExecutionResult


def test_nuclei_destructive_template_is_rejected_by_default() -> None:
    adapter = NucleiSafeTemplateAdapter()

    with pytest.raises(ProbeAdapterUnavailable):
        adapter.build_command(
            target_hint="https://example.test",
            mode="vulnerability_validation",
            metadata={"nuclei_template_tags": ["cve", "destructive"]},
        )


def test_nuclei_rejection_can_be_returned_as_parsed_probe_result() -> None:
    adapter = NucleiSafeTemplateAdapter()

    parsed = adapter.parse_output(
        execution_result=ToolExecutionResult(command=["nuclei"], success=False, category="policy_denied"),
        target_hint="https://example.test",
        mode="vulnerability_validation",
        metadata={"template_tags": "exploit,cve"},
    )

    assert parsed.blocked is True
    assert parsed.blocked_reason == "unsafe_nuclei_template"
    assert parsed.evidence["blocked_tags"] == ["exploit"]


def test_nuclei_jsonl_output_normalizes_to_evidence_and_entities() -> None:
    adapter = NucleiSafeTemplateAdapter()
    execution_result = ToolExecutionResult(
        command=["nuclei", "-jsonl"],
        success=True,
        category="success",
        exit_code=0,
        stdout=(
            '{"template-id":"http-missing-header","info":{"name":"Missing Header",'
            '"severity":"low","tags":["safe","misconfig"]},"matched-at":"https://example.test"}\n'
        ),
    )

    parsed = adapter.parse_output(
        execution_result=execution_result,
        target_hint="https://example.test",
        mode="vulnerability_validation",
        metadata={"nuclei_template_tags": ["safe"]},
    )

    assert parsed.success is True
    assert parsed.evidence["adapter"] == "nuclei"
    assert parsed.evidence["findings"][0]["template_id"] == "http-missing-header"
    assert parsed.entities[0]["type"] == "VulnerabilityCandidate"
