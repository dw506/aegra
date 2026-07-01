from __future__ import annotations

from src.core.models.finding import Finding, RiskScore
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.runtime.report_generator import ReportGenerator


def _state() -> RuntimeState:
    state = RuntimeState(operation_id="op-report", execution=OperationRuntime(operation_id="op-report"))
    state.execution.metadata["control_plane"] = {"audit_redaction_enabled": True}
    state.execution.metadata["evidence_artifacts"] = [
        {
            "evidence_id": "evidence-1",
            "kind": "vulnerability_validation",
            "summary": "curl Authorization: Bearer secret-token",
            "payload_ref": "runtime://worker-results/evidence-1",
            "execution_ref": "execution-1",
            "tool_output_ref": "runtime://worker-results/evidence-1",
            "metadata": {"api_key": "secret-key", "safe": "ok"},
            "refs": [],
            "created_at": "2026-05-12T00:00:00+00:00",
        }
    ]
    finding = Finding(
        finding_id="finding-1",
        title="Validated vuln",
        affected_asset_refs=["svc-1"],
        service_ref="svc-1",
        vulnerability_ref="vuln-1",
        evidence_refs=["evidence-1"],
        validation_status="validated",
        severity="high",
        confidence=0.9,
        false_positive_risk=0.1,
        remediation="patch",
        risk_score=RiskScore(score=80, severity="high"),
    )
    state.execution.metadata["findings"] = [finding.model_dump(mode="json")]
    return state


def test_report_generator_exports_json_csv_and_markdown() -> None:
    state = _state()
    generator = ReportGenerator()

    report = generator.export(state, format="json")
    csv_report = generator.export(state, format="csv")
    md_report = generator.export(state, format="md")

    assert isinstance(report, dict)
    assert report["findings"][0]["finding_id"] == "finding-1"
    assert "finding_id,title,severity" in csv_report
    assert "finding-1" in csv_report
    assert "# Findings Report: op-report" in md_report
    assert "Validated vuln" in md_report


def test_report_generator_redacts_sensitive_evidence_fields() -> None:
    report = ReportGenerator().export(_state(), format="json")
    serialized = str(report)

    assert "secret-key" not in serialized
    assert "secret-token" not in serialized
    assert "[REDACTED]" in serialized
