"""Report generation for findings and evidence."""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Literal

from src.core.models.runtime import RuntimeState, utc_now


ReportFormat = Literal["json", "csv", "md"]

_REDACTED_VALUE = "[REDACTED]"
_SENSITIVE_KEY_MARKERS = ("token", "password", "secret", "api_key", "apikey", "authorization")
_INLINE_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)([^\s,;]+)"),
    re.compile(r"(?i)(\b(?:token|password|secret|api[_-]?key)\b\s*[:=]\s*)([^\s,;]+)"),
)


class ReportGenerator:
    """Build sanitized JSON, CSV and Markdown finding reports."""

    def export(self, state: RuntimeState, *, format: ReportFormat = "json") -> dict[str, Any] | str:
        report = self.build_report(state)
        if format == "json":
            return report
        if format == "csv":
            return self.to_csv(report)
        if format == "md":
            return self.to_markdown(report)
        raise ValueError("format must be one of: json, csv, md")

    def build_report(self, state: RuntimeState) -> dict[str, Any]:
        payload = {
            "operation_id": state.operation_id,
            "exported_at": utc_now().isoformat(),
            "operation_status": state.operation_status.value,
            "findings": self.findings(state),
            "evidence": self.evidence(state),
            "audit": self.audit(state),
        }
        return self._sanitize(payload, redaction_enabled=self._redaction_enabled(state))

    def findings(self, state: RuntimeState) -> list[dict[str, Any]]:
        return self._metadata_items(state, "findings")

    def evidence(self, state: RuntimeState) -> list[dict[str, Any]]:
        return self._metadata_items(state, "evidence_artifacts")

    def graph(self, state: RuntimeState) -> dict[str, Any]:
        findings = self.findings(state)
        evidence = self.evidence(state)
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        seen_nodes: set[str] = set()

        def add_node(node_id: str, node_type: str, label: str | None = None, **props: Any) -> None:
            if not node_id or node_id in seen_nodes:
                return
            seen_nodes.add(node_id)
            nodes.append({"id": node_id, "type": node_type, "label": label or node_id, **props})

        for finding in findings:
            finding_id = str(finding.get("finding_id"))
            add_node(finding_id, "Finding", finding.get("title"), severity=finding.get("severity"))
            service_ref = str(finding.get("service_ref") or "")
            vulnerability_ref = str(finding.get("vulnerability_ref") or "")
            evidence_refs = [str(item) for item in finding.get("evidence_refs", []) if str(item)]
            host_refs = [str(item) for item in finding.get("affected_asset_refs", []) if str(item)]
            for key, node_type, relation in (
                ("service_ref", "Service", "AFFECTS"),
                ("vulnerability_ref", "Vulnerability", "DERIVED_FROM"),
            ):
                ref = str(finding.get(key) or "")
                if not ref:
                    continue
                add_node(ref, node_type)
                edges.append({"source": finding_id, "target": ref, "type": relation})
            for evidence_ref in evidence_refs:
                add_node(evidence_ref, "Evidence")
                edges.append({"source": finding_id, "target": evidence_ref, "type": "SUPPORTED_BY"})
            for host_ref in host_refs:
                add_node(host_ref, "Host")
                edges.append({"source": finding_id, "target": host_ref, "type": "AFFECTS"})
                if service_ref:
                    edges.append({"source": host_ref, "target": service_ref, "type": "EXPOSES"})
            if service_ref and vulnerability_ref:
                edges.append({"source": service_ref, "target": vulnerability_ref, "type": "HAS_VULNERABILITY"})
            if vulnerability_ref:
                for evidence_ref in evidence_refs:
                    edges.append({"source": vulnerability_ref, "target": evidence_ref, "type": "SUPPORTED_BY"})

        for item in evidence:
            evidence_id = str(item.get("evidence_id"))
            add_node(evidence_id, "Evidence", item.get("summary"))
            execution_ref = str(item.get("execution_ref") or "")
            if execution_ref:
                add_node(execution_ref, "Execution")
                edges.append({"source": evidence_id, "target": execution_ref, "type": "DERIVED_FROM"})

        return self._sanitize(
            {"nodes": nodes, "edges": edges},
            redaction_enabled=self._redaction_enabled(state),
        )

    @staticmethod
    def to_csv(report: dict[str, Any]) -> str:
        output = io.StringIO()
        fieldnames = [
            "finding_id",
            "title",
            "severity",
            "risk_score",
            "validation_status",
            "service_ref",
            "vulnerability_ref",
            "evidence_refs",
            "confidence",
            "false_positive_risk",
            "created_at",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for finding in report.get("findings", []):
            risk = finding.get("risk_score") if isinstance(finding.get("risk_score"), dict) else {}
            writer.writerow(
                {
                    "finding_id": finding.get("finding_id"),
                    "title": finding.get("title"),
                    "severity": finding.get("severity"),
                    "risk_score": risk.get("score"),
                    "validation_status": finding.get("validation_status"),
                    "service_ref": finding.get("service_ref"),
                    "vulnerability_ref": finding.get("vulnerability_ref"),
                    "evidence_refs": ";".join(finding.get("evidence_refs", [])),
                    "confidence": finding.get("confidence"),
                    "false_positive_risk": finding.get("false_positive_risk"),
                    "created_at": finding.get("created_at"),
                }
            )
        return output.getvalue()

    @staticmethod
    def to_markdown(report: dict[str, Any]) -> str:
        lines = [
            f"# Findings Report: {report.get('operation_id')}",
            "",
            f"Exported at: {report.get('exported_at')}",
            "",
        ]
        findings = list(report.get("findings", []))
        if not findings:
            lines.extend(["No findings.", ""])
            return "\n".join(lines)
        for finding in findings:
            risk = finding.get("risk_score") if isinstance(finding.get("risk_score"), dict) else {}
            lines.extend(
                [
                    f"## {finding.get('title')}",
                    "",
                    f"- Finding ID: `{finding.get('finding_id')}`",
                    f"- Severity: {finding.get('severity')}",
                    f"- Risk score: {risk.get('score')}",
                    f"- Validation: {finding.get('validation_status')}",
                    f"- Service: `{finding.get('service_ref')}`",
                    f"- Vulnerability: `{finding.get('vulnerability_ref')}`",
                    f"- Evidence: {', '.join(f'`{item}`' for item in finding.get('evidence_refs', []))}",
                    f"- Remediation: {finding.get('remediation') or 'Review vendor guidance and apply compensating controls.'}",
                    "",
                ]
            )
        return "\n".join(lines)

    def audit(self, state: RuntimeState) -> list[dict[str, Any]]:
        return self._metadata_items(state, "finding_audit")

    @staticmethod
    def _metadata_items(state: RuntimeState, key: str) -> list[dict[str, Any]]:
        value = state.execution.metadata.get(key, [])
        if not isinstance(value, list):
            return []
        return [dict(item) for item in value if isinstance(item, dict)]

    @staticmethod
    def _redaction_enabled(state: RuntimeState) -> bool:
        control_plane = state.execution.metadata.get("control_plane", {})
        if not isinstance(control_plane, dict):
            return True
        return bool(control_plane.get("audit_redaction_enabled", True))

    def _sanitize(self, value: Any, *, redaction_enabled: bool, key: str | None = None) -> Any:
        if key is not None and redaction_enabled and self._is_sensitive_key(key):
            return _REDACTED_VALUE
        if isinstance(value, dict):
            return {str(item_key): self._sanitize(item_value, redaction_enabled=redaction_enabled, key=str(item_key)) for item_key, item_value in value.items()}
        if isinstance(value, list):
            return [self._sanitize(item, redaction_enabled=redaction_enabled, key=key) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize(item, redaction_enabled=redaction_enabled, key=key) for item in value]
        if isinstance(value, str) and redaction_enabled:
            return self._redact_inline(value)
        return value

    @staticmethod
    def _is_sensitive_key(key: str) -> bool:
        lowered = key.lower()
        return any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS) or lowered.endswith("_token")

    @staticmethod
    def _redact_inline(text: str) -> str:
        redacted = text
        for pattern in _INLINE_SECRET_PATTERNS:
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        return redacted


def report_to_json_string(report: dict[str, Any]) -> str:
    """Return a stable JSON representation for CLI-style callers."""

    return json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)


__all__ = ["ReportFormat", "ReportGenerator", "report_to_json_string"]
