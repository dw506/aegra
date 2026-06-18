"""Deterministic fact extractor for successful ToolTraces.

This module extracts KG/AG/Runtime-writable facts from successful tool call
outputs, independent of LLM post-processing. It is called by ResultApplier
when a ToolTrace has success=True, even if subsequent LLM processing failed.

Facts extracted here are conservative (only from clearly successful tool calls)
and generic (no hardcoded environment values).

Extraction is tool-name-driven, not environment-driven.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExtractedFact(BaseModel):
    """One fact extracted from a ToolTrace output."""

    model_config = ConfigDict(extra="forbid")

    fact_type: str
    entity_type: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    evidence_ref: str = ""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    source_tool: str = ""
    zone_ref: str = ""


class FactExtractionResult(BaseModel):
    """Result of extracting facts from one ToolTrace."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    tool_name: str
    facts: list[ExtractedFact] = Field(default_factory=list)
    writeback_status: str = "pending"
    error: str | None = None
    llm_postprocess_failed: bool = False


# ---------------------------------------------------------------------------
# Tool-specific extractors
# ---------------------------------------------------------------------------


def _extract_nmap_scan(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract Host/Service facts from nmap_scan output."""
    facts: list[ExtractedFact] = []
    parsed = trace.get("parsed_output") or {}
    stdout = str(trace.get("stdout") or "")
    args = trace.get("arguments") or {}
    target = str(args.get("target") or args.get("host") or "")
    zone_ref = str(args.get("zone_ref") or "")

    # Parse open ports from stdout
    for line in stdout.splitlines():
        m = re.match(r"(\d+)/(tcp|udp)\s+open\s+(\S+)", line)
        if m:
            port = int(m.group(1))
            proto = m.group(2)
            service_name = m.group(3)
            facts.append(
                ExtractedFact(
                    fact_type="Service",
                    entity_type="Service",
                    label=f"{service_name}:{port}/{proto}@{target}",
                    properties={
                        "port": port,
                        "protocol": proto,
                        "service_name": service_name,
                        "address": target,
                        "zone_ref": zone_ref,
                    },
                    confidence=0.9,
                    source_tool="nmap_scan",
                    zone_ref=zone_ref,
                )
            )

    # Extract hosts from parsed output
    hosts = parsed.get("hosts") or []
    if isinstance(hosts, list):
        for host in hosts:
            addr = str(host.get("address") or host.get("ip") or "")
            if addr:
                facts.append(
                    ExtractedFact(
                        fact_type="Host",
                        entity_type="Host",
                        label=f"host:{addr}",
                        properties={"address": addr, "zone_ref": zone_ref, **host},
                        confidence=0.9,
                        source_tool="nmap_scan",
                        zone_ref=zone_ref,
                    )
                )

    # If no structured parsing, still record a Host if target is known
    if not facts and target:
        facts.append(
            ExtractedFact(
                fact_type="Host",
                entity_type="Host",
                label=f"host:{target}",
                properties={"address": target, "zone_ref": zone_ref},
                confidence=0.7,
                source_tool="nmap_scan",
                zone_ref=zone_ref,
            )
        )
    return facts


def _extract_http_probe(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract WebService/Evidence facts from http_probe output."""
    facts: list[ExtractedFact] = []
    parsed = trace.get("parsed_output") or {}
    args = trace.get("arguments") or {}
    url = str(args.get("url") or args.get("target") or "")
    zone_ref = str(args.get("zone_ref") or "")
    status = parsed.get("status_code") or parsed.get("status")
    if url:
        facts.append(
            ExtractedFact(
                fact_type="Service",
                entity_type="Service",
                label=f"http-service:{url}",
                properties={
                    "url": url,
                    "status_code": status,
                    "service_name": "http",
                    "zone_ref": zone_ref,
                },
                confidence=0.8,
                source_tool="http_probe",
                zone_ref=zone_ref,
            )
        )
    return facts


def _extract_web_fingerprint(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract Technology/Fingerprint Evidence from web_fingerprint output."""
    facts: list[ExtractedFact] = []
    parsed = trace.get("parsed_output") or {}
    args = trace.get("arguments") or {}
    url = str(args.get("url") or args.get("target") or "")
    zone_ref = str(args.get("zone_ref") or "")
    technologies = parsed.get("technologies") or parsed.get("fingerprints") or []
    if isinstance(technologies, list):
        for tech in technologies:
            tech_name = str(tech.get("name") or tech) if isinstance(tech, dict) else str(tech)
            facts.append(
                ExtractedFact(
                    fact_type="Evidence",
                    entity_type="Evidence",
                    label=f"fingerprint:{tech_name}@{url}",
                    properties={
                        "technology": tech_name,
                        "url": url,
                        "evidence_kind": "web_fingerprint",
                        "zone_ref": zone_ref,
                    },
                    confidence=0.75,
                    source_tool="web_fingerprint",
                    zone_ref=zone_ref,
                )
            )
    elif url:
        facts.append(
            ExtractedFact(
                fact_type="Evidence",
                entity_type="Evidence",
                label=f"fingerprint-probe:{url}",
                properties={"url": url, "evidence_kind": "web_fingerprint", "zone_ref": zone_ref},
                confidence=0.6,
                source_tool="web_fingerprint",
                zone_ref=zone_ref,
            )
        )
    return facts


def _extract_vuln_profile_match(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract VulnerabilityCandidate from vulnerability profile match."""
    facts: list[ExtractedFact] = []
    parsed = trace.get("parsed_output") or {}
    args = trace.get("arguments") or {}
    matches = parsed.get("matches") or parsed.get("candidates") or []
    target = str(args.get("target") or "")
    zone_ref = str(args.get("zone_ref") or "")
    if isinstance(matches, list):
        for match in matches:
            vuln_id = str(match.get("vuln_profile_id") or match.get("id") or "unknown")
            confidence = float(match.get("confidence") or 0.5)
            facts.append(
                ExtractedFact(
                    fact_type="VulnerabilityCandidate",
                    entity_type="VulnerabilityCandidate",
                    label=f"vuln-candidate:{vuln_id}@{target}",
                    properties={
                        "vuln_profile_id": vuln_id,
                        "target_ref": target,
                        "status": "candidate",
                        "zone_ref": zone_ref,
                        **{k: v for k, v in match.items() if k not in ("vuln_profile_id",)},
                    },
                    confidence=confidence,
                    source_tool="vuln_profile_match",
                    zone_ref=zone_ref,
                )
            )
    return facts


def _extract_exploit_execute(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract ExploitCapability/Session from successful exploit execution.

    Only creates ValidatedVulnerability / ExploitCapability if the exploit
    explicitly succeeded (success=True on the trace).
    """
    facts: list[ExtractedFact] = []
    parsed = trace.get("parsed_output") or {}
    args = trace.get("arguments") or {}
    target = str(args.get("target") or "")
    vuln_id = str(args.get("exploit_profile_id") or args.get("vuln_profile_id") or "")
    zone_ref = str(args.get("zone_ref") or "")
    runtime_hints = parsed.get("runtime_hints") if isinstance(parsed.get("runtime_hints"), dict) else {}
    exploit_success = parsed.get("exploit_success") or parsed.get("success") or trace.get("success")
    if exploit_success:
        facts.append(
            ExtractedFact(
                fact_type="ExploitCapability",
                entity_type="ExploitCapability",
                label=f"exploit-capability:{vuln_id}@{target}",
                properties={
                    "vuln_ref": vuln_id,
                    "target_ref": target,
                    "status": "active",
                    "zone_ref": zone_ref,
                    "exploit_profile_id": vuln_id,
                    "capability_kind": runtime_hints.get("capability_kind"),
                    "post_access_observable": bool(runtime_hints.get("post_access_observable")),
                    "observable_zones": list(runtime_hints.get("observable_zones") or []),
                    "next_tools": list(runtime_hints.get("next_tools") or []),
                },
                confidence=0.9,
                source_tool="lab_authorized_exploit_execute",
                zone_ref=zone_ref,
            )
        )
    # Also record as Evidence regardless of exploit success/failure
    facts.append(
        ExtractedFact(
            fact_type="Evidence",
            entity_type="Evidence",
            label=f"exploit-attempt:{vuln_id}@{target}",
            properties={
                "evidence_kind": "exploit_attempt",
                "target": target,
                "vuln_profile_id": vuln_id,
                "outcome": "success" if exploit_success else "failure",
                "zone_ref": zone_ref,
            },
            confidence=0.9,
            source_tool="lab_authorized_exploit_execute",
            zone_ref=zone_ref,
        )
    )
    return facts


def _extract_post_access(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract PostAccessObservation/LabHint/LabFlag from post-access tools."""
    facts: list[ExtractedFact] = []
    parsed = trace.get("parsed_output") or {}
    args = trace.get("arguments") or {}
    target = str(args.get("target") or "")
    zone_ref = str(args.get("zone_ref") or "")
    tool_name = str(trace.get("tool_name") or "")

    if "lab_marker" in tool_name or "read_lab" in tool_name:
        entity_type = "LabFlag"
        evidence_kind = "lab_flag"
    elif "hint" in tool_name:
        entity_type = "LabHint"
        evidence_kind = "lab_hint"
    else:
        entity_type = "PostAccessObservation"
        evidence_kind = "post_access"

    content_summary = str(parsed.get("content_summary") or parsed.get("summary") or "")
    facts.append(
        ExtractedFact(
            fact_type=entity_type,
            entity_type=entity_type,
            label=f"{evidence_kind}:{target}",
            properties={
                "evidence_kind": evidence_kind,
                "target_ref": target,
                "content_summary": content_summary[:200],  # truncate, never store raw flags
                "zone_ref": zone_ref,
            },
            confidence=0.9,
            source_tool=tool_name,
            zone_ref=zone_ref,
        )
    )
    return facts


def _extract_credential(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract Credential facts from credential discovery tools."""
    facts: list[ExtractedFact] = []
    parsed = trace.get("parsed_output") or {}
    args = trace.get("arguments") or {}
    target = str(args.get("target") or "")
    zone_ref = str(args.get("zone_ref") or "")
    credentials = parsed.get("credentials") or parsed.get("results") or []
    if isinstance(credentials, list) and credentials:
        for cred in credentials:
            cred_type = str(cred.get("type") or "password")
            principal = str(cred.get("username") or cred.get("principal") or "unknown")
            facts.append(
                ExtractedFact(
                    fact_type="Credential",
                    entity_type="Credential",
                    label=f"credential:{cred_type}:{principal}@{target}",
                    properties={
                        "credential_kind": cred_type,
                        "principal": principal,
                        "target_ref": target,
                        "zone_ref": zone_ref,
                    },
                    confidence=0.8,
                    source_tool=str(trace.get("tool_name") or ""),
                    zone_ref=zone_ref,
                )
            )
    elif not credentials:
        # Still record a hint if credential discovery ran successfully
        facts.append(
            ExtractedFact(
                fact_type="Evidence",
                entity_type="Evidence",
                label=f"credential-probe:{target}",
                properties={
                    "evidence_kind": "credential_probe",
                    "target_ref": target,
                    "zone_ref": zone_ref,
                },
                confidence=0.7,
                source_tool=str(trace.get("tool_name") or ""),
                zone_ref=zone_ref,
            )
        )
    return facts


def _extract_session(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract Session facts from session management tools."""
    facts: list[ExtractedFact] = []
    parsed = trace.get("parsed_output") or {}
    args = trace.get("arguments") or {}
    target = str(args.get("target") or "")
    zone_ref = str(args.get("zone_ref") or "")
    session_id = str(parsed.get("session_id") or parsed.get("id") or "")
    if session_id:
        facts.append(
            ExtractedFact(
                fact_type="Session",
                entity_type="Session",
                label=f"session:{session_id}@{target}",
                properties={
                    "session_id": session_id,
                    "bound_target": target,
                    "session_kind": str(args.get("session_kind") or "shell"),
                    "status": "active",
                    "zone_ref": zone_ref,
                },
                confidence=0.9,
                source_tool=str(trace.get("tool_name") or ""),
                zone_ref=zone_ref,
            )
        )
    return facts


def _extract_pivot_route(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract PivotRoute facts from pivot tools."""
    facts: list[ExtractedFact] = []
    parsed = trace.get("parsed_output") or {}
    args = trace.get("arguments") or {}
    from_zone = str(args.get("from_zone_ref") or args.get("from_zone") or "entry")
    to_zone = str(args.get("to_zone_ref") or args.get("to_zone") or "restricted")
    via_host = str(args.get("via_host") or args.get("pivot_host") or "")
    zone_ref = to_zone
    route_id = str(parsed.get("route_id") or f"pivot-{from_zone}-to-{to_zone}")
    facts.append(
        ExtractedFact(
            fact_type="PivotRoute",
            entity_type="PivotRoute",
            label=f"pivot:{from_zone}→{to_zone}",
            properties={
                "route_id": route_id,
                "from_zone_ref": from_zone,
                "to_zone_ref": to_zone,
                "via_host": via_host,
                "status": "active",
                "route_type": "pivot_route",
                "zone_ref": zone_ref,
            },
            confidence=0.9,
            source_tool=str(trace.get("tool_name") or ""),
            zone_ref=zone_ref,
        )
    )
    return facts


def _extract_internal_discovery(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract Service facts for restricted-zone discovery."""
    facts: list[ExtractedFact] = []
    parsed = trace.get("parsed_output") or {}
    args = trace.get("arguments") or {}
    zone_ref = str(args.get("zone_ref") or "restricted")
    hosts = parsed.get("hosts") or []
    services = parsed.get("services") or []
    for service in services:
        port = service.get("port")
        svc_name = str(service.get("service_name") or service.get("name") or "unknown")
        addr = str(service.get("address") or service.get("host") or "")
        facts.append(
            ExtractedFact(
                fact_type="Service",
                entity_type="Service",
                label=f"internal-service:{svc_name}:{port}@{addr}",
                properties={
                    "port": port,
                    "service_name": svc_name,
                    "address": addr,
                    "zone_ref": zone_ref,
                    "internal": True,
                },
                confidence=0.85,
                source_tool=str(trace.get("tool_name") or ""),
                zone_ref=zone_ref,
            )
        )
    for host in hosts if isinstance(hosts, list) else []:
        addr = str(host.get("address") or host.get("ip") or host)
        facts.append(
            ExtractedFact(
                fact_type="Host",
                entity_type="Host",
                label=f"internal-host:{addr}",
                properties={"address": addr, "zone_ref": zone_ref, "internal": True},
                confidence=0.85,
                source_tool=str(trace.get("tool_name") or ""),
                zone_ref=zone_ref,
            )
        )
    return facts


def _extract_goal_check(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract GoalCheck/GoalProof from goal check tools."""
    facts: list[ExtractedFact] = []
    parsed = trace.get("parsed_output") or {}
    args = trace.get("arguments") or {}
    goal_id = str(args.get("goal_id") or parsed.get("goal_id") or "")
    passed = bool(parsed.get("passed") or parsed.get("success"))
    # proof_token is opaque and safe to store
    proof_token = str(parsed.get("proof_token") or "")
    redacted_summary = str(parsed.get("redacted_summary") or parsed.get("summary") or "")
    if passed and proof_token:
        facts.append(
            ExtractedFact(
                fact_type="GoalProof",
                entity_type="GoalProof",
                label=f"goal-proof:{goal_id}",
                properties={
                    "goal_id": goal_id,
                    "passed": True,
                    "proof_token": proof_token,
                    "redacted_summary": redacted_summary,
                },
                confidence=1.0,
                source_tool=str(trace.get("tool_name") or ""),
            )
        )
    facts.append(
        ExtractedFact(
            fact_type="GoalCheck",
            entity_type="GoalCheck",
            label=f"goal-check:{goal_id}",
            properties={
                "goal_id": goal_id,
                "passed": passed,
                "redacted_summary": redacted_summary,
            },
            confidence=0.9,
            source_tool=str(trace.get("tool_name") or ""),
        )
    )
    return facts


def _extract_controlled_data_read(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract redacted proof facts from bounded data-service reads."""
    facts: list[ExtractedFact] = []
    parsed = trace.get("parsed_output") or {}
    args = trace.get("arguments") or {}
    for evidence in parsed.get("evidence") or []:
        if not isinstance(evidence, dict):
            continue
        host = str(evidence.get("host") or args.get("host") or "")
        port = evidence.get("port") or args.get("port")
        service = str(evidence.get("service") or args.get("service") or "data_service")
        facts.append(
            ExtractedFact(
                fact_type="ControlledDataReadProof",
                entity_type="ControlledDataReadProof",
                label=f"controlled-data-proof:{service}:{port}@{host}",
                properties={
                    "host": host,
                    "port": port,
                    "service_name": service,
                    "row_count": evidence.get("row_count"),
                    "proof_sha256": evidence.get("proof_sha256"),
                    "redacted": True,
                },
                confidence=0.95,
                source_tool=str(trace.get("tool_name") or ""),
                zone_ref=str(args.get("zone_ref") or "restricted"),
            )
        )
    return facts


# ---------------------------------------------------------------------------
# Tool name → extractor mapping
# ---------------------------------------------------------------------------

_TOOL_EXTRACTORS: dict[str, Any] = {
    "nmap_scan": _extract_nmap_scan,
    "http_probe": _extract_http_probe,
    "web_fingerprint": _extract_web_fingerprint,
    "web_discover": _extract_web_fingerprint,
    "fingerprint_extract": _extract_web_fingerprint,
    "vulnerability_profile_match": _extract_vuln_profile_match,
    "vuln_profile_match": _extract_vuln_profile_match,
    "lab_authorized_exploit_execute": _extract_exploit_execute,
    "exploit_execute": _extract_exploit_execute,
    "post_access_observe": _extract_post_access,
    "read_lab_marker": _extract_post_access,
    "list_lab_hints": _extract_post_access,
    "credential_discover_lab": _extract_credential,
    "credential_check": _extract_credential,
    "session_open_lab": _extract_session,
    "session_probe": _extract_session,
    "pivot_route_probe": _extract_pivot_route,
    "pivot_route_register": _extract_pivot_route,
    "pivoted_nmap_scan": _extract_internal_discovery,
    "internal_service_discover": _extract_internal_discovery,
    "goal_check": _extract_goal_check,
    "internal_goal_check": _extract_goal_check,
    "chain_goal_check": _extract_goal_check,
    "controlled_data_read_proof": _extract_controlled_data_read,
}


class ToolTraceFactExtractor:
    """Extract KG-writable facts from successful ToolTraces.

    Called by ResultApplier after every stage cycle, even when LLM
    post-processing has failed. Only processes traces with success=True.
    """

    def extract(self, tool_trace: dict[str, Any] | Any) -> FactExtractionResult:
        """Extract facts from one tool trace dict (or ToolTrace Pydantic model)."""
        if hasattr(tool_trace, "model_dump"):
            trace = tool_trace.model_dump(mode="json")
        else:
            trace = dict(tool_trace)

        trace_id = str(trace.get("trace_id") or "unknown")
        tool_name = str(trace.get("tool_name") or "")
        success = bool(trace.get("success"))

        if not success:
            return FactExtractionResult(
                trace_id=trace_id,
                tool_name=tool_name,
                facts=[],
                writeback_status="skipped_tool_failed",
            )

        extractor = _TOOL_EXTRACTORS.get(tool_name)
        if extractor is None:
            # Fallback: extract a generic Evidence node
            facts = [
                ExtractedFact(
                    fact_type="Evidence",
                    entity_type="Evidence",
                    label=f"tool-output:{tool_name}",
                    properties={"tool_name": tool_name, "evidence_kind": "tool_output"},
                    confidence=0.5,
                    source_tool=tool_name,
                )
            ]
            return FactExtractionResult(
                trace_id=trace_id,
                tool_name=tool_name,
                facts=facts,
                writeback_status="extracted_generic",
            )

        try:
            facts = extractor(trace)
            for fact in facts:
                fact.evidence_ref = trace_id
            return FactExtractionResult(
                trace_id=trace_id,
                tool_name=tool_name,
                facts=facts,
                writeback_status="extracted",
            )
        except Exception as exc:
            return FactExtractionResult(
                trace_id=trace_id,
                tool_name=tool_name,
                facts=[],
                writeback_status="extraction_error",
                error=str(exc),
            )

    def extract_all(self, tool_traces: list[Any]) -> list[FactExtractionResult]:
        """Extract facts from a list of tool traces."""
        return [self.extract(t) for t in tool_traces]
