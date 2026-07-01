"""Deterministic fact extractor for successful ToolTraces.

This module extracts KG/AG/Runtime-writable facts from successful tool call
outputs, independent of LLM post-processing. It is called by ResultApplier
when a ToolTrace has success=True, even if subsequent LLM processing failed.

Facts extracted here are conservative (only from clearly successful tool calls)
and generic (no hardcoded environment values).

Extraction prefers the tool-declared ``fact_kind`` semantic, falling back to a
tool-name map for legacy tools. It never mints environment-specific node types.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

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
    # A single-IP target may anchor services; a CIDR/range target must NOT be used
    # as an address (it is not locatable). Per-host attribution comes from the
    # "Nmap scan report for <ip>" lines below.
    target_ip = target if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", target) else ""

    # Parse open ports from stdout, attributing each service to the host most
    # recently named by an "Nmap scan report for <ip>" line so a range/CIDR scan
    # yields per-host addresses (not the range string).
    current_host = target_ip
    seen_hosts: set[str] = set()
    for line in stdout.splitlines():
        hm = re.search(r"Nmap scan report for (?:[^()]*\()?(\d{1,3}(?:\.\d{1,3}){3})\)?", line)
        if hm:
            current_host = hm.group(1)
            if current_host not in seen_hosts:
                seen_hosts.add(current_host)
                facts.append(
                    ExtractedFact(
                        fact_type="Host",
                        entity_type="Host",
                        label=f"host:{current_host}",
                        properties={"address": current_host, "zone_ref": zone_ref},
                        confidence=0.9,
                        source_tool="nmap_scan",
                        zone_ref=zone_ref,
                    )
                )
            continue
        m = re.match(r"(\d+)/(tcp|udp)\s+open\s+(\S+)", line)
        if m:
            port = int(m.group(1))
            proto = m.group(2)
            service_name = m.group(3)
            facts.append(
                ExtractedFact(
                    fact_type="Service",
                    entity_type="Service",
                    label=f"{service_name}:{port}/{proto}@{current_host or target}",
                    properties={
                        "port": port,
                        "protocol": proto,
                        "service_name": service_name,
                        "address": current_host or None,
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


def _extract_run_command(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract conservative service facts from generic command output.

    ``run_command`` is intentionally freeform, so only parse well-known scanner
    shapes: nmap output and simple host:port sweep lines. This covers live
    bounded recon without inventing facts from arbitrary shell output.
    """
    args = trace.get("arguments") or {}
    argv = args.get("argv")
    command = str(args.get("command") or "")
    argv_str = " ".join(str(item) for item in argv) if isinstance(argv, list) else str(argv or command)
    stdout = str(trace.get("stdout") or "")
    if "nmap" in argv_str.lower() or "Nmap scan report for" in stdout:
        nmap_trace = dict(trace)
        nmap_trace["tool_name"] = "nmap_scan"
        return [
            fact.model_copy(update={"source_tool": "run_command"})
            for fact in _extract_nmap_scan(nmap_trace)
        ]

    facts: list[ExtractedFact] = []
    seen_hosts: set[str] = set()
    for line in stdout.splitlines():
        match = re.search(r"\b(\d{1,3}(?:\.\d{1,3}){3}):(\d{1,5})\b", line)
        if not match:
            continue
        host = match.group(1)
        port = int(match.group(2))
        if host not in seen_hosts:
            seen_hosts.add(host)
            facts.append(
                ExtractedFact(
                    fact_type="Host",
                    entity_type="Host",
                    label=f"host:{host}",
                    properties={"address": host},
                    confidence=0.75,
                    source_tool="run_command",
                )
            )
        facts.append(
            ExtractedFact(
                fact_type="Service",
                entity_type="Service",
                label=f"tcp:{port}@{host}",
                properties={"address": host, "port": port, "protocol": "tcp"},
                confidence=0.75,
                source_tool="run_command",
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
        # Carry the host/port so the Service node is locatable for zone (CIDR)
        # resolution; a url-only Service has no address and can never be placed
        # in the entry zone, leaving entry_zone_service_discovered unsatisfiable.
        host = str(args.get("host") or args.get("address") or "")
        port = args.get("port")
        if not host:
            parsed_url = urlparse(url if "://" in url else f"//{url}")
            host = parsed_url.hostname or ""
            port = port or parsed_url.port
        facts.append(
            ExtractedFact(
                fact_type="Service",
                entity_type="Service",
                label=f"http-service:{url}",
                properties={
                    "url": url,
                    "address": host or None,
                    "host": host or None,
                    "port": port,
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


def _extract_post_access(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract a generic post-access fact from post-access tools.

    The semantic flavor (a captured proof artifact, a discovered hint, or a
    plain observation) is carried as a ``kind`` property, never as a
    lab-specific node type. The tool may declare ``fact_kind`` directly; absent
    that, the flavor is inferred from the tool name for compatibility.
    """
    facts: list[ExtractedFact] = []
    parsed = trace.get("parsed_output") or {}
    args = trace.get("arguments") or {}
    target = str(args.get("target") or "")
    zone_ref = str(args.get("zone_ref") or "")
    tool_name = str(trace.get("tool_name") or "")

    declared_kind = str(parsed.get("fact_kind") or "")
    if declared_kind:
        kind = declared_kind
    elif "lab_marker" in tool_name or "read_lab" in tool_name:
        kind = "proof"
    elif "hint" in tool_name:
        kind = "hint"
    else:
        kind = "post_access"

    # Every post-access flavor is generic Evidence discriminated by `kind`
    # (post_access / proof / hint); the specialised PostAccessObservation node
    # type was collapsed into Evidence{kind}.
    content_summary = str(parsed.get("content_summary") or parsed.get("summary") or "")
    facts.append(
        ExtractedFact(
            fact_type="Evidence",
            entity_type="Evidence",
            label=f"{kind}:{target}",
            properties={
                "kind": kind,
                "evidence_kind": kind,
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


def _extract_msf(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract facts from a metasploit_exec call.

    Running the exploit module is itself an exploit attempt (recorded as
    Evidence{kind:exploit_attempt} for contract #6, success or not); a successful
    run additionally opens a real Session (via _extract_session, satisfying
    #7/#8). cg.md G.6 phase 2: kind is the cross-cutting ToolFact discriminator.
    """
    facts: list[ExtractedFact] = list(_extract_session(trace))
    args = trace.get("arguments") or {}
    target = str(args.get("target") or "")
    module = str(args.get("module") or "")
    facts.append(
        ExtractedFact(
            fact_type="Evidence",
            entity_type="Evidence",
            label=f"exploit-attempt:{module}@{target}",
            properties={
                "kind": "exploit_attempt",
                "evidence_kind": "exploit_attempt",
                "module": module,
                "target_ref": target,
            },
            confidence=0.9,
            source_tool="metasploit_exec",
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


def _extract_goal_check(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract GoalProof + Evidence{kind:goal_check} from goal check tools."""
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
    # A goal-check attempt (pass or fail) is generic Evidence{kind:goal_check};
    # the specialised GoalCheck node type was collapsed into Evidence{kind}. The
    # passing proof is still the separate GoalProof node above.
    facts.append(
        ExtractedFact(
            fact_type="Evidence",
            entity_type="Evidence",
            label=f"goal-check:{goal_id}",
            properties={
                "kind": "goal_check",
                "evidence_kind": "goal_check",
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
                fact_type="Evidence",
                entity_type="Evidence",
                label=f"controlled-data-proof:{service}:{port}@{host}",
                properties={
                    "kind": "controlled_read",
                    "evidence_kind": "controlled_read",
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


def _internal_services_from_nmap_stdout(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Parse raw nmap stdout into restricted-zone Host/Service facts.

    Mirrors ``_extract_internal_discovery``'s node shape (zone_ref=restricted,
    internal=True) so ``service_discovered_via_route`` resolves identically
    whether the scan ran via the canned ``internal_service_discover`` or via the
    generic ``pivot_exec`` transport primitive.
    """
    facts: list[ExtractedFact] = []
    stdout = str(trace.get("stdout") or "")
    zone_ref = "restricted"
    current_host = ""
    for line in stdout.splitlines():
        hm = re.search(r"Nmap scan report for (?:[^()]*\()?(\d{1,3}(?:\.\d{1,3}){3})\)?", line)
        if hm:
            current_host = hm.group(1)
            facts.append(
                ExtractedFact(
                    fact_type="Host",
                    entity_type="Host",
                    label=f"internal-host:{current_host}",
                    properties={"address": current_host, "zone_ref": zone_ref, "internal": True},
                    confidence=0.85,
                    source_tool="pivot_exec",
                    zone_ref=zone_ref,
                )
            )
            continue
        m = re.match(r"(\d+)/(tcp|udp)\s+open\s+(\S+)", line)
        if m:
            port = int(m.group(1))
            proto = m.group(2)
            svc = m.group(3)
            facts.append(
                ExtractedFact(
                    fact_type="Service",
                    entity_type="Service",
                    label=f"internal-service:{svc}:{port}@{current_host}",
                    properties={
                        "port": port,
                        "protocol": proto,
                        "service_name": svc,
                        "address": current_host or None,
                        "zone_ref": zone_ref,
                        "internal": True,
                    },
                    confidence=0.85,
                    source_tool="pivot_exec",
                    zone_ref=zone_ref,
                )
            )
    return facts


def _extract_pivot_exec(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract facts from the generic ``pivot_exec`` transport primitive.

    A successful ``pivot_exec`` is causal proof the route is live (PivotRoute,
    via ``_extract_pivot_route``). When the argv ran an nmap scan, also parse the
    raw stdout into restricted-zone Service facts so internal service discovery
    has a real-tool source — the freeform replacement for the bespoke
    ``internal_service_discover`` / ``pivoted_nmap_scan`` (cg.md G.5 phase 1).
    """
    facts: list[ExtractedFact] = list(_extract_pivot_route(trace))
    args = trace.get("arguments") or {}
    argv = args.get("argv")
    argv_str = " ".join(str(a) for a in argv) if isinstance(argv, list) else str(argv or "")
    if "nmap" in argv_str.lower():
        facts.extend(_internal_services_from_nmap_stdout(trace))
    return facts


def _extract_nuclei_scan(trace: dict[str, Any]) -> list[ExtractedFact]:
    """Extract vulnerability-candidate Evidence from nuclei template hits.

    A nuclei match is a real vulnerability signal; record it as generic
    ``Evidence{kind:vuln_candidate}`` — the collapse target for the deprecated
    ``VulnerabilityCandidate`` node type (cg.md G.3). The contract migration
    (G.6 phase 2) will switch vuln_candidate conditions to read this kind.
    """
    facts: list[ExtractedFact] = []
    parsed = trace.get("parsed_output") or {}
    args = trace.get("arguments") or {}
    url = str(args.get("url") or args.get("target") or "")
    zone_ref = str(args.get("zone_ref") or "")
    hits = parsed.get("findings") or parsed.get("results") or parsed.get("matches") or []
    for hit in hits if isinstance(hits, list) else []:
        if not isinstance(hit, dict):
            continue
        info = hit.get("info") if isinstance(hit.get("info"), dict) else {}
        template = str(hit.get("template_id") or hit.get("template") or hit.get("id") or hit.get("name") or "nuclei-hit")
        severity = str(hit.get("severity") or info.get("severity") or "")
        facts.append(
            ExtractedFact(
                fact_type="Evidence",
                entity_type="Evidence",
                label=f"vuln-candidate:{template}@{url}",
                properties={
                    "kind": "vuln_candidate",
                    "evidence_kind": "vuln_candidate",
                    "template_id": template,
                    "severity": severity,
                    "url": url,
                    "target_ref": url,
                    "zone_ref": zone_ref,
                },
                confidence=0.8,
                source_tool="nuclei_scan",
                zone_ref=zone_ref,
            )
        )
    if not facts and url:
        # A completed scan with no hit is still fingerprint-grade Evidence.
        facts.append(
            ExtractedFact(
                fact_type="Evidence",
                entity_type="Evidence",
                label=f"nuclei-scan:{url}",
                properties={"kind": "vuln_scan", "evidence_kind": "vuln_scan", "url": url, "zone_ref": zone_ref},
                confidence=0.6,
                source_tool="nuclei_scan",
                zone_ref=zone_ref,
            )
        )
    return facts


# ---------------------------------------------------------------------------
# Tool name → extractor mapping
# ---------------------------------------------------------------------------

_TOOL_EXTRACTORS: dict[str, Any] = {
    "run_command": _extract_run_command,
    "nmap_scan": _extract_nmap_scan,
    "http_probe": _extract_http_probe,
    "web_fingerprint": _extract_web_fingerprint,
    "web_discover": _extract_web_fingerprint,
    "whatweb_fingerprint": _extract_web_fingerprint,
    "fingerprint_extract": _extract_web_fingerprint,
    "nuclei_scan": _extract_nuclei_scan,
    "post_access_observe": _extract_post_access,
    "read_lab_marker": _extract_post_access,
    "list_lab_hints": _extract_post_access,
    # Step 5 / G.5 phase 1a: a real metasploit_exec opens a real session (Session,
    # #7/#8) and records the attempt (Evidence{kind:exploit_attempt}, #6).
    "metasploit_exec": _extract_msf,
    # Step 5 generic transport primitive: a successful pivot_exec is causal proof
    # the route is live (PivotRoute); when its argv ran nmap it also yields
    # restricted-zone Service facts (the real-tool source for #12).
    "pivot_exec": _extract_pivot_exec,
    "goal_check": _extract_goal_check,
    "internal_goal_check": _extract_goal_check,
    "chain_goal_check": _extract_goal_check,
    "controlled_data_read_proof": _extract_controlled_data_read,
}


class ToolTraceFactExtractor:
    """Extract KG-writable facts from successful ToolTraces.

    Called by ResultApplier after every execution cycle, even when LLM
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
