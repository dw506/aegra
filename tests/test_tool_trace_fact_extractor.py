from __future__ import annotations

from src.core.runtime.tool_trace_fact_extractor import ToolTraceFactExtractor


def _extract(trace: dict) -> list:
    return ToolTraceFactExtractor().extract(trace).facts


def test_pivot_exec_nmap_argv_yields_pivot_route_and_internal_service() -> None:
    # cg.md G.5 phase 1 (#12): a pivot_exec that ran nmap is the real-tool source
    # for restricted-zone service discovery — it must record BOTH the live route
    # and the internal Service (zone_ref=restricted) so service_discovered_via_route
    # resolves without the canned internal_service_discover.
    facts = _extract(
        {
            "trace_id": "t-pivot-nmap",
            "tool_name": "pivot_exec",
            "success": True,
            "arguments": {
                "route_id": "pivot-entry-to-restricted",
                "argv": ["nmap", "-Pn", "-p-", "10.30.0.50"],
            },
            "stdout": "Nmap scan report for 10.30.0.50\n5432/tcp open postgresql\n",
            "parsed_output": {"route_id": "pivot-entry-to-restricted"},
        }
    )

    assert any(f.fact_type == "PivotRoute" for f in facts)
    service = next(f for f in facts if f.fact_type == "Service")
    assert service.properties["zone_ref"] == "restricted"
    assert service.properties["internal"] is True
    assert service.properties["port"] == 5432
    assert service.properties["service_name"] == "postgresql"
    assert service.properties["address"] == "10.30.0.50"


def test_pivot_exec_non_nmap_argv_yields_only_pivot_route() -> None:
    # A plain command over the pivot proves the route is live but must NOT
    # invent service facts from arbitrary stdout.
    facts = _extract(
        {
            "trace_id": "t-pivot-cat",
            "tool_name": "pivot_exec",
            "success": True,
            "arguments": {"route_id": "pivot-entry-to-restricted", "argv": ["cat", "/opt/loot/db.env"]},
            "stdout": "DB_HOST=10.30.0.50\nDB_USER=app\n",
            "parsed_output": {"route_id": "pivot-entry-to-restricted"},
        }
    )

    assert {f.fact_type for f in facts} == {"PivotRoute"}


def test_nuclei_scan_hit_yields_vuln_candidate_evidence() -> None:
    # cg.md G.3: nuclei hits collapse into Evidence{kind:vuln_candidate}, the
    # real-tool replacement for the canned VulnerabilityCandidate node type.
    facts = _extract(
        {
            "trace_id": "t-nuclei",
            "tool_name": "nuclei_scan",
            "success": True,
            "arguments": {"url": "http://10.20.0.10:8080/"},
            "parsed_output": {
                "findings": [{"template_id": "apache-struts-rce", "severity": "critical"}]
            },
        }
    )

    evidence = next(f for f in facts if f.fact_type == "Evidence")
    assert evidence.properties["kind"] == "vuln_candidate"
    assert evidence.properties["template_id"] == "apache-struts-rce"
    assert evidence.properties["severity"] == "critical"


def test_nuclei_scan_no_hit_still_yields_scan_evidence() -> None:
    facts = _extract(
        {
            "trace_id": "t-nuclei-empty",
            "tool_name": "nuclei_scan",
            "success": True,
            "arguments": {"url": "http://10.20.0.10:8080/"},
            "parsed_output": {"findings": []},
        }
    )

    evidence = next(f for f in facts if f.fact_type == "Evidence")
    assert evidence.properties["kind"] == "vuln_scan"


def test_metasploit_exec_session_yields_session_node() -> None:
    # cg.md G.5 phase 1a: a real metasploit_exec opens a session whose id flows
    # to a KG Session node (real-tool source for contract #7/#8).
    facts = _extract(
        {
            "trace_id": "t-msf",
            "tool_name": "metasploit_exec",
            "success": True,
            "arguments": {"module": "exploit/multi/http/struts2_content_type_ognl", "target": "10.20.0.10"},
            "parsed_output": {"session_id": "1"},
        }
    )

    session = next(f for f in facts if f.fact_type == "Session")
    assert session.properties["session_id"] == "1"
    assert session.properties["bound_target"] == "10.20.0.10"
    assert session.properties["status"] == "active"
    attempt = next(f for f in facts if f.fact_type == "Evidence")
    assert attempt.properties["kind"] == "exploit_attempt"


def test_metasploit_exec_no_session_still_yields_attempt_evidence() -> None:
    facts = _extract(
        {
            "trace_id": "t-msf-nosession",
            "tool_name": "metasploit_exec",
            "success": True,
            "arguments": {"module": "exploit/multi/http/struts2_content_type_ognl", "target": "10.20.0.10"},
            "parsed_output": {
                "runtime_hints": {
                    "exploit_executed": True,
                    "session_opened": False,
                    "module": "exploit/multi/http/struts2_content_type_ognl",
                }
            },
        }
    )

    assert {f.fact_type for f in facts} == {"Evidence"}
    evidence = facts[0]
    assert evidence.properties["kind"] == "exploit_attempt"
    assert evidence.properties["module"] == "exploit/multi/http/struts2_content_type_ognl"


def test_run_command_nmap_stdout_yields_entry_service() -> None:
    facts = _extract(
        {
            "trace_id": "t-run-nmap",
            "tool_name": "run_command",
            "success": True,
            "arguments": {"argv": ["bash", "-lc", "nmap -Pn -n --top-ports 20 10.20.0.4-254"]},
            "stdout": "Nmap scan report for 10.20.0.10\n8080/tcp open http-proxy\n",
            "parsed_output": {},
        }
    )

    service = next(f for f in facts if f.fact_type == "Service")
    assert service.source_tool == "run_command"
    assert service.properties["address"] == "10.20.0.10"
    assert service.properties["port"] == 8080


def test_run_command_host_port_sweep_yields_service() -> None:
    facts = _extract(
        {
            "trace_id": "t-run-sweep",
            "tool_name": "run_command",
            "success": True,
            "arguments": {"argv": ["sh", "-lc", "tcp sweep"]},
            "stdout": "10.20.0.11:80\n10.20.0.10:8080\n",
            "parsed_output": {},
        }
    )

    services = [f for f in facts if f.fact_type == "Service"]
    assert {(s.properties["address"], s.properties["port"]) for s in services} == {
        ("10.20.0.11", 80),
        ("10.20.0.10", 8080),
    }


def test_whatweb_fingerprint_yields_fingerprint_evidence() -> None:
    facts = _extract(
        {
            "trace_id": "t-whatweb",
            "tool_name": "whatweb_fingerprint",
            "success": True,
            "arguments": {"url": "http://10.20.0.10:8080/"},
            "parsed_output": {"technologies": [{"name": "Apache Struts"}]},
        }
    )

    evidence = next(f for f in facts if f.fact_type == "Evidence")
    assert evidence.properties["evidence_kind"] == "web_fingerprint"
    assert evidence.properties["technology"] == "Apache Struts"

