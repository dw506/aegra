from __future__ import annotations

from src.integrations.mcp_lab.catalog import build_default_lab_tool_catalog


def test_default_open_lab_tool_catalog_includes_all_core_lab_tools() -> None:
    catalog = build_default_lab_tool_catalog()
    names = {tool["name"] for tool in catalog["pentest-tools"]["tools"]}

    assert {
        "run_command",
        "nmap_scan",
        "http_probe",
        "safe_vuln_validate",
        "pivot_route_probe",
        "internal_service_discover",
        "goal_check",
    } <= names
