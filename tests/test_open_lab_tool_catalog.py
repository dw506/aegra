from __future__ import annotations

from src.integrations.mcp_lab.catalog import build_default_lab_tool_catalog


def test_default_open_lab_tool_catalog_includes_all_core_lab_tools() -> None:
    catalog = build_default_lab_tool_catalog()
    names = {tool["name"] for tool in catalog["pentest-tools"]["tools"]}

    assert {
        "run_command",
        "nmap_scan",
        "http_probe",
        "pivot_exec",
        "controlled_data_read_proof",
        "goal_check",
    } <= names
