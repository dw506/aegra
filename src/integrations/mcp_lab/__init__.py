"""Experimental MCP lab server integration."""

from src.integrations.mcp_lab.tools import LAB_TOOL_SPECS, call_lab_tool, load_hidden_fixture_from_env

__all__ = ["LAB_TOOL_SPECS", "call_lab_tool", "load_hidden_fixture_from_env"]
