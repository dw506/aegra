"""Default open lab MCP tool catalog."""

from __future__ import annotations

from typing import Any

from src.integrations.mcp_lab.tools import lab_tool_specs


def build_default_lab_tool_catalog(*, server_id: str = "pentest-tools", include_unavailable: bool = True) -> dict[str, Any]:
    """Return all lab MCP tools under one server catalog.

    The catalog is an existence list for StageAgents. Policy enforcement is
    audit-only for MCP execution in authorized lab mode.
    """

    return {
        server_id: {
            "available": True,
            "tools": lab_tool_specs(include_unavailable=include_unavailable),
        }
    }


__all__ = ["build_default_lab_tool_catalog"]
