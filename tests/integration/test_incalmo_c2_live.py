from __future__ import annotations

import os

import pytest

from src.core.execution.adapters.incalmo_c2_adapter import IncalmoC2Adapter
from src.core.execution.tool_plan import ToolPlan
from src.integrations.incalmo.client import IncalmoClient


def test_live_incalmo_c2_adapter_returns_tool_execution_result() -> None:
    if os.getenv("AEGRA_RUN_LIVE_INCALMO_TEST") != "1":
        pytest.skip("set AEGRA_RUN_LIVE_INCALMO_TEST=1 to run live Incalmo C2 test")

    c2_url = os.getenv("INCALMO_C2_URL", "http://127.0.0.1:8888")
    client = IncalmoClient(c2_url, timeout_seconds=10)
    agent_id = os.getenv("INCALMO_C2_AGENT_ID") or _first_agent_id(client)
    if not agent_id:
        pytest.skip("no Incalmo agent available; set INCALMO_C2_AGENT_ID")

    command = os.getenv("INCALMO_C2_COMMAND", "echo aegra-live-smoke")
    result = IncalmoC2Adapter(client).execute(
        ToolPlan(
            task_id="live-incalmo-smoke",
            tool="live_command",
            adapter="incalmo_c2",
            command=command,
            target_agent_ref=agent_id,
            timeout_seconds=int(os.getenv("INCALMO_C2_TIMEOUT_ATTEMPTS", "10")),
        )
    )

    assert result.adapter == "incalmo_c2"
    assert result.tool == "live_command"
    assert result.command_id
    assert result.payload_ref == f"incalmo://commands/{result.command_id}" or result.payload_ref
    assert result.success is True


def _first_agent_id(client: IncalmoClient) -> str | None:
    agents = client.get_agents()
    if not agents:
        return None
    first = agents[0]
    return str(getattr(first, "agent_id", None) or getattr(first, "id", None) or "")
