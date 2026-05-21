from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.core.execution.adapters.incalmo_c2_adapter import IncalmoC2Adapter
from src.core.execution.tool_plan import ToolPlan
from src.integrations.incalmo.client import IncalmoClient, IncalmoClientConfig
from src.integrations.incalmo.mapper import IncalmoMapper
from src.integrations.incalmo.models import Agent, CommandResult, CommandStatus


class _Handler(BaseHTTPRequestHandler):
    command_status = "succeeded"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/agents":
            self._json({"agents": [{"agent_id": "agent-1", "hostname": "host-1", "status": "online"}]})
            return
        if self.path == "/agents/agent-1":
            self._json({"agent_id": "agent-1", "hostname": "host-1", "status": "online"})
            return
        if self.path == "/agents/agent-1/commands/cmd-1":
            self._json(
                {
                    "command_id": "cmd-1",
                    "agent_id": "agent-1",
                    "status": self.command_status,
                    "stdout": "ok",
                    "stderr": "",
                    "exit_code": 0,
                }
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/agents/agent-1/commands":
            _ = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
            self._json({"command_id": "cmd-1", "agent_id": "agent-1", "command": "whoami", "status": "pending"})
            return
        if self.path == "/environment":
            self._json({"ok": True})
            return
        self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        return None

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_incalmo_client_sends_and_polls_success() -> None:
    server = _server()
    try:
        client = IncalmoClient(IncalmoClientConfig(c2_url=f"http://127.0.0.1:{server.server_port}", poll_interval_sec=0.01))

        agents = client.get_agents()
        command = client.send_command("agent-1", "whoami")
        result = client.wait_for_command("agent-1", command.command_id, timeout_sec=1)

        assert agents[0].agent_id == "agent-1"
        assert result.status == CommandStatus.SUCCEEDED
        assert result.stdout == "ok"
    finally:
        server.shutdown()


def test_incalmo_mapper_maps_agent_and_result() -> None:
    mapper = IncalmoMapper()
    session = mapper.agent_to_session(Agent(agent_id="agent-1", address="10.0.0.5", status="online"))
    result = mapper.command_result_to_task_result(
        CommandResult(command_id="cmd-1", agent_id="agent-1", status=CommandStatus.SUCCEEDED, stdout="ok"),
        operation_id="op-1",
        task_id="task-1",
        tg_node_id="task-1",
    )

    assert session.session_id == "incalmo::agent-1"
    assert result.status.value == "succeeded"
    assert result.evidence[0].kind == "incalmo_command_output"


def test_incalmo_adapter_returns_canonical_result() -> None:
    server = _server()
    try:
        client = IncalmoClient(IncalmoClientConfig(c2_url=f"http://127.0.0.1:{server.server_port}", poll_interval_sec=0.01))
        adapter = IncalmoC2Adapter(client)
        plan = ToolPlan(
            task_id="task-1",
            tool="whoami",
            adapter="incalmo_c2",
            command="whoami",
            payloads={"agent_id": "agent-1"},
            target_agent_ref="agent-1",
        )

        result = adapter.execute(plan)

        assert result.success is True
        assert result.stdout == "ok"
        assert result.command_id == "cmd-1"
        assert result.metadata["agent_id"] == "agent-1"
    finally:
        server.shutdown()
