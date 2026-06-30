from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.execution.configured_mcp_client import ConfiguredMCPClient
from src.integrations.mcp_lab.http_server import MCPHTTPHandler
from src.integrations.mcp_lab.server import handle_request
from src.integrations.mcp_lab import tools as mcp_tools
from src.integrations.mcp_lab.tools import LAB_TOOL_SPECS, _parse_nmap_output, call_lab_tool, load_hidden_fixture_from_env


def test_lab_tool_specs_include_v1_tools() -> None:
    names = {tool["name"] for tool in LAB_TOOL_SPECS}

    assert names >= {
        "run_command",
        "nmap_scan",
        "http_probe",
        "web_fingerprint",
        "web_discover",
        "dns_lookup",
        "tls_probe",
        "tcp_connect_probe",
        "http_basic_auth_check",
        "goal_check",
        "artifact_store",
        "nuclei_scan",
        "whatweb_fingerprint",
        "ffuf_discover",
        "chain_goal_check",
        "controlled_data_read_proof",
        "pivot_exec",
        "metasploit_exec",
        "session_exec",
    }


def test_goal_check_specs_expose_hidden_fixture_marker_id() -> None:
    specs = {tool["name"]: tool for tool in LAB_TOOL_SPECS}

    for name in ("goal_check", "chain_goal_check"):
        properties = specs[name]["inputSchema"]["properties"]
        assert properties["fixture_marker_id"] == {"type": "string"}


def test_parse_nmap_output_uses_report_host_for_services() -> None:
    parsed = _parse_nmap_output(
        "10.0.0.0/24",
        "\n".join(
            [
                "Nmap scan report for 10.0.0.5",
                "Host is up.",
                "80/tcp open  http    Apache httpd 2.4.57",
                "Nmap scan report for app.local (10.0.0.6)",
                "8080/tcp open  http    Jetty 9.2.11",
            ]
        ),
    )

    services = [item for item in parsed["entities"] if item["type"] == "Service"]

    assert {item["host"] for item in services} == {"10.0.0.5", "10.0.0.6"}
    assert "service::10.0.0.5:80/tcp" in {item["id"] for item in services}
    assert "service::10.0.0.6:8080/tcp" in {item["id"] for item in services}


def test_lab_tools_require_lab_mode(monkeypatch) -> None:
    monkeypatch.delenv("AEGRA_LAB_MODE", raising=False)

    payload = call_lab_tool("run_command", {"command": "echo blocked"})

    assert payload["success"] is False
    assert payload["exit_code"] == "lab_mode_required"
    assert payload["parsed"]["runtime_hints"]["required_env"] == "AEGRA_LAB_MODE=1"


def test_lab_run_command_returns_unified_payload(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")

    payload = call_lab_tool(
        "run_command",
        {
            "argv": [sys.executable, "-c", "print('lab-ok')"],
            "operation_id": "op-run-command",
            "trace_id": "trace-1",
        },
    )

    assert payload["success"] is True
    assert payload["stdout"].strip() == "lab-ok"
    assert payload["exit_code"] == 0
    assert payload["parsed"]["writeback_hints"]["observation_category"] == "command_execution"
    assert payload["metadata"]["raw_output_ref"].replace("\\", "/").endswith(
        "var/runtime/op-run-command/tool-outputs/trace-1.json"
    )
    assert Path(payload["metadata"]["raw_output_ref"]).exists()
    assert payload["artifacts"] == []


def test_nmap_scan_passes_operation_context_to_run_command(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    captured: dict[str, Any] = {}

    def fake_run_command(arguments: dict[str, Any]) -> dict[str, Any]:
        captured.update(arguments)
        return {
            "success": True,
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "parsed": {},
            "metadata": {"raw_output_ref": "raw.json"},
        }

    monkeypatch.setattr(mcp_tools, "_run_command", fake_run_command)

    payload = call_lab_tool(
        "nmap_scan",
        {"target": "127.0.0.1", "operation_id": "op-nmap", "trace_id": "trace-nmap"},
    )

    assert payload["success"] is True
    assert captured["operation_id"] == "op-nmap"
    assert captured["trace_id"] == "trace-nmap"


def test_nmap_scan_accepts_target_list_and_marks_no_targets_as_failure(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    captured: dict[str, Any] = {}

    def fake_run_command(arguments: dict[str, Any]) -> dict[str, Any]:
        captured.update(arguments)
        return {
            "success": True,
            "stdout": "WARNING: No targets were specified, so 0 hosts scanned.",
            "stderr": "Failed to resolve \"198.51.100.10,198.51.100.11\".",
            "exit_code": 0,
            "parsed": {},
            "metadata": {"raw_output_ref": "raw.json"},
        }

    monkeypatch.setattr(mcp_tools, "_run_command", fake_run_command)

    payload = call_lab_tool(
        "nmap_scan",
        {"target": ["198.51.100.10", "198.51.100.11"], "operation_id": "op-nmap", "trace_id": "trace-nmap"},
    )

    assert captured["argv"][-2:] == ["198.51.100.10", "198.51.100.11"]
    assert payload["success"] is False
    assert payload["exit_code"] == "no_targets_scanned"
    assert payload["parsed"]["runtime_hints"]["blocked_by"] == "nmap_no_targets_scanned"


def test_hidden_fixture_loads_only_inside_mcp_tool_env(tmp_path, monkeypatch) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps({"markers": [{"id": "goal-marker", "literal": "secret-marker-value"}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AEGRA_LAB_FIXTURE_PATH", str(fixture_path))

    fixture = load_hidden_fixture_from_env()

    assert fixture["markers"][0]["id"] == "goal-marker"


def test_goal_check_hidden_marker_does_not_leak_literal(tmp_path, monkeypatch) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps({"markers": [{"id": "goal-marker", "literal": "secret-marker-value"}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    monkeypatch.setenv("AEGRA_LAB_FIXTURE_PATH", str(fixture_path))
    monkeypatch.setattr(
        mcp_tools,
        "_open_url",
        lambda *_, **__: {
            "url": "http://internal.example/",
            "status": 200,
            "headers": {},
            "body_excerpt": "prefix secret-marker-value suffix",
        },
    )

    payload = call_lab_tool("goal_check", {"url": "http://internal.example/", "fixture_marker_id": "goal-marker"})
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["success"] is True
    assert payload["parsed"]["findings"][0]["checks"][-1]["matched"] is True
    assert "secret-marker-value" not in serialized


def test_non_command_lab_tool_writes_raw_output_ref(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    monkeypatch.setattr(
        mcp_tools,
        "_open_url",
        lambda *_, **__: {
            "status": 200,
            "headers": {"content-type": "text/html"},
            "body_excerpt": "<html><title>Fingerprint</title></html>",
        },
    )

    payload = call_lab_tool(
        "web_fingerprint",
        {"url": "http://127.0.0.1:8080/", "operation_id": "op-web-fp", "trace_id": "trace-web-fp"},
    )

    assert payload["success"] is True
    assert payload["metadata"]["raw_output_ref"].replace("\\", "/").endswith(
        "var/runtime/op-web-fp/tool-outputs/trace-web-fp.json"
    )
    assert payload["parsed"]["writeback_hints"]["raw_output_ref"] == payload["metadata"]["raw_output_ref"]


def test_mcp_lab_server_handles_tools_list_and_call(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")

    listed = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    called = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "run_command", "arguments": {"argv": [sys.executable, "-c", "print('server-ok')"]}},
        }
    )

    assert listed is not None
    assert listed["result"]["tools"][0]["name"] == "run_command"
    assert called is not None
    assert called["result"]["structuredContent"]["success"] is True
    assert called["result"]["structuredContent"]["stdout"].strip() == "server-ok"


def test_configured_mcp_client_stdio_lists_and_calls_lab_server() -> None:
    client = ConfiguredMCPClient(
        {
            "servers": {
                "pentest-tools": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": ["-m", "src.integrations.mcp_lab.server"],
                    "cwd": ".",
                    "env": {"AEGRA_LAB_MODE": "1"},
                }
            }
        }
    )
    try:
        catalog = client.list_tools()
        result = client.call_tool(
            server_id="pentest-tools",
            tool_name="run_command",
            arguments={"argv": [sys.executable, "-c", "print('client-ok')"]},
            timeout_seconds=10,
        )
    finally:
        client.close()

    assert _tool_names(catalog) >= {"run_command", "http_probe", "tls_probe"}
    assert result.success is True
    assert result.stdout.strip() == "client-ok"
    assert result.exit_code == 0
    assert result.content["parsed"]["writeback_hints"]["observation_category"] == "command_execution"


def test_configured_mcp_client_trusts_structured_success_for_http_status_exit_code() -> None:
    client = ConfiguredMCPClient({"servers": {}})
    result = client._tool_result_from_rpc(  # noqa: SLF001
        server_id="pentest-tools",
        raw={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "isError": False,
                "structuredContent": {
                    "success": True,
                    "exit_code": 200,
                    "stdout": "ok",
                    "stderr": "",
                },
            },
        },
    )

    assert result.success is True
    assert result.exit_code == 200


def test_configured_mcp_client_http_lists_lab_tools_without_proxy(monkeypatch) -> None:
    from http.server import ThreadingHTTPServer

    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    server = ThreadingHTTPServer(("127.0.0.1", 0), MCPHTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = ConfiguredMCPClient(
        {
            "servers": {
                "external-tools": {
                    "transport": "http",
                    "url": f"http://{host}:{port}/mcp",
                }
            }
        }
    )
    try:
        catalog = client.list_tools()
    finally:
        client.close()
        server.shutdown()
        server.server_close()

    assert _tool_names_for(catalog, "external-tools") >= {"run_command", "http_probe", "tls_probe"}


def test_configured_mcp_client_stdio_ignores_startup_logs() -> None:
    script = (
        "import json, sys\n"
        "print('[*] startup log before json-rpc', flush=True)\n"
        "for line in sys.stdin:\n"
        "    req=json.loads(line)\n"
        "    method=req.get('method')\n"
        "    if method == 'initialize':\n"
        "        print(json.dumps({'jsonrpc':'2.0','id':req.get('id'),'result':{'protocolVersion':'2024-11-05','capabilities':{'tools':{}},'serverInfo':{'name':'fixture','version':'1'}}}), flush=True)\n"
        "    elif method == 'tools/list':\n"
        "        print('loaded tools', flush=True)\n"
        "        print(json.dumps({'jsonrpc':'2.0','id':req.get('id'),'result':{'tools':[{'name':'fixture_tool','inputSchema':{'type':'object'}}]}}), flush=True)\n"
        "    elif method.startswith('notifications/'):\n"
        "        pass\n"
    )
    client = ConfiguredMCPClient(
        {
            "servers": {
                "loggy-tools": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": ["-c", script],
                }
            }
        }
    )
    try:
        catalog = client.list_tools()
    finally:
        client.close()

    assert _tool_names_for(catalog, "loggy-tools") == {"fixture_tool"}


def test_lab_artifact_store_returns_artifact(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    monkeypatch.setenv("AEGRA_MCP_ARTIFACT_DIR", str(tmp_path))

    payload = call_lab_tool("artifact_store", {"name": "sample evidence", "json_content": {"ok": True}})

    assert payload["success"] is True
    assert payload["artifacts"][0]["content_type"] == "application/json"
    assert payload["parsed"]["writeback_hints"]["observation_category"] == "evidence_artifact"


def test_lab_tcp_connect_probe_reports_unreachable(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")

    payload = call_lab_tool("tcp_connect_probe", {"host": "127.0.0.1", "port": 1, "timeout_seconds": 1})

    assert payload["success"] is False
    assert payload["exit_code"] == "unreachable"
    assert payload["parsed"]["runtime_hints"]["reachable"] is False


def test_controlled_data_read_proof_returns_redacted_hash(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    monkeypatch.delenv("AEGRA_LAB_DB_USER", raising=False)
    monkeypatch.delenv("AEGRA_LAB_DB_PASSWORD", raising=False)
    monkeypatch.delenv("AEGRA_LAB_DB_NAME", raising=False)
    monkeypatch.setattr(mcp_tools, "_pivot_transport_available", lambda arguments=None: True)
    monkeypatch.setattr(
        mcp_tools,
        "_run_psql_query_via_configured_pivot",
        lambda **kwargs: {"success": True, "stdout": "labdb|lab|PostgreSQL 16\n", "stderr": ""},
    )

    payload = call_lab_tool(
        "controlled_data_read_proof",
        {
            "host": "10.0.2.50",
            "port": 5432,
            "route_id": "route-1",
            "database": "labdb",
            "username": "lab",
            "password": "labpass",
            "pivot_host": "10.0.1.50",
            "pivot_username": "pivot",
            "pivot_password": "pivotpass",
            "timeout_seconds": 1,
        },
    )

    assert payload["success"] is True
    assert "labpass" not in json.dumps(payload)
    assert "labdb|lab|PostgreSQL" not in json.dumps(payload)
    evidence = payload["parsed"]["evidence"][0]
    assert evidence["kind"] == "controlled_data_read_proof"
    assert evidence["row_count"] == 1
    assert payload["parsed"]["runtime_hints"]["controlled_data_read"] is True


def test_optional_external_tool_returns_structured_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    monkeypatch.setenv("PATH", "")

    payload = call_lab_tool("nuclei_scan", {"url": "http://127.0.0.1"})

    assert payload["success"] is False
    assert payload["exit_code"] == "tool_unavailable"
    assert payload["parsed"]["runtime_hints"]["missing_binary"] == "nuclei"


def test_pivot_exec_runs_argv_through_configured_route(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    captured: dict[str, Any] = {}

    def fake_pivot(*, route_id, argv, timeout, env=None, username=None, password=None):
        captured["route_id"] = route_id
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, returncode=0, stdout="uid=0(root)\n", stderr="")

    monkeypatch.setattr(mcp_tools, "_run_via_configured_pivot", fake_pivot)

    payload = call_lab_tool(
        "pivot_exec",
        {"route_id": "route-9", "argv": ["id"], "timeout_seconds": 5},
    )

    assert payload["success"] is True
    assert payload["stdout"] == "uid=0(root)\n"
    assert captured == {"route_id": "route-9", "argv": ["id"]}
    # parsed.route_id lets the fact extractor record this as an active PivotRoute.
    assert payload["parsed"]["route_id"] == "route-9"
    assert payload["parsed"]["runtime_hints"]["pivot_exec"] is True


def test_pivot_exec_accepts_shell_string_argv(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    captured: dict[str, Any] = {}

    def fake_pivot(*, route_id, argv, timeout, env=None, username=None, password=None):
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(mcp_tools, "_run_via_configured_pivot", fake_pivot)

    payload = call_lab_tool("pivot_exec", {"route_id": "r1", "argv": "cat /etc/passwd"})

    assert captured["argv"] == ["cat", "/etc/passwd"]
    assert payload["success"] is False
    assert payload["exit_code"] == "pivot_exec_failed"


class _FakeMsfSessions:
    def __init__(self) -> None:
        self._d: dict[str, Any] = {}

    @property
    def list(self) -> dict[str, Any]:
        return dict(self._d)

    def session(self, sid: str) -> "_FakeMsfShell":
        return _FakeMsfShell()


class _FakeMsfShell:
    def write(self, data: str) -> None:
        self._last = data

    def read(self) -> str:
        return "uid=0(root)\n"


class _FakeMsfModule:
    def __init__(self, sessions: _FakeMsfSessions) -> None:
        self._sessions = sessions
        self.opts: dict[str, Any] = {}

    def __setitem__(self, key: str, value: Any) -> None:
        self.opts[key] = value

    def execute(self, payload: Any = None) -> dict[str, Any]:
        # Executing the exploit opens a real session in the live engine; the fake
        # mirrors that by registering one keyed "1".
        self._sessions._d["1"] = {
            "type": "shell",
            "target_host": self.opts.get("RHOSTS"),
            "via_exploit": "struts2",
        }
        return {"job_id": 0, "uuid": "u"}


class _FakeMsfModules:
    def __init__(self, sessions: _FakeMsfSessions) -> None:
        self._sessions = sessions

    def use(self, mtype: str, mpath: str) -> _FakeMsfModule:
        return _FakeMsfModule(self._sessions)


class _FakeMsfClient:
    def __init__(self) -> None:
        self.sessions = _FakeMsfSessions()
        self.modules = _FakeMsfModules(self.sessions)


def test_metasploit_exec_opens_session_via_rpc(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    monkeypatch.setattr(
        mcp_tools,
        "_load_msf_config",
        lambda: {"host": "10.20.0.60", "port": 55553, "password": "x", "lhost": "10.20.0.60"},
    )
    monkeypatch.setattr(mcp_tools, "_msf_client", lambda config: _FakeMsfClient())

    payload = call_lab_tool(
        "metasploit_exec",
        {"module": "exploit/multi/http/struts2_content_type_ognl", "target": "10.20.0.10", "rport": 8080},
    )

    assert payload["success"] is True
    # session_id flows to parsed -> ToolTraceFactExtractor._extract_session -> KG Session.
    assert payload["parsed"]["session_id"] == "1"
    assert payload["parsed"]["runtime_hints"]["bound_target"] == "10.20.0.10"


def test_metasploit_exec_without_config_is_blocked(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    monkeypatch.setattr(mcp_tools, "_load_msf_config", lambda: None)

    payload = call_lab_tool("metasploit_exec", {"module": "x", "target": "10.20.0.10"})

    assert payload["success"] is False
    assert payload["exit_code"] == "msf_not_configured"


def test_metasploit_exec_no_session_returns_structured_attempt(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    monkeypatch.setattr(
        mcp_tools,
        "_load_msf_config",
        lambda: {"host": "10.20.0.60", "port": 55553, "password": "x", "lhost": "10.20.0.60"},
    )
    monkeypatch.setattr(mcp_tools, "_msf_client", lambda config: _FakeMsfClient())
    monkeypatch.setattr(mcp_tools, "_msf_wait_for_session", lambda client, before, timeout: None)

    payload = call_lab_tool(
        "metasploit_exec",
        {
            "module": "exploit/multi/http/struts2_content_type_ognl",
            "target": "10.20.0.10",
            "rport": 8080,
            "timeout_seconds": 10,
        },
    )

    assert payload["success"] is True
    assert payload["exit_code"] == "no_session"
    assert payload["parsed"]["runtime_hints"]["exploit_executed"] is True
    assert payload["parsed"]["runtime_hints"]["session_opened"] is False


def _tool_names(catalog: dict[str, Any]) -> set[str]:
    return _tool_names_for(catalog, "pentest-tools")


def _tool_names_for(catalog: dict[str, Any], server_id: str) -> set[str]:
    tools = catalog[server_id]["tools"]
    return {tool["name"] for tool in tools}
