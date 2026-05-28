from __future__ import annotations

import sys
import threading
from typing import Any

from src.core.execution.configured_mcp_client import ConfiguredMCPClient
from src.integrations.mcp_lab.http_server import MCPHTTPHandler
from src.integrations.mcp_lab.server import handle_request
from src.integrations.mcp_lab.tools import LAB_TOOL_SPECS, call_lab_tool


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
        "vuln_profile_match",
        "validation_precheck",
        "safe_vuln_validate",
        "credential_check",
        "session_probe",
        "session_open_lab",
        "identity_context_probe",
        "privilege_context_probe",
        "pivot_route_probe",
        "internal_service_discover",
        "chain_goal_check",
    }


def test_lab_tools_require_lab_mode(monkeypatch) -> None:
    monkeypatch.delenv("AEGRA_LAB_MODE", raising=False)

    payload = call_lab_tool("run_command", {"command": "echo blocked"})

    assert payload["success"] is False
    assert payload["exit_code"] == "lab_mode_required"
    assert payload["parsed"]["runtime_hints"]["required_env"] == "AEGRA_LAB_MODE=1"


def test_lab_run_command_returns_unified_payload(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")

    payload = call_lab_tool("run_command", {"argv": [sys.executable, "-c", "print('lab-ok')"]})

    assert payload["success"] is True
    assert payload["stdout"].strip() == "lab-ok"
    assert payload["exit_code"] == 0
    assert payload["parsed"]["writeback_hints"]["observation_category"] == "command_execution"
    assert payload["artifacts"] == []


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


def test_lab_safe_vuln_validate_rejects_unsafe_mode(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")

    payload = call_lab_tool(
        "safe_vuln_validate",
        {"target_url": "http://127.0.0.1", "profile_id": "lab-http-accessible", "safe_mode": False},
    )

    assert payload["success"] is False
    assert payload["exit_code"] == "unsafe_mode_rejected"
    assert payload["parsed"]["runtime_hints"]["blocked_by"] == "unsafe_mode_rejected"


def test_lab_session_open_returns_runtime_hints(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")

    payload = call_lab_tool(
        "session_open_lab",
        {"session_id": "sess-1", "bound_target": "host-1", "bound_identity": "alice", "reuse_policy": "shared"},
    )

    assert payload["success"] is True
    assert payload["parsed"]["runtime_hints"]["open_session"] is True
    assert payload["parsed"]["runtime_hints"]["session_id"] == "sess-1"


def test_lab_pivot_route_probe_returns_route_hints(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")

    payload = call_lab_tool(
        "pivot_route_probe",
        {"route_id": "route-1", "destination_host": "127.0.0.1", "destination_port": 1, "timeout_seconds": 1},
    )

    assert payload["success"] is False
    assert payload["parsed"]["runtime_hints"]["register_pivot_route"] is True
    assert payload["parsed"]["runtime_hints"]["route_id"] == "route-1"


def test_optional_external_tool_returns_structured_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    monkeypatch.setenv("PATH", "")

    payload = call_lab_tool("nuclei_scan", {"url": "http://127.0.0.1"})

    assert payload["success"] is False
    assert payload["exit_code"] == "tool_unavailable"
    assert payload["parsed"]["runtime_hints"]["missing_binary"] == "nuclei"


def _tool_names(catalog: dict[str, Any]) -> set[str]:
    return _tool_names_for(catalog, "pentest-tools")


def _tool_names_for(catalog: dict[str, Any], server_id: str) -> set[str]:
    tools = catalog[server_id]["tools"]
    return {tool["name"] for tool in tools}
