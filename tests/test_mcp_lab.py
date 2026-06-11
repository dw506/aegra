from __future__ import annotations

import json
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
        "pivoted_nmap_scan",
        "chain_goal_check",
        "controlled_data_read_proof",
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


def test_internal_service_discover_uses_configured_pivot_for_restricted_data_service(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    monkeypatch.setenv("AEGRA_LAB_PIVOT_HOST", "pivot.example")
    monkeypatch.setenv("AEGRA_LAB_PIVOT_USER", "pivot")
    monkeypatch.setenv("AEGRA_LAB_PIVOT_PASSWORD", "hidden")
    monkeypatch.setattr(mcp_tools, "_tcp_connect_probe", lambda arguments: {"success": False, "parsed": mcp_tools._default_parsed()})
    monkeypatch.setattr(mcp_tools, "_probe_tcp_via_configured_pivot", lambda **kwargs: {"reachable": True, "stderr": ""})

    payload = call_lab_tool(
        "internal_service_discover",
        {"host": "10.0.2.50", "port": 5432, "route_id": "route-1", "timeout_seconds": 1},
    )

    assert payload["success"] is True
    assert payload["parsed"]["services"][0]["service_name"] == "postgres"
    satisfied = payload["parsed"]["runtime_hints"]["satisfied_conditions"]
    assert {"condition": "restricted_data_service_discovered", "evidence_ids": ["internal-service::route-1::10.0.2.50:5432"]} in satisfied


def test_identity_context_probe_exposes_redacted_pivot_candidates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "adapter_policy": {
                    "pivot": {
                        "default_route": {
                            "route_id": "route-1",
                            "source_host": "10.0.1.10",
                            "via_host": "10.0.1.30",
                            "destination_cidr": "10.0.2.0/24",
                            "protocol": "ssh",
                            "transport": {"adapter": "proxy_shell"},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AEGRA_RUNTIME_POLICY_PATH", str(policy_path))

    payload = call_lab_tool("identity_context_probe", {})

    candidates = payload["parsed"]["runtime_hints"]["pivot_route_candidates"]
    assert candidates == [
        {
            "route_id": "route-1",
            "source_host": "10.0.1.10",
            "via_host": "10.0.1.30",
            "destination_cidr": "10.0.2.0/24",
            "destination_host": None,
            "protocol": "ssh",
            "transport_adapter": "proxy_shell",
        }
    ]
    assert "password" not in json.dumps(payload).lower()


def test_pivoted_nmap_scan_emits_restricted_data_service_condition(monkeypatch) -> None:
    class Completed:
        returncode = 0
        stdout = "\n".join(
            [
                "Nmap scan report for 10.0.2.50",
                "Host is up.",
                "PORT     STATE SERVICE    VERSION",
                "5432/tcp open  postgresql PostgreSQL DB",
            ]
        )
        stderr = ""

    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    monkeypatch.setattr(mcp_tools, "_run_via_configured_pivot", lambda **kwargs: Completed())

    payload = call_lab_tool(
        "pivoted_nmap_scan",
        {"target": "10.0.2.0/24", "ports": "5432", "route_id": "route-1", "timeout_seconds": 1},
    )

    assert payload["success"] is True
    conditions = payload["parsed"]["runtime_hints"]["satisfied_conditions"]
    assert any(item["condition"] == "restricted_data_service_discovered" for item in conditions)


def test_controlled_data_read_proof_returns_redacted_hash(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")
    monkeypatch.setenv("AEGRA_LAB_DB_USER", "lab")
    monkeypatch.setenv("AEGRA_LAB_DB_PASSWORD", "labpass")
    monkeypatch.setenv("AEGRA_LAB_DB_NAME", "labdb")
    monkeypatch.setattr(mcp_tools, "_pivot_transport_available", lambda: True)
    monkeypatch.setattr(
        mcp_tools,
        "_run_psql_query_via_configured_pivot",
        lambda **kwargs: {"success": True, "stdout": "labdb|lab|PostgreSQL 16\n", "stderr": ""},
    )

    payload = call_lab_tool(
        "controlled_data_read_proof",
        {"host": "10.0.2.50", "port": 5432, "route_id": "route-1", "timeout_seconds": 1},
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


def test_vuln_profile_match_uses_title_path_framework_and_banner_signals(monkeypatch) -> None:
    monkeypatch.setenv("AEGRA_LAB_MODE", "1")

    struts = call_lab_tool(
        "vuln_profile_match",
        {
            "service": "http",
            "target_url": "http://10.20.0.10:8080/",
            "title": "Struts2 Showcase - Fileupload sample",
            "path": "/struts2-showcase/fileupload/doUpload.action",
            "banner": "Apache-Coyote",
        },
    )
    thinkphp = call_lab_tool(
        "vuln_profile_match",
        {
            "service": "http",
            "target_url": "http://10.20.0.11/",
            "framework": "ThinkPHP",
            "body_excerpt": "ThinkPHP V5.0.23 application",
        },
    )

    assert any(item["vulnerability_id"] == "struts2-s2-045" for item in json.loads(struts["stdout"]))
    assert any(item["vulnerability_id"] == "thinkphp-5-rce" for item in json.loads(thinkphp["stdout"]))


def _tool_names(catalog: dict[str, Any]) -> set[str]:
    return _tool_names_for(catalog, "pentest-tools")


def _tool_names_for(catalog: dict[str, Any], server_id: str) -> set[str]:
    tools = catalog[server_id]["tools"]
    return {tool["name"] for tool in tools}
