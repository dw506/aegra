"""Newline-delimited JSON-RPC MCP server for isolated lab tooling."""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from src.integrations.mcp_lab.tools import call_lab_tool, lab_tool_specs

#基于 stdin / stdout 的 JSON-RPC MCP server
SERVER_INFO = {"name": "aegra-mcp-lab", "version": "0.1.0"}
SERVER_CAPABILITIES = {"tools": {"listChanged": False}}


def main() -> int:
    """Run the stdio MCP lab server."""

    run_stdio_server(sys.stdin, sys.stdout)
    return 0


def run_stdio_server(stdin: TextIO, stdout: TextIO) -> None:
    """Serve newline-delimited JSON-RPC requests on stdin/stdout."""

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
        except Exception as exc:
            response = _response(None, error={"code": -32700, "message": str(exc)})
        if response is None:
            continue
        stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        stdout.flush()


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC request or notification."""

    method = str(request.get("method", ""))
    request_id = request.get("id")
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    if request_id is None and method.startswith("notifications/"):
        return None
    if method == "initialize":
        return _response(
            request_id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": SERVER_CAPABILITIES,
                "serverInfo": SERVER_INFO,
            },
        )
    if method == "tools/list":
        return _response(request_id, result={"tools": lab_tool_specs()})
    if method == "tools/call":
        tool_name = str(params.get("name", ""))
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        payload = call_lab_tool(tool_name, arguments)
        return _response(request_id, result=_mcp_tool_result(payload))
    return _response(request_id, error={"code": -32601, "message": f"method not found: {method}"})


def _mcp_tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": payload,
        "isError": not bool(payload.get("success", False)),
    }


def _response(
    request_id: Any,
    *,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        response["error"] = error
    else:
        response["result"] = result or {}
    return response


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["handle_request", "main", "run_stdio_server"]
