"""HTTP JSON-RPC MCP server for local lab tooling."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from src.integrations.mcp_lab.server import handle_request


class MCPHTTPHandler(BaseHTTPRequestHandler):
    """Serve MCP JSON-RPC requests over HTTP POST /mcp."""

    server_version = "AegraMCPHTTP/0.1"

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/mcp":
            self._send_json(404, {"error": "not found"})
            return
        self._send_json(
            200,
            {
                "status": "ok",
                "message": "POST JSON-RPC requests to /mcp",
                "methods": ["initialize", "tools/list", "tools/call"],
            },
        )

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/mcp":
            self._send_json(404, _json_rpc_error(None, -32601, "method endpoint not found"))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            request = json.loads(raw)
        except Exception as exc:
            self._send_json(400, _json_rpc_error(None, -32700, str(exc)))
            return
        if not isinstance(request, dict):
            self._send_json(400, _json_rpc_error(None, -32600, "JSON-RPC request must be an object"))
            return
        response = handle_request(request)
        if response is None:
            self.send_response(204)
            self.end_headers()
            return
        self._send_json(200, response)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _json_rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def run_http_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), MCPHTTPHandler)
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Aegra lab MCP HTTP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_http_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MCPHTTPHandler", "main", "run_http_server"]
