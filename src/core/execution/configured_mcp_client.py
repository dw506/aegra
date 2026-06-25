"""Configuration-driven MCP client for experimental worker execution."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from pydantic import BaseModel, ConfigDict, Field

from src.core.execution.mcp_client import MCPToolCallResult


class MCPServerConfig(BaseModel):
    """One configured MCP server endpoint."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    transport: Literal["stdio", "http"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class MCPRuntimeConfig(BaseModel):
    """Runtime MCP configuration loaded from settings or environment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class ConfiguredMCPClient:
    """JSON-RPC MCP client with configured stdio and HTTP transports.

    The implementation is intentionally compact, but stdio servers are kept as
    managed sessions so tool catalog lookups and tool calls can share one MCP
    initialization handshake.
    """

    def __init__(self, config: MCPRuntimeConfig | dict[str, Any] | None = None) -> None:
        self._config = config if isinstance(config, MCPRuntimeConfig) else MCPRuntimeConfig.model_validate(config or {})
        self._stdio_sessions: dict[str, _StdioMCPSession] = {}
        self._tools_cache: dict[str, Any] | None = None

    @classmethod
    def from_sources(
        cls,
        *,
        config_path: Path | str | None = None,
        config_json: dict[str, Any] | None = None,
    ) -> "ConfiguredMCPClient":
        payload: dict[str, Any] = {}
        if config_path is not None:
            path = Path(config_path).expanduser().resolve()
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise ValueError(f"MCP config file not found: {path}") from exc
            except Exception as exc:
                raise ValueError(f"failed to read MCP config file '{path}': {exc}") from exc
            if not isinstance(loaded, dict):
                raise ValueError(f"MCP config file '{path}' must contain a JSON object")
            payload.update(loaded)
        if config_json:
            payload.update(config_json)
        return cls(payload)

    def is_available(self, server_id: str | None = None) -> bool:
        if not server_id:
            return bool(self._config.servers)
        return server_id in self._config.servers

    def close(self) -> None:
        """Close all managed stdio sessions."""

        for session in list(self._stdio_sessions.values()):
            session.close()
        self._stdio_sessions.clear()

    def list_tools(self) -> dict[str, Any]:
        if self._tools_cache is not None:
            return dict(self._tools_cache)
        catalog: dict[str, Any] = {}
        for server_id, server in self._config.servers.items():
            try:
                raw = self._call_rpc(server_id=server_id, server=server, method="tools/list", params={}, timeout_seconds=15)
                result = raw.get("result") if isinstance(raw, dict) else None
                catalog[server_id] = result if isinstance(result, dict) else {"available": True}
            except Exception as exc:
                catalog[server_id] = {"available": False, "error": str(exc)}
        self._tools_cache = dict(catalog)
        return catalog

    def call_tool(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_seconds: int,
    ) -> MCPToolCallResult:
        server = self._config.servers.get(server_id)
        if server is None:
            return MCPToolCallResult(
                success=False,
                exit_code="mcp_server_not_configured",
                stderr=f"MCP server '{server_id}' is not configured",
            )
        try:
            raw = self._call_rpc(
                server_id=server_id,
                server=server,
                method="tools/call",
                params={"name": tool_name, "arguments": arguments},
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError as exc:
            return MCPToolCallResult(success=False, exit_code="timeout", stderr=str(exc), metadata={"server_id": server_id})
        except Exception as exc:
            return MCPToolCallResult(success=False, exit_code="mcp_error", stderr=str(exc), metadata={"server_id": server_id})
        return self._tool_result_from_rpc(server_id=server_id, raw=raw)

    def _call_rpc(
        self,
        *,
        server_id: str,
        server: MCPServerConfig,
        method: str,
        params: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        if server.transport == "http":
            return self._call_http(server=server, method=method, params=params, timeout_seconds=timeout_seconds)
        return self._call_stdio(server_id=server_id, server=server, method=method, params=params, timeout_seconds=timeout_seconds)

    def _call_http(
        self,
        *,
        server: MCPServerConfig,
        method: str,
        params: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        if not server.url:
            raise ValueError("HTTP MCP server requires url")
        payload = self._rpc_payload(1, method, params)
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **server.headers}
        request = Request(server.url, data=body, headers=headers, method="POST")
        opener = self._http_opener_for_url(server.url)
        open_call = opener.open if opener is not None else urlopen
        with open_call(request, timeout=timeout_seconds) as response:
            decoded = response.read().decode("utf-8")
        parsed = json.loads(decoded)
        if not isinstance(parsed, dict):
            raise ValueError("MCP HTTP response must be a JSON object")
        return parsed

    @staticmethod
    def _http_opener_for_url(url: str) -> Any | None:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if host in {"127.0.0.1", "localhost", "::1"}:
            return build_opener(ProxyHandler({}))
        return None

    def _call_stdio(
        self,
        *,
        server_id: str,
        server: MCPServerConfig,
        method: str,
        params: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        if not server.command:
            raise ValueError("stdio MCP server requires command")
        session = self._stdio_sessions.get(server_id)
        if session is None or not session.is_alive:
            session = _StdioMCPSession(server_id=server_id, server=server)
            self._stdio_sessions[server_id] = session
        return session.request(method=method, params=params, timeout_seconds=timeout_seconds)

    def _tool_result_from_rpc(self, *, server_id: str, raw: dict[str, Any]) -> MCPToolCallResult:
        if raw.get("error"):
            return MCPToolCallResult(
                success=False,
                exit_code="mcp_error",
                stderr=json.dumps(raw["error"], ensure_ascii=True, sort_keys=True),
                metadata={"server_id": server_id, "raw_mcp": raw},
            )
        result = raw.get("result")
        content = self._structured_content(result)
        is_error = bool(result.get("isError", False)) if isinstance(result, dict) else False
        stdout = self._payload_field(content, "stdout") or self._content_to_stdout(content)
        stderr = self._payload_field(content, "stderr")
        exit_code = self._payload_field(content, "exit_code")
        payload_success = self._payload_field(content, "success")
        parsed_output = self._payload_field(content, "parsed")
        payload_metadata = self._payload_field(content, "metadata")
        raw_output_ref = self._payload_field(content, "raw_output_ref")
        metadata = payload_metadata if isinstance(payload_metadata, dict) else {}
        if raw_output_ref and "raw_output_ref" not in metadata:
            metadata["raw_output_ref"] = str(raw_output_ref)
        if isinstance(payload_success, bool):
            success = (not is_error) and payload_success
        else:
            success = (not is_error) and self._successful_exit_code(exit_code)
        return MCPToolCallResult(
            success=success,
            content=content,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            metadata={
                "server_id": server_id,
                "raw_mcp": raw,
                "parsed_output": parsed_output or {},
                **metadata,
            },
        )

    @staticmethod
    def _rpc_payload(request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}

    @staticmethod
    def _write_message(process: subprocess.Popen[str], payload: dict[str, Any]) -> None:
        if process.stdin is None:
            raise RuntimeError("MCP stdio process has no stdin")
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()

    @staticmethod
    def _structured_content(result: Any) -> Any:
        if not isinstance(result, dict):
            return result
        if "structuredContent" in result:
            return result["structuredContent"]
        content_items = result.get("content")
        if isinstance(content_items, list):
            text_parts: list[str] = []
            for item in content_items:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            text = "\n".join(part for part in text_parts if part)
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
        return result

    @staticmethod
    def _payload_field(content: Any, field: str) -> Any:
        if isinstance(content, dict):
            return content.get(field)
        return None

    @staticmethod
    def _successful_exit_code(exit_code: Any) -> bool:
        if exit_code is None:
            return True
        return exit_code == 0 or exit_code == "0"

    @staticmethod
    def _content_to_stdout(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=True, sort_keys=True)


class _StdioMCPSession:
    """Managed newline-delimited JSON-RPC session for one stdio MCP server."""

    def __init__(self, *, server_id: str, server: MCPServerConfig) -> None:
        self.server_id = server_id
        self.server = server
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, Any] | Exception | None] = queue.Queue()
        self._pending: dict[int, dict[str, Any]] = {}
        self._stderr_lines: list[str] = []
        self._request_id = 0
        self._initialized = False
        self._lock = threading.RLock()

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def request(self, *, method: str, params: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
        with self._lock:
            self._ensure_started()
            self._ensure_initialized(timeout_seconds=timeout_seconds)
            request_id = self._next_id()
            self._write(ConfiguredMCPClient._rpc_payload(request_id, method, params))
            return self._wait_for_response(request_id=request_id, timeout_seconds=timeout_seconds)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=1)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        finally:
            self._process = None
            self._initialized = False

    def _ensure_started(self) -> None:
        if self.is_alive:
            return
        env = os.environ.copy()
        env.update(self.server.env)
        self._process = subprocess.Popen(
            [str(self.server.command), *self.server.args],
            cwd=self.server.cwd or None,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._initialized = False
        self._responses = queue.Queue()
        self._pending.clear()
        self._stderr_lines.clear()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _ensure_initialized(self, *, timeout_seconds: int) -> None:
        if self._initialized:
            return
        request_id = self._next_id()
        self._write(
            ConfiguredMCPClient._rpc_payload(
                request_id,
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "aegra", "version": "0.1.0"},
                },
            )
        )
        response = self._wait_for_response(request_id=request_id, timeout_seconds=timeout_seconds)
        if response.get("error"):
            raise RuntimeError(f"MCP stdio server '{self.server_id}' initialize failed: {response['error']}")
        self._write({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        self._initialized = True

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _write(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError(f"MCP stdio server '{self.server_id}' is not running")
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()

    def _wait_for_response(self, *, request_id: int, timeout_seconds: int) -> dict[str, Any]:
        pending = self._pending.pop(request_id, None)
        if pending is not None:
            return pending
        while True:
            try:
                item = self._responses.get(timeout=timeout_seconds)
            except queue.Empty as exc:
                raise TimeoutError(f"MCP stdio request {request_id} to '{self.server_id}' timed out") from exc
            if item is None:
                stderr = self._stderr_excerpt()
                raise RuntimeError(f"MCP stdio process '{self.server_id}' exited before response {request_id}: {stderr}")
            if isinstance(item, Exception):
                raise RuntimeError(f"MCP stdio process '{self.server_id}' emitted invalid JSON: {item}") from item
            raw_id = item.get("id")
            if raw_id == request_id:
                return item
            if isinstance(raw_id, int):
                self._pending[raw_id] = item

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except Exception:
                    self._stderr_lines.append(f"stdout non-json: {line}")
                    del self._stderr_lines[:-20]
                    continue
                if isinstance(parsed, dict):
                    self._responses.put(parsed)
                else:
                    self._stderr_lines.append(f"stdout non-object json: {line}")
                    del self._stderr_lines[:-20]
        finally:
            self._responses.put(None)

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            text = line.strip()
            if text:
                self._stderr_lines.append(text)
                del self._stderr_lines[:-20]

    def _stderr_excerpt(self) -> str:
        return "\n".join(self._stderr_lines[-20:])

    def __del__(self) -> None:
        self.close()


__all__ = ["ConfiguredMCPClient", "MCPRuntimeConfig", "MCPServerConfig"]
