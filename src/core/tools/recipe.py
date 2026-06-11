"""Safe tool recipe conversion for planner-produced task hints.

The LLM-facing contract is intentionally a small task object. This adapter owns
the command/function construction so free-form shell commands never need to come
from a model response.
"""

from __future__ import annotations

import re
import socket
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.tools.tool_runner import ToolExecutionSpec


class ToolRecipeError(ValueError):
    """Raised when a task cannot be converted into a safe recipe."""


class ToolTask(BaseModel):
    """Small planner-facing task description."""

    model_config = ConfigDict(extra="forbid")

    task_type: str = Field(min_length=1)
    tool_hint: Literal["http_probe", "nmap_service_scan"]
    target: str = Field(min_length=1)
    timeout_sec: int = Field(default=5, ge=1, le=60)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolRecipe(BaseModel):
    """Executable recipe produced by ``ToolRecipeAdapter``."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    recipe_name: Literal["http_probe", "nmap_service_scan"]
    execution_kind: Literal["python_function", "command"]
    target: str
    command: list[str] | None = None
    function: Callable[[], dict[str, Any]] | None = None
    timeout_sec: int = Field(default=5, ge=1, le=60)
    command_allowlist: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_execution_spec(self) -> ToolExecutionSpec:
        """Convert a command-backed recipe into a ToolRunner spec."""

        if self.execution_kind != "command" or not self.command:
            raise ToolRecipeError("only command recipes can be converted to ToolExecutionSpec")
        return ToolExecutionSpec(
            command=list(self.command),
            timeout_sec=self.timeout_sec,
            command_allowlist=set(self.command_allowlist),
            acceptable_exit_codes={0},
            stdout_max_bytes=65536,
            stderr_max_bytes=16384,
            policy_metadata={
                "kind": self.recipe_name,
                "name": self.recipe_name,
                "operation": self.metadata.get("task_type"),
                "tags": ["safe_probe", "recipe"],
            },
        )


class ToolRecipeAdapter:
    """Convert controlled task objects into safe commands or Python calls."""

    def __init__(self, *, nmap_path: str = "nmap") -> None:
        self._nmap_path = nmap_path

    def build(self, task: ToolTask | dict[str, Any]) -> ToolRecipe:
        """Build a recipe from a validated task object."""

        normalized = task if isinstance(task, ToolTask) else ToolTask.model_validate(task)
        if normalized.tool_hint == "http_probe":
            return self._build_http_probe(normalized)
        if normalized.tool_hint == "nmap_service_scan":
            return self._build_nmap_service_scan(normalized)
        raise ToolRecipeError(f"unsupported tool hint: {normalized.tool_hint}")

    def _build_http_probe(self, task: ToolTask) -> ToolRecipe:
        target = _normalize_http_url(task.target)
        parsed = urlparse(target)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        def run_probe() -> dict[str, Any]:
            return _http_probe(target=target, host=host, port=port, timeout_sec=task.timeout_sec)

        return ToolRecipe(
            recipe_name="http_probe",
            execution_kind="python_function",
            target=target,
            function=run_probe,
            timeout_sec=task.timeout_sec,
            metadata={"task_type": task.task_type, "host": host, "port": port, **task.metadata},
        )

    def _build_nmap_service_scan(self, task: ToolTask) -> ToolRecipe:
        host, port = _normalize_nmap_target(task.target, task.metadata)
        command = [self._nmap_path, "-n", "-Pn", "-sV", "-p", str(port), host]
        return ToolRecipe(
            recipe_name="nmap_service_scan",
            execution_kind="command",
            target=f"{host}:{port}",
            command=command,
            timeout_sec=max(task.timeout_sec, 10),
            command_allowlist={_command_basename(self._nmap_path)},
            metadata={"task_type": task.task_type, "host": host, "port": port, **task.metadata},
        )


def _http_probe(*, target: str, host: str, port: int, timeout_sec: int) -> dict[str, Any]:
    body = ""
    headers: dict[str, str] = {}
    status: int | None = None
    reachable = False
    failure_reason: str | None = None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(target, headers={"User-Agent": "Aegra-HTTP-Probe/1.0"}, method="GET")
    try:
        with opener.open(request, timeout=timeout_sec) as response:
            status = int(response.getcode())
            headers = {str(key): str(value) for key, value in response.headers.items()}
            body = response.read(4096).decode("utf-8", "ignore")
            reachable = True
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        headers = {str(key): str(value) for key, value in exc.headers.items()}
        body = exc.read(4096).decode("utf-8", "ignore")
        reachable = True
        failure_reason = f"http_error:{exc.code}"
    except Exception as exc:  # pragma: no cover - exact network errors vary by OS.
        failure_reason = str(exc)

    title = _extract_title(body)
    banner = headers.get("Server") or title or "http"
    service_id = f"{host}:{port}/tcp"
    return {
        "summary": f"http probe {status if status is not None else 'unreachable'} for {target}",
        "reachable": reachable,
        "success": reachable,
        "confidence": 0.9 if reachable else 0.0,
        "failure_reason": failure_reason,
        "target": target,
        "host": host,
        "port": port,
        "http_status": status,
        "title": title,
        "banner": banner,
        "entities": [
            {"id": host, "type": "Host", "label": host, "hostname": host, "status": "validated"},
            {
                "id": service_id,
                "type": "Service",
                "label": f"http-{port}",
                "host_id": host,
                "port": port,
                "protocol": "tcp",
                "service_name": "http",
                "status": "validated" if reachable else "observed",
                "http_status": status,
                "title": title,
                "banner": banner,
            },
        ]
        if reachable
        else [],
        "relations": [
            {
                "type": "HOSTS",
                "source": host,
                "target": service_id,
                "attributes": {"port": port, "protocol": "tcp", "state": "open"},
            }
        ]
        if reachable
        else [],
        "runtime_hints": {"reachable": reachable, "http_status": status, "target_url": target},
        "raw_output": body[:4096],
    }


def _normalize_http_url(target: str) -> str:
    parsed = urlparse(str(target).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ToolRecipeError("http_probe target must be an http(s) URL")
    _validate_host(parsed.hostname)
    return urlunparse(parsed)


def _normalize_nmap_target(target: str, metadata: dict[str, Any]) -> tuple[str, int]:
    parsed = urlparse(str(target).strip())
    host = parsed.hostname
    port: int | None = parsed.port
    if host is None:
        raw = str(target).strip()
        if ":" in raw and not raw.startswith("["):
            host_part, port_part = raw.rsplit(":", 1)
            host = host_part.strip("[]")
            port = int(port_part)
        else:
            host = raw
    if port is None and metadata.get("port") is not None:
        port = int(metadata["port"])
    if port is None:
        raise ToolRecipeError("nmap_service_scan requires a target port")
    _validate_host(host)
    if port < 1 or port > 65535:
        raise ToolRecipeError("target port must be between 1 and 65535")
    return host, port


def _validate_host(host: str) -> None:
    if not host:
        raise ToolRecipeError("target host is required")
    try:
        socket.inet_pton(socket.AF_INET, host)
        return
    except OSError:
        pass
    if not re.fullmatch(r"[A-Za-z0-9.-]{1,253}", host):
        raise ToolRecipeError("target host contains unsafe characters")


def _extract_title(body: str) -> str | None:
    match = re.search(r"<title[^>]*>(?P<title>.*?)</title>", body, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    title = re.sub(r"\s+", " ", match.group("title")).strip()
    return title[:160] or None


def _command_basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1].lower()


__all__ = ["ToolRecipe", "ToolRecipeAdapter", "ToolRecipeError", "ToolTask"]

