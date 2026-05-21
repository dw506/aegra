"""HTTP client for an external Incalmo-compatible C2 API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict, Field

from src.app.settings import AppSettings
from src.integrations.incalmo.models import Agent, Command, CommandResult, CommandStatus


class IncalmoClientConfig(BaseModel):
    """Resolved Incalmo client configuration."""

    model_config = ConfigDict(extra="forbid")

    c2_url: str = Field(min_length=1)
    poll_interval_sec: float = Field(default=1.0, gt=0.0)
    command_timeout_sec: float = Field(default=60.0, gt=0.0)


class IncalmoClient:
    """Minimal JSON-over-HTTP client for Incalmo C2 operations."""

    def __init__(self, config: IncalmoClientConfig | str, timeout_seconds: int = 45) -> None:
        if isinstance(config, str):
            config = IncalmoClientConfig(c2_url=config, command_timeout_sec=float(timeout_seconds))
        self.config = config
        self._base_url = config.c2_url.rstrip("/")
        self._command_agents: dict[str, str] = {}
        self._opener = build_opener(ProxyHandler({}))

    @classmethod
    def from_settings(cls, settings: AppSettings, *, config_path: Path | None = None) -> "IncalmoClient":
        payload = _load_simple_yaml(config_path or Path("configs/incalmo.yaml"))
        c2_url = settings.incalmo_c2_url or str(payload.get("c2_url") or "")
        if not c2_url:
            raise ValueError("Incalmo C2 URL is required")
        return cls(
            IncalmoClientConfig(
                c2_url=c2_url,
                poll_interval_sec=float(payload.get("poll_interval_sec") or settings.incalmo_poll_interval_sec),
                command_timeout_sec=float(payload.get("command_timeout_sec") or settings.incalmo_command_timeout_sec),
            )
        )

    def get_agents(self) -> list[Agent]:
        payload = self._request("GET", "/agents")
        items = payload.get("agents", payload if isinstance(payload, list) else [])
        return [Agent.model_validate(item) for item in items if isinstance(item, dict)]

    def get_agent(self, agent_id: str) -> Agent:
        return Agent.model_validate(self._request("GET", f"/agents/{agent_id}"))

    def send_command(self, agent_id: str, command: str, payloads: Any = None) -> Command:
        response = self._request(
            "POST",
            f"/agents/{agent_id}/commands",
            {"command": command, "payloads": payloads or {}},
        )
        if "command_id" not in response:
            response["command_id"] = str(response.get("id") or response.get("uuid") or "")
        response.setdefault("agent_id", agent_id)
        response.setdefault("command", command)
        model = Command.model_validate(response)
        self._command_agents[model.command_id] = agent_id
        return model

    def command_status(self, agent_id: str, command_id: str | None = None) -> CommandResult:
        if command_id is None:
            command_id = agent_id
            agent_id = self._command_agents.get(command_id) or ""
        if not agent_id:
            raise ValueError(f"agent id is unknown for command_id={command_id}")
        response = self._request("GET", f"/agents/{agent_id}/commands/{command_id}")
        response.setdefault("agent_id", agent_id)
        response.setdefault("command_id", command_id)
        return CommandResult.model_validate(response)

    def wait_for_command(self, agent_id: str, command_id: str, *, timeout_sec: float | None = None) -> CommandResult:
        deadline = time.monotonic() + (timeout_sec or self.config.command_timeout_sec)
        last_result = self.command_status(agent_id, command_id)
        while last_result.status.value in {"pending", "running"} and time.monotonic() < deadline:
            time.sleep(self.config.poll_interval_sec)
            last_result = self.command_status(agent_id, command_id)
        if last_result.status.value in {"pending", "running"}:
            return last_result.model_copy(update={"status": CommandStatus.TIMEOUT})
        return last_result

    def wait_for_command_result(
        self,
        *,
        command_id: str,
        poll_interval_seconds: float = 1.0,
        max_attempts: int = 45,
    ) -> dict[str, Any]:
        agent_id = self._command_agents.get(command_id)
        if not agent_id:
            raise ValueError(f"agent id is unknown for command_id={command_id}")
        original_interval = self.config.poll_interval_sec
        try:
            self.config.poll_interval_sec = poll_interval_seconds
            result = self.wait_for_command(agent_id, command_id, timeout_sec=max(1, max_attempts) * poll_interval_seconds)
        finally:
            self.config.poll_interval_sec = original_interval
        payload = result.model_dump(mode="json")
        payload.setdefault("id", result.command_id)
        payload.setdefault("output", result.stdout)
        return payload

    def report_environment_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/environment", payload)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self._base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=self.config.command_timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Incalmo C2 HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Incalmo C2 request failed: {exc.reason}") from exc
        return json.loads(raw) if raw else {}


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"').strip("'")
        if value.lower() in {"true", "false"}:
            data[key.strip()] = value.lower() == "true"
        else:
            try:
                data[key.strip()] = float(value) if "." in value else int(value)
            except ValueError:
                data[key.strip()] = value
    return data
