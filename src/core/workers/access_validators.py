"""Access validation adapters used by AccessWorker."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.events import AgentTaskRequest
from src.core.models.runtime import CredentialStatus, RuntimeState
from src.core.runtime.credential_manager import RuntimeCredentialManager
from src.core.runtime.session_manager import RuntimeSessionManager
from src.core.workers.tool_runner import ToolExecutionResult, ToolExecutionSpec, ToolRunner


class SessionProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    status: str = ""
    usable: bool = False
    blocked: bool = False
    failure_reason: str | None = None
    bound_target: str | None = None
    bound_identity: str | None = None
    reused_from_runtime: bool = False
    evidence: dict[str, Any] = Field(default_factory=dict)
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class CredentialValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    credential_id: str | None = None
    status: str = CredentialStatus.UNKNOWN.value
    validated: bool = False
    blocked: bool = False
    failure_reason: str | None = None
    principal: str | None = None
    target_id: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class PrivilegeValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validated: bool = True
    required_level: str | None = None
    blocked: bool = False
    failure_reason: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class SessionProbeAdapter:
    """Resolve session usability from runtime snapshot or an optional command."""

    def __init__(
        self,
        *,
        session_manager: RuntimeSessionManager | None = None,
        tool_runner: ToolRunner | None = None,
    ) -> None:
        self._session_manager = session_manager or RuntimeSessionManager()
        self._tool_runner = tool_runner or ToolRunner()

    def probe(
        self,
        *,
        request: AgentTaskRequest,
        runtime_state: RuntimeState | None,
        bound_target: str | None,
        bound_identity: str | None,
    ) -> SessionProbeResult:
        raw = _metadata_view(request, "session_probe", "runtime_session")
        session_id = _string(request.context.session_id) or _string(raw.get("session_id"))
        command = raw.get("command") or request.metadata.get("session_probe_command")
        if isinstance(command, list) and command:
            return self._probe_with_command(
                command=[str(item) for item in command],
                request=request,
                runtime_state=runtime_state,
                raw=raw,
                session_id=session_id,
                bound_target=bound_target,
                bound_identity=bound_identity,
            )

        status = str(raw.get("status", "")).lower()
        usable = bool(session_id and (raw.get("usable") is True or status in {"active", "opened", "ready"}))
        if runtime_state is not None and session_id is not None:
            try:
                session = self._session_manager.get_session(runtime_state, session_id)
            except ValueError:
                session = None
            if session is not None:
                return SessionProbeResult(
                    session_id=session.session_id,
                    status=session.status.value,
                    usable=self._session_manager.is_session_usable(runtime_state, session.session_id),
                    bound_target=session.bound_target,
                    bound_identity=session.bound_identity,
                    evidence={"source": "runtime_snapshot"},
                    raw_payload=raw | {
                        "session_id": session.session_id,
                        "status": session.status.value,
                        "bound_target": session.bound_target,
                        "bound_identity": session.bound_identity,
                    },
                )

        if runtime_state is not None and not usable:
            reusable = self._session_manager.list_reusable_sessions(
                runtime_state,
                bound_target=bound_target,
                bound_identity=bound_identity,
            )
            if reusable:
                session = reusable[0]
                return SessionProbeResult(
                    session_id=session.session_id,
                    status=session.status.value,
                    usable=True,
                    bound_target=session.bound_target,
                    bound_identity=session.bound_identity,
                    reused_from_runtime=True,
                    evidence={"source": "runtime_snapshot"},
                    raw_payload=raw | {
                        "session_id": session.session_id,
                        "status": session.status.value,
                        "bound_target": session.bound_target,
                        "bound_identity": session.bound_identity,
                        "reused_from_runtime": True,
                    },
                )

        return SessionProbeResult(
            session_id=session_id,
            status=status,
            usable=usable,
            bound_target=_string(raw.get("bound_target")) or bound_target,
            bound_identity=_string(raw.get("bound_identity")) or bound_identity,
            failure_reason=None if usable else _string(raw.get("reason")),
            evidence={"source": "request_metadata"},
            raw_payload=raw,
        )

    def _probe_with_command(
        self,
        *,
        command: list[str],
        request: AgentTaskRequest,
        runtime_state: RuntimeState | None,
        raw: dict[str, Any],
        session_id: str | None,
        bound_target: str | None,
        bound_identity: str | None,
    ) -> SessionProbeResult:
        tool_result = self._tool_runner.run(
            ToolExecutionSpec(
                command=command,
                timeout_sec=int(request.metadata.get("session_probe_timeout_sec", 15)),
                retries=int(request.metadata.get("session_probe_retries", 0)),
                cwd=request.metadata.get("session_probe_cwd"),
                env={str(key): str(value) for key, value in dict(request.metadata.get("session_probe_env", {})).items()},
            )
        )
        if tool_result.category in {"command_not_found", "process_error", "timeout"}:
            return SessionProbeResult(
                session_id=session_id,
                status="blocked",
                usable=False,
                blocked=True,
                failure_reason=_tool_failure_reason(tool_result, "session_probe"),
                bound_target=bound_target,
                bound_identity=bound_identity,
                evidence={"tool": tool_result.to_payload()},
                raw_payload=raw,
            )
        payload = _parse_json_payload(tool_result)
        if payload is None:
            usable = bool(tool_result.success)
            return SessionProbeResult(
                session_id=session_id,
                status=("active" if usable else "failed"),
                usable=usable,
                failure_reason=(None if usable else _tool_failure_reason(tool_result, "session_probe")),
                bound_target=bound_target,
                bound_identity=bound_identity,
                evidence={"tool": tool_result.to_payload()},
                raw_payload=raw | {"stdout": tool_result.stdout},
            )
        payload_session_id = _string(payload.get("session_id")) or session_id
        if runtime_state is not None and payload_session_id is not None:
            try:
                session = self._session_manager.get_session(runtime_state, payload_session_id)
            except ValueError:
                session = None
            if session is not None:
                bound_target = session.bound_target or bound_target
                bound_identity = session.bound_identity or bound_identity
        return SessionProbeResult(
            session_id=payload_session_id,
            status=str(payload.get("status", "active" if payload.get("usable") else "failed")).lower(),
            usable=bool(payload.get("usable", tool_result.success)),
            blocked=bool(payload.get("blocked", False)),
            failure_reason=_string(payload.get("failure_reason")) or _string(payload.get("reason")),
            bound_target=_string(payload.get("bound_target")) or bound_target,
            bound_identity=_string(payload.get("bound_identity")) or bound_identity,
            evidence={"tool": tool_result.to_payload()},
            raw_payload=payload,
        )


class CredentialValidatorAdapter:
    """Resolve credential validity from runtime snapshot or an optional command."""

    def __init__(
        self,
        *,
        credential_manager: RuntimeCredentialManager | None = None,
        tool_runner: ToolRunner | None = None,
    ) -> None:
        self._credential_manager = credential_manager or RuntimeCredentialManager()
        self._tool_runner = tool_runner or ToolRunner()

    def validate(
        self,
        *,
        request: AgentTaskRequest,
        runtime_state: RuntimeState | None,
        target_id: str | None,
    ) -> CredentialValidationResult:
        raw = _metadata_view(request, "credential_validation", "runtime_credential")
        credential_id = _string(raw.get("credential_id"))
        command = raw.get("command") or request.metadata.get("credential_validator_command")
        if isinstance(command, list) and command:
            return self._validate_with_command(
                command=[str(item) for item in command],
                request=request,
                raw=raw,
                credential_id=credential_id,
                target_id=target_id,
            )

        status = str(raw.get("status", request.metadata.get("credential_status", CredentialStatus.UNKNOWN.value))).lower()
        validated = raw.get("validated")
        if validated is None:
            validated = status == CredentialStatus.VALID.value

        if runtime_state is not None and credential_id is not None:
            credential = runtime_state.credentials.get(credential_id)
            if credential is not None and status == CredentialStatus.UNKNOWN.value:
                status = credential.status.value
                validated = credential.is_usable()
            elif credential is None and _string(raw.get("principal")):
                credential = self._credential_manager.upsert_credential(
                    runtime_state,
                    credential_id,
                    _string(raw.get("principal")) or "unknown-principal",
                    kind=str(raw.get("kind", "password")),
                    secret_ref=_string(raw.get("secret_ref")),
                    metadata={k: v for k, v in raw.items() if k not in {"credential_id", "principal", "kind", "secret_ref"}},
                )
                status = credential.status.value

            if credential is not None:
                self._credential_manager.record_validation(
                    runtime_state,
                    credential_id,
                    status=status,
                    target_id=target_id,
                    metadata={"validator_output": dict(raw)},
                )
                credential = runtime_state.credentials[credential_id]
                status = credential.status.value
                validated = credential.is_usable()

        return CredentialValidationResult(
            credential_id=credential_id,
            status=status,
            validated=bool(validated),
            failure_reason=(None if bool(validated) else _string(raw.get("reason"))),
            principal=_string(raw.get("principal")),
            target_id=target_id,
            evidence={"source": "runtime_snapshot" if runtime_state is not None else "request_metadata"},
            raw_payload=raw,
        )

    def _validate_with_command(
        self,
        *,
        command: list[str],
        request: AgentTaskRequest,
        raw: dict[str, Any],
        credential_id: str | None,
        target_id: str | None,
    ) -> CredentialValidationResult:
        tool_result = self._tool_runner.run(
            ToolExecutionSpec(
                command=command,
                timeout_sec=int(request.metadata.get("credential_validator_timeout_sec", 15)),
                retries=int(request.metadata.get("credential_validator_retries", 0)),
                cwd=request.metadata.get("credential_validator_cwd"),
                env={str(key): str(value) for key, value in dict(request.metadata.get("credential_validator_env", {})).items()},
                acceptable_exit_codes={0},
            )
        )
        if tool_result.category in {"command_not_found", "process_error", "timeout"}:
            return CredentialValidationResult(
                credential_id=credential_id,
                status=CredentialStatus.UNKNOWN.value,
                validated=False,
                blocked=True,
                failure_reason=_tool_failure_reason(tool_result, "credential_validator"),
                target_id=target_id,
                evidence={"tool": tool_result.to_payload()},
                raw_payload=raw,
            )
        payload = _parse_json_payload(tool_result)
        if payload is None:
            validated = bool(tool_result.success)
            return CredentialValidationResult(
                credential_id=credential_id,
                status=(CredentialStatus.VALID.value if validated else CredentialStatus.INVALID.value),
                validated=validated,
                failure_reason=(None if validated else _tool_failure_reason(tool_result, "credential_validator")),
                target_id=target_id,
                evidence={"tool": tool_result.to_payload()},
                raw_payload=raw | {"stdout": tool_result.stdout},
            )
        status = str(payload.get("status", CredentialStatus.VALID.value if payload.get("validated") else CredentialStatus.INVALID.value)).lower()
        return CredentialValidationResult(
            credential_id=_string(payload.get("credential_id")) or credential_id,
            status=status,
            validated=bool(payload.get("validated", status == CredentialStatus.VALID.value)),
            blocked=bool(payload.get("blocked", False)),
            failure_reason=_string(payload.get("failure_reason")) or _string(payload.get("reason")),
            principal=_string(payload.get("principal")),
            target_id=_string(payload.get("target_id")) or target_id,
            evidence={"tool": tool_result.to_payload()},
            raw_payload=payload,
        )


class PrivilegeValidatorAdapter:
    """Normalize privilege validation metadata into a stable adapter contract."""

    def validate(self, *, request: AgentTaskRequest) -> PrivilegeValidationResult:
        raw = _metadata_view(request, "privilege_validation")
        if "validated" not in raw:
            raw = {"validated": not bool(request.metadata.get("privilege_gap_detected", False)), **raw}
        return PrivilegeValidationResult(
            validated=bool(raw.get("validated", True)),
            required_level=_string(raw.get("required_level")),
            blocked=bool(raw.get("blocked", False)),
            failure_reason=_string(raw.get("failure_reason")) or _string(raw.get("reason")),
            evidence={"source": "request_metadata"},
            raw_payload=raw,
        )


def _metadata_view(request: AgentTaskRequest, *keys: str) -> dict[str, Any]:
    for key in keys:
        raw = request.metadata.get(key)
        if isinstance(raw, dict):
            return dict(raw)
        raw = request.context.metadata.get(key)
        if isinstance(raw, dict):
            return dict(raw)
    return {}


def _parse_json_payload(tool_result: ToolExecutionResult) -> dict[str, Any] | None:
    stdout = tool_result.stdout.strip()
    if not stdout:
        return None
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else {"value": payload}


def _tool_failure_reason(tool_result: ToolExecutionResult, prefix: str) -> str:
    detail = tool_result.error_message or tool_result.stderr.strip()
    if detail:
        return f"{prefix}: {detail}"
    if tool_result.exit_code is not None:
        return f"{prefix} exited with code {tool_result.exit_code}"
    return f"{prefix} failed"


def _string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "CredentialValidationResult",
    "CredentialValidatorAdapter",
    "PrivilegeValidationResult",
    "PrivilegeValidatorAdapter",
    "SessionProbeAdapter",
    "SessionProbeResult",
]
