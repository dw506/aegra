"""Application API surface."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.app.orchestrator import AppOrchestrator, TargetHost
from src.app.settings import AppSettings

try:  # pragma: no cover - exercised only when FastAPI is installed
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover - default path in this workspace
    FastAPI = None
    HTTPException = RuntimeError


FASTAPI_UNAVAILABLE_MESSAGE = (
    "FastAPI is not installed. Install 'fastapi' and an ASGI server such as 'uvicorn' "
    "to use the HTTP control surface."
)


class OperationCreateRequest(BaseModel):
    """Request payload for creating a new operation."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImportTargetsRequest(BaseModel):
    """Request payload for updating the target inventory."""

    model_config = ConfigDict(extra="forbid")

    targets: list[TargetHost] = Field(default_factory=list)


class OperationActionRequest(BaseModel):
    """Request payload for simple operation management actions."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="manual_request", min_length=1)


def create_app(
    orchestrator: AppOrchestrator | None = None,
    settings: AppSettings | None = None,
):
    """Build the phase-one control API application."""

    if FastAPI is None:
        raise RuntimeError(FASTAPI_UNAVAILABLE_MESSAGE)

    resolved_settings = settings or AppSettings.from_env()
    resolved_orchestrator = orchestrator or AppOrchestrator(settings=resolved_settings)
    app = FastAPI(
        title=resolved_settings.control_api_title,
        version=resolved_settings.control_api_version,
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "service": resolved_settings.control_api_title,
            "status": "ok",
            "docs_url": "/docs",
            "openapi_url": "/openapi.json",
        }

    @app.get("/health")
    def healthcheck() -> dict[str, Any]:
        return resolved_orchestrator.get_health_status()

    @app.get("/ready")
    def readiness() -> dict[str, Any]:
        return resolved_orchestrator.get_readiness_status()

    @app.get("/operations")
    def list_operations() -> list[dict[str, Any]]:
        return [summary.model_dump(mode="json") for summary in resolved_orchestrator.list_operations()]

    @app.post("/operations")
    def create_operation(request: OperationCreateRequest) -> dict[str, Any]:
        try:
            state = resolved_orchestrator.create_operation(request.operation_id, metadata=request.metadata)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return state.model_dump(mode="json")

    @app.post("/operations/{operation_id}/targets")
    def import_targets(operation_id: str, request: ImportTargetsRequest) -> dict[str, Any]:
        try:
            state = resolved_orchestrator.import_targets(operation_id, request.targets)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return state.model_dump(mode="json")

    @app.post("/operations/{operation_id}/start")
    def start_operation(operation_id: str) -> dict[str, Any]:
        try:
            state = resolved_orchestrator.start_operation(operation_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return state.model_dump(mode="json")

    @app.get("/operations/{operation_id}")
    def get_operation(operation_id: str) -> dict[str, Any]:
        try:
            state = resolved_orchestrator.get_operation_state(operation_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return state.model_dump(mode="json")

    @app.get("/operations/{operation_id}/summary")
    def get_operation_summary(operation_id: str) -> dict[str, Any]:
        try:
            summary = resolved_orchestrator.get_operation_summary(operation_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return summary.model_dump(mode="json")

    @app.post("/operations/{operation_id}/resume")
    def resume_operation(operation_id: str, request: OperationActionRequest) -> dict[str, Any]:
        try:
            state = resolved_orchestrator.resume_operation(operation_id, reason=request.reason)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return resolved_orchestrator.get_operation_summary(operation_id).model_dump(mode="json")

    @app.post("/operations/{operation_id}/recover")
    def recover_operation(operation_id: str, request: OperationActionRequest) -> dict[str, Any]:
        try:
            resolved_orchestrator.recover_operation(operation_id, reason=request.reason)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return resolved_orchestrator.get_operation_summary(operation_id).model_dump(mode="json")

    @app.get("/operations/{operation_id}/audit")
    def export_audit(operation_id: str) -> dict[str, Any]:
        try:
            return resolved_orchestrator.export_audit_report(operation_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/operations/{operation_id}/llm-decisions")
    def get_llm_decisions(operation_id: str, limit: int = 20) -> list[dict[str, Any]]:
        try:
            return resolved_orchestrator.get_llm_decision_history(operation_id, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


app = create_app() if FastAPI is not None else None


__all__ = [
    "FASTAPI_UNAVAILABLE_MESSAGE",
    "ImportTargetsRequest",
    "OperationActionRequest",
    "OperationCreateRequest",
    "app",
    "create_app",
]
