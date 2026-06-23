"""Application API surface."""

from __future__ import annotations

from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.app.orchestrator import AppOrchestrator, TargetHost
from src.app.settings import AppSettings
from src.core.agents.agent_protocol import GraphRef, GraphScope
from src.core.models.runtime import RuntimeStatus
from src.core.runtime.policy import policy_from_runtime_state

try:  # pragma: no cover - exercised only when FastAPI is installed
    from fastapi import FastAPI, HTTPException, Query, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
except ImportError:  # pragma: no cover - default path in this workspace
    FastAPI = None
    HTTPException = RuntimeError
    Query = None
    Response = None
    CORSMiddleware = None
    StaticFiles = None


FASTAPI_UNAVAILABLE_MESSAGE = (
    "FastAPI is not installed. Install 'fastapi' and an ASGI server such as 'uvicorn' "
    "to use the HTTP control surface."
)
LLM_DECISION_QUERY_LIMIT_MAX = 200
CONTROL_CYCLE_QUERY_LIMIT_MAX = 200
AUDIT_REPORT_QUERY_LIMIT_MAX = 500


class OperationCreateRequest(BaseModel):
    """Request payload for creating a new operation."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImportTargetsRequest(BaseModel):
    """Request payload for updating the target inventory."""

    model_config = ConfigDict(extra="forbid")

    targets: list[TargetHost] = Field(default_factory=list)


class WorkspaceCreateRequest(BaseModel):
    """Product-level workspace creation payload."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssetCreateRequest(TargetHost):
    """Product-level authorized asset import payload."""


class OperationActionRequest(BaseModel):
    """Request payload for simple operation management actions."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="manual_request", min_length=1)


class OperationCycleRequest(BaseModel):
    """Request payload for one operation control cycle."""

    model_config = ConfigDict(extra="forbid")

    graph_refs: list[GraphRef] = Field(default_factory=list)
    planner_payload: dict[str, Any] = Field(default_factory=dict)
    feedback_payload: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class OperationRunRequest(OperationCycleRequest):
    """Request payload for bounded multi-cycle operation execution."""

    max_cycles: int = Field(default=5, ge=1, le=50)
    stop_when_quiescent: bool = True
    max_replans: int = Field(default=3, ge=0, le=20)
    consecutive_llm_rejections: int = Field(default=3, ge=1, le=20)


class OperationSubmitRequest(OperationRunRequest):
    """Unified operation payload.

    When targets are provided, `POST /operations` creates the operation, imports
    the targets, starts the runtime and runs the bounded operation loop.
    """

    operation_id: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    targets: list[TargetHost] = Field(default_factory=list)
    auto_start: bool = True
    run: bool = True


def _default_graph_refs() -> list[GraphRef]:
    return [
        GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
        GraphRef(graph=GraphScope.AG, ref_id="ag-root", ref_type="graph"),
    ]


def _ensure_operation_runnable(orchestrator: AppOrchestrator, operation_id: str) -> None:
    """Apply minimal API-side execution guard before dispatching work.

    Policy is intentionally audit-only in the current automated main path. The
    API must not block `/cycle` or `/run` because of runtime policy scope fields;
    tool policy checks record their original decisions separately.
    """

    state = orchestrator.get_operation_state(operation_id)
    if state.operation_status == RuntimeStatus.CANCELLED:
        raise HTTPException(status_code=409, detail="operation is cancelled")
    targets = list(state.execution.metadata.get("target_inventory", []))
    if not targets:
        raise HTTPException(status_code=409, detail="operation has no imported targets")
    for target in targets:
        address = str(
            target.get("value") or target.get("address") or target.get("url") or target.get("hostname")
            if isinstance(target, dict)
            else ""
        ).strip()
        if not address:
            raise HTTPException(status_code=400, detail="target inventory contains a target without address")


def _workspace_from_summary(summary: Any) -> dict[str, Any]:
    metadata = dict(summary.metadata or {})
    workspace = dict(metadata.get("workspace") or {})
    return {
        "id": summary.operation_id,
        "name": workspace.get("name") or metadata.get("name") or summary.operation_id,
        "description": workspace.get("description"),
        "operation_id": summary.operation_id,
        "status": summary.operation_status.value,
        "asset_count": summary.target_count,
        "last_updated": summary.last_updated,
        "metadata": metadata,
        "links": {
            "assets": f"/workspaces/{summary.operation_id}/assets",
            "operation": f"/operations/{summary.operation_id}",
            "audit": f"/operations/{summary.operation_id}/audit-report",
        },
    }


def _assets_from_state(state: Any) -> list[dict[str, Any]]:
    assets = state.execution.metadata.get("target_inventory", [])
    if not isinstance(assets, list):
        return []
    return [
        {
            **dict(item),
            "workspace_id": state.operation_id,
            "operation_id": state.operation_id,
            "authorized": True,
            "links": {
                "audit": f"/operations/{state.operation_id}/audit-report",
                "operation": f"/operations/{state.operation_id}",
            },
        }
        for item in assets
        if isinstance(item, dict)
    ]


def _operation_policy_summary(orchestrator: AppOrchestrator, operation_id: str) -> dict[str, Any]:
    state = orchestrator.get_operation_state(operation_id)
    policy = policy_from_runtime_state(state)
    return {
        "scope": {
            "target_count": int(state.execution.metadata.get("target_count", 0)),
            "targets": list(state.execution.metadata.get("target_inventory", [])),
        },
        "policy": policy.to_runtime_metadata(),
        "audit_link": f"/operations/{operation_id}/audit-report",
    }


def _run_operation_response(
    orchestrator: AppOrchestrator,
    operation_id: str,
    request: OperationRunRequest,
) -> dict[str, Any]:
    _ensure_operation_runnable(orchestrator, operation_id)
    results = orchestrator.run_until_quiescent(
        operation_id,
        graph_refs=request.graph_refs or _default_graph_refs(),
        planner_payload=dict(request.planner_payload),
        feedback_payload=dict(request.feedback_payload),
        context=dict(request.context),
        max_cycles=request.max_cycles,
        max_replans=request.max_replans,
        consecutive_llm_rejections=request.consecutive_llm_rejections,
        stop_when_quiescent=request.stop_when_quiescent,
    )
    run_summary = orchestrator.get_operation_run_summary(
        operation_id,
        cycle_results=results,
        max_cycles=request.max_cycles,
    ).model_dump(mode="json")
    return {
        **run_summary,
        "result": run_summary,
        "operation": orchestrator.get_operation_summary(operation_id).model_dump(mode="json"),
        "policy_summary": _operation_policy_summary(orchestrator, operation_id),
        "cycles": [cycle_result.model_dump(mode="json") for cycle_result in results],
    }


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

    if CORSMiddleware is not None and resolved_settings.control_api_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_settings.control_api_cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    dashboard_dist = Path(__file__).resolve().parents[3] / "web" / "dashboard" / "dist"
    if StaticFiles is not None and dashboard_dist.exists():
        app.mount("/ui", StaticFiles(directory=dashboard_dist, html=True), name="ui")

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

    @app.get("/workspaces")
    def list_workspaces() -> list[dict[str, Any]]:
        return [_workspace_from_summary(summary) for summary in resolved_orchestrator.list_operations()]

    @app.post("/workspaces")
    def create_workspace(request: WorkspaceCreateRequest) -> dict[str, Any]:
        metadata = dict(request.metadata)
        metadata["workspace"] = {
            "id": request.id,
            "name": request.name or request.id,
            "description": request.description,
        }
        try:
            state = resolved_orchestrator.create_operation(request.id, metadata=metadata)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _workspace_from_summary(resolved_orchestrator.get_operation_summary(state.operation_id))

    @app.get("/workspaces/{workspace_id}/assets")
    def list_workspace_assets(workspace_id: str) -> list[dict[str, Any]]:
        try:
            state = resolved_orchestrator.get_operation_state(workspace_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _assets_from_state(state)

    @app.post("/workspaces/{workspace_id}/assets")
    def create_workspace_asset(workspace_id: str, request: AssetCreateRequest) -> dict[str, Any]:
        try:
            state = resolved_orchestrator.get_operation_state(workspace_id)
            target_inventory = state.execution.metadata.get("target_inventory", [])
            existing = [
                TargetHost.model_validate(item)
                for item in target_inventory
                if isinstance(item, dict)
            ]
            updated = resolved_orchestrator.import_targets(workspace_id, [*existing, TargetHost.model_validate(request.model_dump())])
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        assets = _assets_from_state(updated)
        return assets[-1] if assets else {}

    @app.post("/operations")
    def submit_operation(request: OperationSubmitRequest) -> dict[str, Any]:
        try:
            state = resolved_orchestrator.create_operation(request.operation_id, metadata=request.metadata)
            if request.targets:
                state = resolved_orchestrator.import_targets(request.operation_id, request.targets)
                if request.auto_start:
                    state = resolved_orchestrator.start_operation(request.operation_id)
                if request.run:
                    return _run_operation_response(resolved_orchestrator, request.operation_id, request)
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

    @app.post("/operations/{operation_id}/cycle")
    def run_operation_cycle(operation_id: str, request: OperationCycleRequest) -> dict[str, Any]:
        try:
            _ensure_operation_runnable(resolved_orchestrator, operation_id)
            result = resolved_orchestrator.run_operation_cycle(
                operation_id,
                graph_refs=request.graph_refs or _default_graph_refs(),
                planner_payload=dict(request.planner_payload),
                feedback_payload=dict(request.feedback_payload),
                context=dict(request.context),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return result.model_dump(mode="json")

    @app.post("/operations/{operation_id}/run")
    def run_operation(operation_id: str, request: OperationRunRequest | None = None) -> dict[str, Any]:
        request = request or OperationRunRequest()
        try:
            return _run_operation_response(resolved_orchestrator, operation_id, request)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/operations/{operation_id}/stop")
    def stop_operation(operation_id: str, request: OperationActionRequest) -> dict[str, Any]:
        try:
            state = resolved_orchestrator.stop_operation(operation_id, reason=request.reason)
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

    @app.get("/operations/{operation_id}/findings")
    def list_findings(operation_id: str) -> list[dict[str, Any]]:
        try:
            return resolved_orchestrator.list_findings(operation_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/operations/{operation_id}/evidence")
    def list_evidence(operation_id: str) -> list[dict[str, Any]]:
        try:
            return resolved_orchestrator.list_evidence(operation_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/operations/{operation_id}/graph")
    def get_findings_graph(operation_id: str) -> dict[str, Any]:
        try:
            return resolved_orchestrator.get_findings_graph(operation_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/operations/{operation_id}/report")
    def export_findings_report(
        operation_id: str,
        format: str = Query("json", pattern="^(json|csv|md)$"),
    ) -> Any:
        try:
            payload = resolved_orchestrator.export_findings_report(operation_id, format=format)  # type: ignore[arg-type]
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if format == "csv":
            return Response(content=str(payload), media_type="text/csv")
        if format == "md":
            return Response(content=str(payload), media_type="text/markdown")
        return payload

    @app.get("/operations/{operation_id}/llm-decisions")
    def get_llm_decisions(
        operation_id: str,
        limit: int = Query(20, ge=0, le=LLM_DECISION_QUERY_LIMIT_MAX),
        agent_kind: str | None = None,
        accepted: bool | None = None,
    ) -> list[dict[str, Any]]:
        try:
            return resolved_orchestrator.get_llm_decision_history(
                operation_id,
                limit=limit,
                agent_kind=agent_kind,
                accepted=accepted,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/operations/{operation_id}/control-cycles")
    def get_control_cycles(
        operation_id: str,
        limit: int = Query(20, ge=0, le=CONTROL_CYCLE_QUERY_LIMIT_MAX),
    ) -> list[dict[str, Any]]:
        try:
            return resolved_orchestrator.get_control_cycle_history(operation_id, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/operations/{operation_id}/audit-report")
    def get_operation_audit_report(
        operation_id: str,
        limit: int = Query(100, ge=0, le=AUDIT_REPORT_QUERY_LIMIT_MAX),
        agent_kind: str | None = None,
        accepted: bool | None = None,
    ) -> dict[str, Any]:
        try:
            return resolved_orchestrator.get_operation_audit_report(
                operation_id,
                limit=limit,
                agent_kind=agent_kind,
                accepted=accepted,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    return app


app = create_app() if FastAPI is not None else None


__all__ = [
    "FASTAPI_UNAVAILABLE_MESSAGE",
    "AUDIT_REPORT_QUERY_LIMIT_MAX",
    "CONTROL_CYCLE_QUERY_LIMIT_MAX",
    "ImportTargetsRequest",
    "LLM_DECISION_QUERY_LIMIT_MAX",
    "OperationActionRequest",
    "OperationCreateRequest",
    "OperationCycleRequest",
    "OperationRunRequest",
    "OperationSubmitRequest",
    "app",
    "create_app",
]
