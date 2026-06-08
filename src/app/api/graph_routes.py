"""Graph and read-model route registration boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.visualization.graph_publisher import graph_delta_publisher
from src.core.visualization.graph_serializer import build_visual_snapshot
from src.core.visualization.unified_visualization import build_unified_visualization

try:  # pragma: no cover - exercised only when FastAPI is installed
    from fastapi import HTTPException, WebSocket, WebSocketDisconnect
except ImportError:  # pragma: no cover
    HTTPException = RuntimeError
    WebSocket = Any  # type: ignore
    WebSocketDisconnect = RuntimeError  # type: ignore


def register_graph_routes(app: Any, orchestrator: Any, settings: Any) -> None:
    """Register visualization graph snapshot and WebSocket routes."""

    @app.get("/operations/{operation_id}/visual-graphs/snapshot")
    def get_visual_graph_snapshot(operation_id: str) -> dict[str, Any]:
        try:
            snapshot = _build_snapshot(orchestrator, operation_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return snapshot.model_dump(mode="json")

    @app.get("/operations/{operation_id}/visualization")
    def get_operation_visualization(operation_id: str) -> dict[str, Any]:
        try:
            return _build_unified_visualization(orchestrator, operation_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/operations/{operation_id}/trace")
    def get_operation_trace(operation_id: str) -> dict[str, Any]:
        try:
            orchestrator.runtime_store.snapshot(operation_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        path = Path(settings.runtime_store_dir) / operation_id / "operation-trace.txt"
        if not path.exists():
            return {
                "operation_id": operation_id,
                "path": str(path),
                "text": "",
                "message": "operation trace file does not exist",
            }
        return {
            "operation_id": operation_id,
            "path": str(path),
            "text": path.read_text(encoding="utf-8"),
        }

    @app.websocket("/operations/{operation_id}/visual-graphs/ws")
    async def visual_graph_ws(websocket: WebSocket, operation_id: str) -> None:
        await websocket.accept()
        try:
            snapshot = _build_snapshot(orchestrator, operation_id)
        except ValueError:
            await websocket.close(code=1008)
            return

        await websocket.send_json(snapshot.model_dump(mode="json"))
        queue = graph_delta_publisher.subscribe(operation_id)
        try:
            while True:
                delta = await queue.get()
                await websocket.send_json(delta.model_dump(mode="json"))
        except WebSocketDisconnect:
            graph_delta_publisher.unsubscribe(operation_id, queue)
        except Exception:
            graph_delta_publisher.unsubscribe(operation_id, queue)
            raise


def _build_snapshot(orchestrator: Any, operation_id: str) -> Any:
    runtime_state = orchestrator.runtime_store.snapshot(operation_id)
    kg = orchestrator.graph_memory_store.load_kg(operation_id)
    ag = orchestrator.graph_memory_store.load_ag(operation_id)
    return build_visual_snapshot(
        operation_id=operation_id,
        kg_payload=kg.to_dict(),
        ag_payload=ag.to_dict(),
        runtime_state=runtime_state,
    )


def _build_unified_visualization(orchestrator: Any, operation_id: str) -> dict[str, Any]:
    runtime_state = orchestrator.runtime_store.snapshot(operation_id)
    kg = orchestrator.graph_memory_store.load_kg(operation_id)
    ag = orchestrator.graph_memory_store.load_ag(operation_id)
    return build_unified_visualization(
        operation_id=operation_id,
        kg_payload=kg.to_dict(),
        ag_payload=ag.to_dict(),
        runtime_state=runtime_state,
    )
