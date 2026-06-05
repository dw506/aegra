"""File-backed Graph Memory Store for operation snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2
from threading import RLock
from typing import Any

from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.models.runtime import RuntimeState


class GraphMemoryStore:
    """Persist KG / AG / Runtime snapshots under one operation directory.

    The first implementation intentionally uses JSON files instead of an
    external graph database so local runs and tests can inspect every artifact.
    """

    KG_FILENAME = "kg.json"
    AG_FILENAME = "ag.json"
    RUNTIME_FILENAME = "runtime.json"
    SNAPSHOTS_DIRNAME = "snapshots"

    def __init__(self, root_dir: str | Path = "runtime-store") -> None:
        self._root_dir = Path(root_dir).resolve()
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    @property
    def root_dir(self) -> Path:
        """Return the resolved storage root."""

        return self._root_dir

    def load_kg(self, operation_id: str) -> KnowledgeGraph:
        """Load a Knowledge Graph snapshot, or return an empty graph when absent."""

        with self._lock:
            path = self._artifact_path(operation_id, self.KG_FILENAME)
            if not path.exists():
                return KnowledgeGraph()
            return KnowledgeGraph.from_dict(self._read_json(path))

    def save_kg(self, operation_id: str, kg: KnowledgeGraph) -> None:
        """Persist a Knowledge Graph snapshot."""

        with self._lock:
            self._write_json(self._artifact_path(operation_id, self.KG_FILENAME), kg.to_dict())

    def load_ag(self, operation_id: str) -> AttackGraph:
        """Load an Attack Graph snapshot, or return an empty graph when absent."""

        with self._lock:
            path = self._artifact_path(operation_id, self.AG_FILENAME)
            if not path.exists():
                return AttackGraph()
            return AttackGraph.from_dict(self._read_json(path))

    def save_ag(self, operation_id: str, ag: AttackGraph) -> None:
        """Persist an Attack Graph snapshot."""

        with self._lock:
            self._write_json(self._artifact_path(operation_id, self.AG_FILENAME), ag.to_dict())

    def load_runtime(self, operation_id: str) -> RuntimeState | None:
        """Load a RuntimeState snapshot, or return None when absent."""

        with self._lock:
            path = self._artifact_path(operation_id, self.RUNTIME_FILENAME)
            if not path.exists():
                return None
            return RuntimeState.model_validate(self._read_json(path))

    def save_runtime(self, operation_id: str, runtime: RuntimeState) -> None:
        """Persist a RuntimeState snapshot."""

        if runtime.operation_id != operation_id:
            raise ValueError("runtime.operation_id must match operation_id")
        with self._lock:
            self._write_json(
                self._artifact_path(operation_id, self.RUNTIME_FILENAME),
                runtime.model_dump(mode="json"),
            )

    def save_snapshot(self, operation_id: str, cycle_index: int) -> Path:
        """Copy current operation graph files into a cycle snapshot directory."""

        if cycle_index < 0:
            raise ValueError("cycle_index must be greater than or equal to 0")

        with self._lock:
            operation_dir = self._operation_dir(operation_id)
            snapshot_dir = operation_dir / self.SNAPSHOTS_DIRNAME / f"cycle-{cycle_index:06d}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            manifest: dict[str, Any] = {
                "operation_id": operation_id,
                "cycle_index": cycle_index,
                "files": [],
            }
            filenames = [
                self.KG_FILENAME,
                self.AG_FILENAME,
                self.RUNTIME_FILENAME,
            ]
            for filename in filenames:
                source = operation_dir / filename
                if not source.exists():
                    continue
                copy2(source, snapshot_dir / filename)
                manifest["files"].append(filename)
            self._write_json(snapshot_dir / "manifest.json", manifest)
            return snapshot_dir

    def _operation_dir(self, operation_id: str) -> Path:
        if not operation_id or operation_id in {".", ".."}:
            raise ValueError("operation_id must be a non-empty path segment")
        if "/" in operation_id or "\\" in operation_id:
            raise ValueError("operation_id must not contain path separators")

        path = (self._root_dir / operation_id).resolve()
        try:
            path.relative_to(self._root_dir)
        except ValueError as exc:
            raise ValueError("operation_id resolves outside the graph memory store root") from exc
        return path

    def _artifact_path(self, operation_id: str, filename: str) -> Path:
        return self._operation_dir(operation_id) / filename

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(path)


__all__ = ["GraphMemoryStore"]
