"""Runtime checkpoint manager.

This module manages recovery anchors for Runtime State. Checkpoints are runtime
bookkeeping objects only; they are not a fact store and must not replace KG,
AG or TG.
"""

from __future__ import annotations

from typing import Any

from src.core.models.runtime import CheckpointRuntime, RuntimeState, utc_now


class RuntimeCheckpointManager:
    """Manage runtime checkpoints, lineage markers and recovery metadata."""

    def create_checkpoint(
        self,
        state: RuntimeState,
        checkpoint_id: str,
        created_after_tasks: list[str],
        kg_version: str | None = None,
        ag_version: str | None = None,
        tg_version: str | None = None,
        summary: str | None = None,
    ) -> CheckpointRuntime:
        """Create or replace one runtime checkpoint anchor."""

        checkpoint = CheckpointRuntime(
            checkpoint_id=checkpoint_id,
            created_at=utc_now(),
            created_after_tasks=list(created_after_tasks),
            kg_version=kg_version,
            ag_version=ag_version,
            tg_version=tg_version,
            summary=summary or "",
            metadata={},
        )
        state.checkpoints[checkpoint_id] = checkpoint
        recovery = self._recovery_metadata(state)
        recovery["last_stable_state"] = {
            "checkpoint_id": checkpoint_id,
            "kg_version": kg_version,
            "ag_version": ag_version,
            "tg_version": tg_version,
            "created_after_tasks": list(created_after_tasks),
            "created_at": checkpoint.created_at.isoformat(),
            "summary": checkpoint.summary,
        }
        state.last_updated = checkpoint.created_at
        return checkpoint

    def get_checkpoint(self, state: RuntimeState, checkpoint_id: str) -> CheckpointRuntime:
        """Return one checkpoint by ID."""

        try:
            return state.checkpoints[checkpoint_id]
        except KeyError as exc:
            raise ValueError(f"checkpoint '{checkpoint_id}' does not exist") from exc

    def list_checkpoints(self, state: RuntimeState) -> list[CheckpointRuntime]:
        """List checkpoints in creation order."""

        return sorted(
            state.checkpoints.values(),
            key=lambda item: (item.created_at, item.checkpoint_id),
        )

    def mark_task_lineage(
        self,
        state: RuntimeState,
        task_id: str,
        replaces_task_id: str | None = None,
        derived_from_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        """Record runtime lineage for one task without mutating TG structure."""

        recovery = self._recovery_metadata(state)
        lineage = recovery.setdefault("task_lineage", {})
        entry = lineage.setdefault(
            task_id,
            {
                "task_id": task_id,
                "replaces_task_id": None,
                "derived_from_checkpoint": None,
                "recorded_at": utc_now().isoformat(),
            },
        )
        if replaces_task_id is not None:
            entry["replaces_task_id"] = replaces_task_id
        if derived_from_checkpoint is not None:
            entry["derived_from_checkpoint"] = derived_from_checkpoint
        entry["recorded_at"] = utc_now().isoformat()
        state.last_updated = utc_now()
        return dict(entry)

    def add_replan_marker(self, state: RuntimeState, marker: dict[str, Any] | str) -> dict[str, Any]:
        """Append one replan marker to runtime recovery metadata."""

        recovery = self._recovery_metadata(state)
        markers = recovery.setdefault("replan_markers", [])
        if isinstance(marker, str):
            normalized = {
                "marker": marker,
                "created_at": utc_now().isoformat(),
            }
        else:
            normalized = dict(marker)
            normalized.setdefault("created_at", utc_now().isoformat())
        markers.append(normalized)
        state.last_updated = utc_now()
        return normalized

    def collect_recovery_context(self, state: RuntimeState, task_id: str) -> dict[str, Any]:
        """Collect the runtime recovery context needed after task failure."""

        recovery = self._recovery_metadata(state)
        task_runtime = state.execution.tasks.get(task_id)
        checkpoint = None
        if task_runtime is not None and task_runtime.checkpoint_ref is not None:
            checkpoint = state.checkpoints.get(task_runtime.checkpoint_ref)
        if checkpoint is None:
            checkpoint = self.latest_stable_checkpoint(state)

        lineage = recovery.get("task_lineage", {}).get(task_id)
        rollback_refs = recovery.get("rollback_refs", {}).get(task_id, [])
        failure_windows = recovery.get("failure_windows", {}).get(task_id, [])

        return {
            "task_id": task_id,
            "task_runtime": task_runtime.model_dump(mode="json") if task_runtime is not None else None,
            "lineage": dict(lineage) if isinstance(lineage, dict) else None,
            "checkpoint": checkpoint.model_dump(mode="json") if checkpoint is not None else None,
            "replan_markers": list(recovery.get("replan_markers", [])),
            "rollback_refs": list(rollback_refs),
            "failure_windows": list(failure_windows),
            "last_stable_state": dict(recovery.get("last_stable_state", {})),
        }

    def latest_stable_checkpoint(self, state: RuntimeState) -> CheckpointRuntime | None:
        """Return the most recent checkpoint, if any."""

        checkpoints = self.list_checkpoints(state)
        return checkpoints[-1] if checkpoints else None

    @staticmethod
    def _recovery_metadata(state: RuntimeState) -> dict[str, Any]:
        """Return the mutable runtime recovery metadata dictionary."""

        metadata = state.execution.metadata.setdefault("recovery", {})
        metadata.setdefault("task_lineage", {})
        metadata.setdefault("rollback_refs", {})
        metadata.setdefault("replan_markers", [])
        metadata.setdefault("failure_windows", {})
        metadata.setdefault("last_stable_state", {})
        return metadata


__all__ = ["RuntimeCheckpointManager"]

