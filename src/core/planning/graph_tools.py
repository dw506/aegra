"""Typed graph tools used by the P3 agentic planner."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph, stable_node_id
from src.core.models.kg_enums import EdgeType, NodeType
from src.core.models.runtime import RuntimeState, utc_now


class RecordFindingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    host_ref: str
    title: str = Field(min_length=1)
    severity: str = "info"
    summary: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class LinkEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    node_ref: str
    evidence_ref: str


class RecordAttackStepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    capability: str
    target_ref: str | None = None
    status: str
    summary: str
    evidence_refs: list[str] = Field(default_factory=list)
    kg_node_refs: list[str] = Field(default_factory=list)


class PlannerGraphTools:
    """Small typed read/write tool surface for planner decisions."""

    def __init__(
        self,
        *,
        operation_id: str,
        cycle_index: int,
        kg: KnowledgeGraph,
        ag: AttackGraph,
        runtime_state: RuntimeState,
    ) -> None:
        self.operation_id = operation_id
        self.cycle_index = cycle_index
        self.kg = kg
        self.ag = ag
        self.runtime_state = runtime_state

    def build_min_summary(self) -> dict[str, Any]:
        progress = dict(self.runtime_state.execution.metadata.get("success_condition_progress") or {})
        attack_steps = [
            node.model_dump(mode="json")
            for node in self.ag.find_process_nodes()
            if str(getattr(node, "node_type", "")) == "AttackProcessNodeType.ATTACK_STEP"
            or str(getattr(node, "node_type", "")) == "ATTACK_STEP"
            or getattr(getattr(node, "node_type", None), "value", None) == "ATTACK_STEP"
        ]
        attack_steps = sorted(attack_steps, key=lambda item: int(item.get("cycle_index") or 0))[-3:]
        return {
            "operation_id": self.operation_id,
            "cycle_index": self.cycle_index,
            "kg_node_count": len(self.kg.list_nodes()),
            "kg_edge_count": len(self.kg.list_edges()),
            "ag_step_count": len(attack_steps),
            "eligible_for_stop": bool(progress.get("eligible_for_stop")),
            "achieved_level": progress.get("achieved_level"),
            "missing": list(progress.get("missing") or []),
            "satisfied": list(progress.get("satisfied") or []),
            "recent_attack_steps": attack_steps,
        }

    @staticmethod
    def tool_manifest() -> dict[str, list[str]]:
        # Push model: the planner advisor is a single-shot LLM call with no
        # tool-call loop, so the LLM cannot invoke read tools mid-decision. The
        # read methods on this class are used by the orchestrator to BUILD the
        # precomputed context (e.g. build_min_summary); only the write tools are
        # advertised to the LLM, dispatched via apply_tool_calls after the turn.
        return {
            "write": ["record_finding", "record_attack_step", "link_evidence"],
        }

    def record_finding(self, request: RecordFindingRequest | dict[str, Any]) -> dict[str, Any]:
        payload = request if isinstance(request, RecordFindingRequest) else RecordFindingRequest.model_validate(request)
        finding_id = stable_node_id(
            "finding",
            {
                "operation_id": self.operation_id,
                "host_ref": payload.host_ref,
                "title": payload.title,
                "cycle_index": self.cycle_index,
            },
        )
        state_delta = {
            "id": f"planner-finding::{finding_id}",
            "delta_type": "upsert_entity",
            "target_ref": {"graph": "kg", "ref_id": finding_id, "ref_type": NodeType.FINDING.value},
            "payload": {"patch_kind": "entity", "fact_kind": "planner_finding"},
            "patch": {
                "entity_kind": "node",
                "entity_id": finding_id,
                "entity_type": NodeType.FINDING.value,
                "label": payload.title,
                "attributes": {
                    "title": payload.title,
                    "summary": payload.summary,
                    "severity": payload.severity,
                    "affected_asset_refs": [payload.host_ref],
                    "evidence_refs": list(payload.evidence_refs),
                    "source_task_id": f"planner::{self.operation_id}::{self.cycle_index}",
                    "confidence": 0.8,
                },
            },
        }
        result = self.kg.apply_patch_batch(
            {
                "operation_id": self.operation_id,
                "base_kg_version": self.kg.version,
                "state_deltas": [state_delta],
                "metadata": {"source": "planner_graph_tool", "cycle_index": self.cycle_index},
            }
        )
        self._record_tool_audit("record_finding", payload.model_dump(mode="json"), result)
        return result

    def link_evidence(self, request: LinkEvidenceRequest | dict[str, Any]) -> dict[str, Any]:
        payload = request if isinstance(request, LinkEvidenceRequest) else LinkEvidenceRequest.model_validate(request)
        relation_id = f"{EdgeType.SUPPORTED_BY.value.lower()}::{payload.node_ref}::{payload.evidence_ref}"
        state_delta = {
            "id": f"planner-link::{relation_id}",
            "delta_type": "upsert_relation",
            "target_ref": {"graph": "kg", "ref_id": relation_id, "ref_type": EdgeType.SUPPORTED_BY.value},
            "payload": {"patch_kind": "relation", "fact_kind": "planner_evidence_link"},
            "patch": {
                "entity_kind": "edge",
                "relation_id": relation_id,
                "relation_type": EdgeType.SUPPORTED_BY.value,
                "source": payload.node_ref,
                "target": payload.evidence_ref,
                "label": "supported by",
                "attributes": {"source_task_id": f"planner::{self.operation_id}::{self.cycle_index}"},
            },
        }
        result = self.kg.apply_patch_batch(
            {
                "operation_id": self.operation_id,
                "base_kg_version": self.kg.version,
                "state_deltas": [state_delta],
                "metadata": {"source": "planner_graph_tool", "cycle_index": self.cycle_index},
            }
        )
        self._record_tool_audit("link_evidence", payload.model_dump(mode="json"), result)
        return result

    def record_attack_step(self, request: RecordAttackStepRequest | dict[str, Any]) -> dict[str, Any]:
        payload = request if isinstance(request, RecordAttackStepRequest) else RecordAttackStepRequest.model_validate(request)
        record = {
            **payload.model_dump(mode="json"),
            "cycle_index": self.cycle_index,
            "recorded_at": utc_now().isoformat(),
            "note": "ResultApplier owns ATTACK_STEP node creation; planner tool records semantic intent only.",
        }
        self.runtime_state.execution.metadata.setdefault("planner_attack_step_records", []).append(record)
        self._record_tool_audit("record_attack_step", payload.model_dump(mode="json"), {"recorded": True})
        return {"recorded": True, "record": record}

    def apply_tool_calls(self, tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for call in tool_calls or []:
            if not isinstance(call, dict):
                continue
            name = str(call.get("tool") or call.get("name") or "")
            args = call.get("arguments") if isinstance(call.get("arguments"), dict) else call.get("args")
            args = args if isinstance(args, dict) else {}
            if name == "record_finding":
                results.append({"tool": name, "result": self.record_finding(args)})
            elif name == "link_evidence":
                results.append({"tool": name, "result": self.link_evidence(args)})
            elif name == "record_attack_step":
                results.append({"tool": name, "result": self.record_attack_step(args)})
            else:
                results.append({"tool": name, "error": "unknown planner graph tool"})
        return results

    def _record_tool_audit(self, tool: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        self.runtime_state.execution.metadata.setdefault("planner_graph_tool_calls", []).append(
            {
                "tool": tool,
                "arguments": arguments,
                "result": result,
                "cycle_index": self.cycle_index,
                "created_at": utc_now().isoformat(),
            }
        )


__all__ = [
    "LinkEvidenceRequest",
    "PlannerGraphTools",
    "RecordAttackStepRequest",
    "RecordFindingRequest",
]
