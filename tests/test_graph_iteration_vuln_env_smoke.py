from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.core.agents.agent_models import EvidenceRecord, ObservationRecord
from src.core.agents.agent_protocol import AgentContext, AgentInput, GraphRef, GraphScope
from src.core.feedback.evidence_extractor import EvidenceExtractor
from src.core.feedback.result_verifier import ResultVerifier
from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.kg_store import KnowledgeGraph
from src.core.graph.tg_builder import AttackGraphTaskBuilder, TaskGenerationRequest
from src.core.models.ag import ActionNodeType, StateNodeType
from src.core.models.kg import Host
from src.core.models.kg_enums import EntityStatus
from src.core.models.tg import TaskGraph, TaskType
from src.core.tools.recipe import ToolRecipeAdapter
from src.core.tools.runner import ToolRecipeRunner
from src.core.agents.state_writer import StateWriterAgent


class _SmokeHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = b"<html><head><title>Vuln Env Smoke</title></head><body>ok</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Server", "AegraSmokeHTTP/1.0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_graph_iteration_discovers_validates_and_plans_web_enumeration() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SmokeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        target = f"http://127.0.0.1:{port}/"
        service_id = f"127.0.0.1:{port}/tcp"

        kg = KnowledgeGraph()
        kg.add_node(
            Host(
                id="127.0.0.1",
                label="127.0.0.1",
                hostname="127.0.0.1",
                status=EntityStatus.OBSERVED,
                confidence=0.8,
            )
        )

        ag0 = AttackGraphProjector().project(kg)
        assert len(kg.get_hosts()) == 1
        assert ag0.find_states(StateNodeType.HOST_KNOWN)

        discovery_result = _run_http_probe(target)
        discovery_output = dict(discovery_result.output)
        discovery_output.pop("http_status", None)
        discovery_output.pop("title", None)
        for entity in discovery_output["entities"]:
            if entity.get("type") == "Service":
                entity["status"] = "observed"
                entity.pop("http_status", None)
                entity.pop("title", None)
        _write_probe_feedback(
            kg=kg,
            output=discovery_output,
            task_ref="cycle-1-http-probe",
            refs=[GraphRef(graph=GraphScope.KG, ref_id="127.0.0.1", ref_type="Host")],
        )

        ag1 = AttackGraphProjector().project(kg)
        tg1 = _build_tg_for_actions(ag1, ActionNodeType.VALIDATE_SERVICE)
        assert kg.get_node(service_id)
        assert kg.get_supporting_evidence(service_id)
        assert ag1.find_states(StateNodeType.SERVICE_KNOWN)
        assert ag1.find_actions(ActionNodeType.VALIDATE_SERVICE)
        assert any(node.task_type == TaskType.SERVICE_VALIDATION for node in tg1.list_nodes())

        validation_result = _run_http_probe(target)
        assert ResultVerifier().verify(validation_result, expected_target=target)["valid"] is True
        _write_probe_feedback(
            kg=kg,
            output=validation_result.output,
            task_ref="cycle-2-validate-http-service",
            refs=[GraphRef(graph=GraphScope.KG, ref_id=service_id, ref_type="Service")],
        )

        ag2 = AttackGraphProjector().project(kg)
        tg2 = _build_tg_for_actions(ag2, ActionNodeType.ENUMERATE_WEB_SURFACE)
        service = kg.get_node(service_id)
        assert service.properties["http_status"] == 200
        assert service.properties["title"] == "Vuln Env Smoke"
        assert ag2.find_states(StateNodeType.WEB_ATTACK_SURFACE)
        assert ag2.find_actions(ActionNodeType.ENUMERATE_WEB_SURFACE)
        assert any(node.task_type == TaskType.WEB_ENUMERATION for node in tg2.list_nodes())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _run_http_probe(target: str):
    recipe = ToolRecipeAdapter().build(
        {
            "task_type": "SERVICE_VALIDATION",
            "tool_hint": "http_probe",
            "target": target,
            "timeout_sec": 5,
        }
    )
    return ToolRecipeRunner().run(recipe)


def _write_probe_feedback(
    *,
    kg: KnowledgeGraph,
    output: dict,
    task_ref: str,
    refs: list[GraphRef],
) -> None:
    service_ref = _service_ref(output)
    record_refs = [*refs, service_ref] if service_ref is not None else refs
    standard_evidence = EvidenceExtractor().extract({"target": output.get("target"), "output": output, "success": output.get("success")})
    observation = ObservationRecord(
        source_agent="graph_iteration_smoke",
        summary=str(output["summary"]),
        confidence=float(output.get("confidence") or 0.8),
        refs=refs,
        payload={
            "observation_kind": "controlled_probe",
            "entities": list(output.get("entities", [])),
            "relations": list(output.get("relations", [])),
        },
    )
    evidence = EvidenceRecord(
        source_agent="graph_iteration_smoke",
        summary=str(output["summary"]),
        confidence=float(output.get("confidence") or 0.8),
        refs=record_refs,
        payload_ref=f"runtime://graph-iteration/{task_ref}",
        payload={
            "evidence_kind": "probe_result",
            "standard_evidence": standard_evidence,
            "target": output.get("target"),
        },
    )
    writer = StateWriterAgent()
    kg_ref = GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph")
    agent_input = AgentInput(
        graph_refs=[kg_ref, *refs],
        task_ref=task_ref,
        context=AgentContext(operation_id="op-graph-iteration-smoke"),
        raw_payload={
            "kg_version": kg.version,
            "observations": [observation.model_dump(mode="json")],
            "evidence": [evidence.model_dump(mode="json")],
        },
    )
    output_record = writer.execute(agent_input)
    apply_request = writer.build_store_apply_request(
        kg_ref=kg_ref,
        state_deltas=output_record.state_deltas,
        agent_input=agent_input,
        base_kg_version=kg.version,
    )
    writer.apply_to_store(store=kg, apply_request=apply_request)


def _service_ref(output: dict) -> GraphRef | None:
    for entity in output.get("entities", []):
        if isinstance(entity, dict) and entity.get("type") == "Service" and entity.get("id"):
            return GraphRef(graph=GraphScope.KG, ref_id=str(entity["id"]), ref_type="Service")
    return None


def _build_tg_for_actions(ag, action_type: ActionNodeType):
    actions = ag.find_actions(action_type)
    result = AttackGraphTaskBuilder().build_candidates(
        ag,
        TaskGenerationRequest(action_ids=[action.id for action in actions], include_evidence_tasks=False),
    )
    assert result.task_graph is not None
    return TaskGraph.from_dict(result.task_graph)
