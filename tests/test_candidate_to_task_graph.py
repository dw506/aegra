from __future__ import annotations

from src.core.graph.tg_builder import TaskGenerationRequest, TaskGraphBuilder
from src.core.agents.task_builder import TaskBuildRequest, TaskBuilderAgent
from src.core.models.ag import GraphRef
from src.core.models.fingerprint import ServiceFingerprint
from src.core.models.tg import TaskType
from src.core.models.vulnerability_candidate import VulnerabilityCandidate


def test_vulnerability_candidate_converts_to_vulnerability_validation_task() -> None:
    fingerprint = ServiceFingerprint(
        service_id="127.0.0.1:8080/tcp",
        host="127.0.0.1",
        port=8080,
        protocol="http",
        service_name="http",
        product="Apache Struts",
        body_signals=["S2-045", "OGNL"],
        confidence=0.9,
    )
    candidate = VulnerabilityCandidate.from_fingerprint(
        fingerprint=fingerprint,
        rule_id="apache-struts-s2-045",
        vulnerability_key="struts2-s2-045",
        vulnerability_name="Struts2 S2-045",
        validator_id="struts2-s2-045",
        confidence=0.8,
        reason="Apache Struts S2-045 signal observed",
        cve="CVE-2017-5638",
        advisory_refs=["S2-045"],
        target_url="http://127.0.0.1:8080/",
        indicators=["struts", "s2-045"],
    )

    task_candidate = candidate.to_task_candidate(
        service_ref=GraphRef(graph="kg", ref_id="127.0.0.1:8080/tcp", ref_type="Service")
    )
    graph_result = TaskGraphBuilder().build_from_candidates(
        TaskGenerationRequest(candidates=[task_candidate], include_evidence_tasks=False)
    )
    task = next(node for node in graph_result.task_graph["nodes"] if node["kind"] == "task")

    assert task_candidate.task_type == TaskType.VULNERABILITY_VALIDATION
    assert task["task_type"] == "VULNERABILITY_VALIDATION"
    assert task["input_bindings"]["validator_id"] == "struts2-s2-045"
    assert task["input_bindings"]["vulnerability_id"] == "vuln::struts2-s2-045::127.0.0.1:8080/tcp"
    assert task["input_bindings"]["candidate"]["confidence"] == 0.8
    assert task["input_bindings"]["candidate"]["reason"] == "Apache Struts S2-045 signal observed"


def test_task_builder_resolves_vulnerability_candidates_to_task_candidates() -> None:
    candidate = VulnerabilityCandidate(
        candidate_id="cand-redis",
        vulnerability_id="vuln::redis-unauth-access::svc-redis",
        vulnerability_name="Redis unauthenticated access",
        service_id="svc-redis",
        validator_id="redis-unauth-access",
        confidence=0.7,
        reason="Redis unauthenticated access signal observed",
        matched_rule_id="redis-unauth-access",
    )
    request = TaskBuildRequest(
        decision={"id": "decision-1", "source_agent": "planner", "summary": "build vuln candidate", "payload": {}},
        vulnerability_candidates=[candidate],
    )

    task_candidates = TaskBuilderAgent()._resolve_task_candidates(request)

    assert task_candidates[0].task_type == TaskType.VULNERABILITY_VALIDATION
    assert task_candidates[0].input_bindings["candidate"]["candidate_id"] == "cand-redis"
