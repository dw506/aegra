from __future__ import annotations

from src.core.stage.llm_stage_advisor import _coerce_graph_update_intents, _legacy_graph_intent_to_schema


def test_legacy_graph_intent_maps_web_endpoint_to_service() -> None:
    intent = _legacy_graph_intent_to_schema(
        {
            "intent": "add_endpoint",
            "match": "http://10.20.0.22:8080/",
            "attributes": {"url": "http://10.20.0.22:8080/"},
        }
    )

    assert intent["entity_type"] == "Service"
    assert intent["payload"]["entity_kind"] == "WebEndpoint"


def test_legacy_graph_intent_maps_fingerprint_to_observation() -> None:
    intent = _legacy_graph_intent_to_schema(
        {
            "intent": "add_fingerprint",
            "match": "fp-1",
            "attributes": {"product": "Apache Struts"},
        }
    )

    assert intent["entity_type"] == "Observation"
    assert intent["payload"]["entity_kind"] == "Fingerprint"


def test_graph_update_intent_normalizes_fingerprint_entity_type() -> None:
    intents = _coerce_graph_update_intents(
        [
            {
                "target_graph": "KG",
                "operation": "add",
                "entity_type": "Fingerprint",
                "entity_ref": "fp-1",
                "payload": {"product": "Apache Struts"},
            }
        ]
    )

    assert intents[0]["entity_type"] == "Observation"
    assert intents[0]["payload"]["entity_kind"] == "Fingerprint"


def test_graph_update_intent_normalizes_vulnerability_candidate_to_finding() -> None:
    intents = _coerce_graph_update_intents(
        [
            {
                "target_graph": "KG",
                "operation": "add",
                "entity_type": "VulnerabilityCandidate",
                "entity_ref": "candidate-1",
                "payload": {"candidate_type": "outdated_component"},
            }
        ]
    )

    assert intents[0]["entity_type"] == "Finding"
    assert intents[0]["payload"]["entity_kind"] == "VulnerabilityCandidate"
    assert intents[0]["payload"]["finding_kind"] == "VulnerabilityCandidate"


def test_graph_update_intent_normalizes_validation_plan_to_observation() -> None:
    intents = _coerce_graph_update_intents(
        [
            {
                "target_graph": "KG",
                "operation": "add",
                "entity_type": "ValidationPlan",
                "entity_ref": "plan-1",
                "payload": {"method": "safe_precheck"},
            }
        ]
    )

    assert intents[0]["entity_type"] == "Observation"
    assert intents[0]["payload"]["entity_kind"] == "ValidationPlan"
    assert intents[0]["payload"]["observation_kind"] == "ValidationPlan"
