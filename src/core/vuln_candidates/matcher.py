"""Fingerprint to vulnerability candidate matching."""

from __future__ import annotations

from collections.abc import Iterable

from src.core.models.fingerprint import ServiceFingerprint
from src.core.models.vulnerability_candidate import VulnerabilityCandidate
from src.core.vuln_candidates.rules import BUILTIN_CANDIDATE_RULES, CandidateRule


class CandidateMatcher:
    """Apply bounded rules to service fingerprints."""

    def __init__(self, rules: Iterable[CandidateRule] | None = None) -> None:
        self._rules = list(rules or BUILTIN_CANDIDATE_RULES)

    def match(self, fingerprint: ServiceFingerprint) -> list[VulnerabilityCandidate]:
        text = self._fingerprint_text(fingerprint)
        candidates: list[VulnerabilityCandidate] = []
        for rule in self._rules:
            if not self._matches(rule, text):
                continue
            indicators = sorted(
                {
                    token
                    for token in (*rule.required_all, *rule.required_any)
                    if token.lower() in text
                }
            )
            target_url = self._target_url(fingerprint)
            candidates.append(
                VulnerabilityCandidate.from_fingerprint(
                    fingerprint=fingerprint,
                    rule_id=rule.rule_id,
                    vulnerability_key=rule.vulnerability_key,
                    vulnerability_name=rule.vulnerability_name,
                    validator_id=rule.validator_id,
                    confidence=min(1.0, rule.confidence * max(0.5, fingerprint.confidence)),
                    reason=rule.reason,
                    cve=rule.cve,
                    cwe=rule.cwe,
                    advisory_refs=list(rule.advisory_refs),
                    indicators=indicators,
                    target_url=target_url,
                    metadata={
                        "host": fingerprint.host,
                        "port": fingerprint.port,
                        "protocol": fingerprint.protocol,
                        "service_name": fingerprint.service_name,
                        "fingerprint": fingerprint.model_dump(mode="json"),
                    },
                )
            )
        return candidates

    @staticmethod
    def _matches(rule: CandidateRule, text: str) -> bool:
        return all(token.lower() in text for token in rule.required_all) and any(
            token.lower() in text for token in rule.required_any
        )

    @staticmethod
    def _fingerprint_text(fingerprint: ServiceFingerprint) -> str:
        parts: list[str] = [
            fingerprint.service_id,
            fingerprint.protocol or "",
            fingerprint.service_name or "",
            fingerprint.product or "",
            fingerprint.version or "",
            fingerprint.http_title or "",
            " ".join(fingerprint.body_signals),
            " ".join(fingerprint.cpe),
            " ".join(f"{key}: {value}" for key, value in fingerprint.headers.items()),
            " ".join(fingerprint.stack.tags),
        ]
        for component in fingerprint.stack.components:
            parts.extend(
                [
                    component.name,
                    component.product or "",
                    component.version or "",
                    component.cpe or "",
                    " ".join(str(value) for value in component.evidence.values()),
                ]
            )
        return " ".join(item for item in parts if item).lower()

    @staticmethod
    def _target_url(fingerprint: ServiceFingerprint) -> str | None:
        if fingerprint.metadata.get("target_url"):
            return str(fingerprint.metadata["target_url"])
        if fingerprint.protocol in {"http", "https"} and fingerprint.host and fingerprint.port:
            return f"{fingerprint.protocol}://{fingerprint.host}:{fingerprint.port}/"
        if fingerprint.service_name and "http" in fingerprint.service_name.lower() and fingerprint.host and fingerprint.port:
            return f"http://{fingerprint.host}:{fingerprint.port}/"
        return None


__all__ = ["CandidateMatcher"]
