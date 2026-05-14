"""Built-in vulnerability candidate rules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CandidateRule:
    rule_id: str
    vulnerability_key: str
    vulnerability_name: str
    validator_id: str
    confidence: float
    reason: str
    cve: str | None = None
    cwe: str | None = None
    advisory_refs: tuple[str, ...] = ()
    required_any: tuple[str, ...] = ()
    required_all: tuple[str, ...] = field(default_factory=tuple)


BUILTIN_CANDIDATE_RULES: tuple[CandidateRule, ...] = (
    CandidateRule(
        rule_id="apache-struts-s2-045",
        vulnerability_key="struts2-s2-045",
        vulnerability_name="Struts2 S2-045",
        validator_id="struts2-s2-045",
        confidence=0.78,
        reason="Apache Struts and S2-045/OGNL signals were observed in service fingerprint",
        cve="CVE-2017-5638",
        cwe="CWE-20",
        advisory_refs=("S2-045", "CVE-2017-5638"),
        required_all=("struts",),
        required_any=("s2-045", "ognl", "jakarta multipart", "content-type ognl"),
    ),
    CandidateRule(
        rule_id="redis-unauth-access",
        vulnerability_key="redis-unauth-access",
        vulnerability_name="Redis unauthenticated access",
        validator_id="redis-unauth-access",
        confidence=0.7,
        reason="Redis service fingerprint includes unauthenticated access signals",
        cwe="CWE-306",
        advisory_refs=("redis-unauth-access",),
        required_all=("redis",),
        required_any=("noauth", "unauthenticated", "no authentication", "redis_version"),
    ),
    CandidateRule(
        rule_id="tomcat-manager-exposed",
        vulnerability_key="tomcat-manager-exposed",
        vulnerability_name="Tomcat Manager exposed",
        validator_id="http-fingerprint",
        confidence=0.68,
        reason="HTTP fingerprint exposes Tomcat Manager surface",
        cwe="CWE-200",
        advisory_refs=("tomcat-manager-exposed",),
        required_all=("tomcat",),
        required_any=("manager/html", "tomcat manager", "apache-coyote"),
    ),
    CandidateRule(
        rule_id="spring-actuator-exposure",
        vulnerability_key="spring-actuator-exposure",
        vulnerability_name="Spring actuator exposure",
        validator_id="http-fingerprint",
        confidence=0.66,
        reason="HTTP fingerprint exposes Spring actuator surface",
        cwe="CWE-200",
        advisory_refs=("spring-actuator-exposure",),
        required_all=("spring",),
        required_any=("actuator", "/actuator", "spring boot"),
    ),
)


__all__ = ["BUILTIN_CANDIDATE_RULES", "CandidateRule"]
