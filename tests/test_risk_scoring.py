from __future__ import annotations

from src.core.runtime.risk_scoring import RiskScorer


def test_risk_score_combines_validation_and_exposure_factors() -> None:
    score = RiskScorer.score(
        cvss=9.8,
        epss=0.9,
        kev=True,
        public_exposed=True,
        validated=True,
        requires_auth=False,
        critical_asset=True,
        confidence=0.95,
    )

    assert score.score >= 90
    assert score.severity == "critical"
    assert score.factors["validated"] is True
    assert "validated finding" in score.rationale


def test_risk_score_reduces_for_auth_and_low_confidence() -> None:
    unauthenticated = RiskScorer.score(cvss=7.5, validated=True, requires_auth=False, confidence=0.9)
    authenticated = RiskScorer.score(cvss=7.5, validated=True, requires_auth=True, confidence=0.4)

    assert authenticated.score < unauthenticated.score
    assert authenticated.factors["requires_auth"] is True
