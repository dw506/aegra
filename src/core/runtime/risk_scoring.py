"""Risk scoring for promoted vulnerability findings."""

from __future__ import annotations

from typing import Any

from src.core.models.finding import FindingSeverity, RiskScore


class RiskScorer:
    """Combine exploitability and local context into a 0-100 risk score."""

    @classmethod
    def score(
        cls,
        *,
        cvss: float | None = None,
        epss: float | None = None,
        kev: bool = False,
        public_exposed: bool = False,
        validated: bool = False,
        requires_auth: bool = False,
        critical_asset: bool = False,
        confidence: float = 0.5,
    ) -> RiskScore:
        parsed_cvss = cls._bounded_float(cvss, minimum=0.0, maximum=10.0)
        parsed_epss = cls._bounded_float(epss, minimum=0.0, maximum=1.0)
        parsed_confidence = cls._bounded_float(confidence, minimum=0.0, maximum=1.0) or 0.0

        base = 10.0
        rationale: list[str] = []
        if parsed_cvss is not None:
            base += parsed_cvss * 5.5
            rationale.append(f"cvss={parsed_cvss:g}")
        else:
            base += 22.0
            rationale.append("cvss=unknown")

        if parsed_epss is not None:
            base += parsed_epss * 18.0
            rationale.append(f"epss={parsed_epss:g}")
        if kev:
            base += 12.0
            rationale.append("known exploited vulnerability")
        if public_exposed:
            base += 8.0
            rationale.append("publicly exposed service")
        if validated:
            base += 10.0
            rationale.append("validated finding")
        if requires_auth:
            base -= 7.0
            rationale.append("requires authentication")
        if critical_asset:
            base += 10.0
            rationale.append("critical asset")

        confidence_adjusted = base * (0.75 + (parsed_confidence * 0.25))
        score = round(max(0.0, min(100.0, confidence_adjusted)), 1)
        return RiskScore(
            score=score,
            severity=cls.severity_for_score(score),
            factors={
                "cvss": parsed_cvss,
                "epss": parsed_epss,
                "kev": bool(kev),
                "public_exposed": bool(public_exposed),
                "validated": bool(validated),
                "requires_auth": bool(requires_auth),
                "critical_asset": bool(critical_asset),
                "confidence": parsed_confidence,
            },
            rationale=rationale,
        )

    @staticmethod
    def severity_for_score(score: float) -> FindingSeverity:
        if score >= 90.0:
            return "critical"
        if score >= 70.0:
            return "high"
        if score >= 40.0:
            return "medium"
        if score >= 10.0:
            return "low"
        return "informational"

    @staticmethod
    def _bounded_float(value: Any, *, minimum: float, maximum: float) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return max(minimum, min(maximum, parsed))


__all__ = ["RiskScorer"]
