"""First-pass verification for controlled tool results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VerificationResult(BaseModel):
    """ResultVerifier output contract."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    task_status: str
    new_information_found: bool
    target_matched: bool = False
    retry_needed: bool = False
    reason: str | None = None


class ResultVerifier:
    """Judge whether a tool run is usable enough for graph feedback."""

    def verify(self, result: Any, *, expected_target: str | None = None) -> dict[str, Any]:
        payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result or {})
        output = payload.get("output") if isinstance(payload.get("output"), dict) else payload
        success = bool(payload.get("success") or output.get("success") or output.get("reachable"))
        target = str(payload.get("target") or output.get("target") or "")
        target_matched = True
        if expected_target:
            target_matched = expected_target == target or expected_target in target or target in expected_target
        new_information = self._has_new_information(output)
        valid = success and target_matched
        retry_needed = not valid and str(payload.get("category") or "").lower() in {"timeout", "failed", "process_error"}
        verified = VerificationResult(
            valid=valid,
            task_status="succeeded" if valid else "failed",
            new_information_found=new_information,
            target_matched=target_matched,
            retry_needed=retry_needed,
            reason=None if valid else str(payload.get("error_message") or output.get("failure_reason") or "verification failed"),
        )
        return verified.model_dump(mode="json")

    @staticmethod
    def _has_new_information(output: dict[str, Any]) -> bool:
        if output.get("entities") or output.get("relations") or output.get("evidence"):
            return True
        return any(output.get(key) is not None for key in ("http_status", "banner", "title", "port", "service_name"))


__all__ = ["ResultVerifier", "VerificationResult"]

