"""GoalOracle: validates goal proofs without exposing private secrets to LLMs.

The oracle reads private marker/token values from environment variables,
validates submitted hashes, and returns an opaque signed proof token.

Private values (raw flags, tokens, markers) are NEVER:
- Written to KG, AG, Runtime, or audit logs
- Returned in any output model
- Passed to any LLM
- Logged anywhere

Only the opaque proof_token (HMAC-signed) and a redacted_summary are returned.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

from src.core.evaluation.models import GoalOracleInput, GoalOracleOutput


def _yaml_load(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import]

        with path.open() as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        import json as _json

        with path.open() as fh:
            return _json.load(fh)


class GoalOracle:
    """Validate goal proof submissions against profile-defined oracle config.

    The oracle is initialized with a path to goal_oracle.yml. Private
    marker/token values are read from environment variables named in the
    oracle config's `private` section.

    Args for validate():
        oracle_config_ref: path to goal_oracle.yml (or pre-loaded dict)
        oracle_input: GoalOracleInput with candidate proof data
    """

    def __init__(self, signing_key_env: str = "AEGRA_ORACLE_SIGNING_KEY") -> None:
        self._signing_key_env = signing_key_env

    def load_config(self, path: str | Path) -> dict[str, Any]:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path(os.getcwd()) / resolved
        return _yaml_load(resolved)

    def validate(
        self,
        oracle_config: dict[str, Any] | str | Path,
        oracle_input: GoalOracleInput,
    ) -> GoalOracleOutput:
        """Validate a goal proof submission.

        oracle_config may be a dict (pre-loaded) or a file path.
        Private secrets are never returned in the output.
        """
        if isinstance(oracle_config, (str, Path)):
            config = self.load_config(oracle_config)
        else:
            config = dict(oracle_config)

        goal_id = config.get("goal_id", oracle_input.goal_id)
        private_cfg = config.get("private", {})
        validation_cfg = config.get("validation", {})

        # Validate that private secrets are configured to NOT be exposed
        if private_cfg.get("expose_to_llm", False):
            raise RuntimeError("oracle config expose_to_llm must be false")

        satisfied_conditions: list[str] = []
        missing_categories: list[str] = []

        # Check structural requirements (route, session, pivot, evidence refs)
        if validation_cfg.get("require_access_path") and not oracle_input.access_path_refs:
            missing_categories.append("access_path")
        else:
            if oracle_input.access_path_refs:
                satisfied_conditions.append("access_path")

        if validation_cfg.get("require_session_ref") and not oracle_input.session_refs:
            missing_categories.append("session_ref")
        else:
            if oracle_input.session_refs:
                satisfied_conditions.append("session_ref")

        if validation_cfg.get("require_pivot_route_ref") and not oracle_input.pivot_route_refs:
            missing_categories.append("pivot_route_ref")
        else:
            if oracle_input.pivot_route_refs:
                satisfied_conditions.append("pivot_route_ref")

        if validation_cfg.get("require_evidence_ref") and not oracle_input.evidence_refs:
            missing_categories.append("evidence_ref")
        else:
            if oracle_input.evidence_refs:
                satisfied_conditions.append("evidence_ref")

        # Validate submitted marker hash against env-var secret
        marker_valid = self._validate_hash(
            submitted_hash=oracle_input.submitted_marker_hash,
            env_var=private_cfg.get("marker_env", ""),
            context=goal_id,
        )
        token_valid = self._validate_hash(
            submitted_hash=oracle_input.submitted_token_hash,
            env_var=private_cfg.get("token_env", ""),
            context=goal_id,
        )

        oracle_type = config.get("type", "marker_oracle")
        if oracle_type == "marker_oracle":
            proof_checks = [marker_valid or token_valid]
        else:
            proof_checks = [True]  # non-marker oracles pass on structural checks only

        structural_ok = len(missing_categories) == 0
        proof_ok = all(proof_checks)

        # Real hash verification requires the marker/token secret in env vars.
        marker_env_set = bool(os.environ.get(private_cfg.get("marker_env", ""), "").strip())
        token_env_set = bool(os.environ.get(private_cfg.get("token_env", ""), "").strip())
        env_configured = marker_env_set or token_env_set

        if not env_configured:
            # No secret to verify against. Only accept a structural-only proof when
            # the config explicitly opts in; otherwise a missing secret must fail
            # closed rather than silently rubber-stamp the goal.
            if validation_cfg.get("allow_structural_only"):
                proof_ok = True
                satisfied_conditions.append("oracle_structural_only")
            else:
                proof_ok = False
                missing_categories.append("oracle_secret_not_configured")

        passed = structural_ok and proof_ok

        proof_token = ""
        if passed:
            proof_token = self._sign_proof(goal_id=goal_id, evidence_refs=oracle_input.evidence_refs)

        return GoalOracleOutput(
            passed=passed,
            proof_token=proof_token,
            goal_id=goal_id,
            redacted_summary=(
                f"Goal '{goal_id}': proof {'accepted' if passed else 'incomplete'}. "
                f"Satisfied: {satisfied_conditions}. Missing: {missing_categories}."
            ),
            satisfied_conditions=satisfied_conditions,
            missing_categories=missing_categories,
            evidence_refs=oracle_input.evidence_refs,
            secret_redacted=True,
        )

    def _validate_hash(self, submitted_hash: str, env_var: str, context: str) -> bool:
        """Compare submitted_hash against HMAC of the env-var secret.

        Returns False (not an error) if env_var is empty or not set.
        The raw secret is never returned or logged.
        """
        if not submitted_hash or not env_var:
            return False
        secret = os.environ.get(env_var, "").strip()
        if not secret:
            return False
        # Expected hash: HMAC-SHA256(key=secret, msg=context+":"+secret)
        expected = hmac.new(
            secret.encode(),
            (context + ":" + secret).encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(submitted_hash, expected)

    def _sign_proof(self, goal_id: str, evidence_refs: list[str]) -> str:
        """Return an opaque HMAC-signed proof token.

        The token encodes goal_id + evidence count, signed with the signing key.
        It does NOT contain any raw secret or flag value.
        """
        signing_key = os.environ.get(self._signing_key_env, "aegra-default-dev-key")
        payload = json.dumps({"goal_id": goal_id, "evidence_count": len(evidence_refs)}, sort_keys=True)
        token = hmac.new(
            signing_key.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"proof:{goal_id}:{token[:16]}"
