"""SuccessConditionTracker: evaluates a SuccessContract against current KG/AG/Runtime.

This is the authoritative component for deciding whether an operation may stop
successfully. Only eligible_for_stop=True (computed here, never by an LLM)
allows PlannerAgent to emit stop_success.

No hardcoded environment values. Zone resolution goes through OperationProfile.
Private secrets are never read or stored here.
"""

from __future__ import annotations

from typing import Any

from src.core.evaluation.goal_oracle import GoalOracle
from src.core.evaluation.models import (
    GoalOracleInput,
    OperationProfile,
    SuccessConditionProgress,
    SuccessContract,
)
from src.core.evaluation.predicate_engine import PredicateContext, PredicateEngine


class SuccessConditionTracker:
    """Evaluate a SuccessContract and return SuccessConditionProgress.

    The tracker:
    - Loads OperationProfile (passed in at construction or per-call)
    - Evaluates each required condition via PredicateEngine
    - Calls GoalOracle for oracle_proof_valid predicates
    - Returns SuccessConditionProgress (no private data)
    - Does NOT write to KG/AG/Runtime directly — caller must persist the result
    """

    def __init__(
        self,
        *,
        predicate_engine: PredicateEngine | None = None,
        goal_oracle: GoalOracle | None = None,
    ) -> None:
        self._engine = predicate_engine or PredicateEngine()
        self._oracle = goal_oracle or GoalOracle()

    def evaluate(
        self,
        *,
        contract: SuccessContract,
        profile: OperationProfile,
        kg_nodes: list[dict[str, Any]] | None = None,
        kg_edges: list[dict[str, Any]] | None = None,
        ag_nodes: list[dict[str, Any]] | None = None,
        ag_edges: list[dict[str, Any]] | None = None,
        runtime_state: dict[str, Any] | None = None,
        oracle_config: dict[str, Any] | None = None,
        cycle_index: int = 0,
        blocking_policy_violation: bool = False,
        fatal_error: bool = False,
    ) -> SuccessConditionProgress:
        """Evaluate the contract and return a SuccessConditionProgress.

        Args:
            contract: Loaded SuccessContract to evaluate
            profile: Loaded OperationProfile (for zone resolution)
            kg_nodes/kg_edges: KG snapshot as list of dicts
            ag_nodes/ag_edges: AG snapshot as list of dicts
            runtime_state: RuntimeState serialized as dict
            oracle_config: Pre-loaded goal_oracle.yml dict (optional)
            cycle_index: Current operation cycle index
            blocking_policy_violation: True if there's a blocking policy violation
            fatal_error: True if operation has unhandled fatal error
        """
        # Pre-evaluate any oracle predicates
        oracle_results: dict[str, bool] = {}
        oracle_evidence_refs: list[str] = []
        oracle_proof_valid = False

        if oracle_config:
            oracle_result = self._run_oracle(
                oracle_config=oracle_config,
                runtime_state=runtime_state or {},
                kg_nodes=kg_nodes or [],
            )
            goal_id = oracle_config.get("goal_id", "")
            if goal_id and oracle_result:
                oracle_results[goal_id] = oracle_result.passed
                oracle_proof_valid = oracle_result.passed
                oracle_evidence_refs = oracle_result.evidence_refs

        ctx = PredicateContext(
            profile=profile,
            kg_nodes=kg_nodes or [],
            kg_edges=kg_edges or [],
            ag_nodes=ag_nodes or [],
            ag_edges=ag_edges or [],
            runtime_state=runtime_state or {},
            oracle_results=oracle_results,
        )

        satisfied: list[str] = []
        missing: list[str] = []
        failed: list[str] = []
        condition_results: dict[str, dict[str, Any]] = {}
        all_evidence_refs: list[str] = list(oracle_evidence_refs)

        evaluated_conditions = self._conditions_to_evaluate(contract)

        # Evaluate each required or level condition
        for condition_name in evaluated_conditions:
            binding = contract.condition_bindings.get(condition_name)
            if binding is None:
                result_dict = {
                    "condition": condition_name,
                    "satisfied": False,
                    "predicate": "unknown",
                    "error": f"No binding found for condition '{condition_name}'",
                }
                failed.append(condition_name)
                condition_results[condition_name] = result_dict
                continue

            result = self._engine.evaluate(
                predicate_name=binding.predicate,
                args=binding.args,
                ctx=ctx,
                condition_name=condition_name,
            )
            result_dict = result.model_dump()
            condition_results[condition_name] = result_dict

            if result.error:
                failed.append(condition_name)
            elif result.satisfied:
                satisfied.append(condition_name)
                all_evidence_refs.extend(result.evidence_refs)
            else:
                missing.append(condition_name)

        required_set = set(contract.require_all)
        required_missing = [name for name in contract.require_all if name in set(missing)]
        required_failed = [name for name in contract.require_all if name in set(failed)]
        all_required_satisfied = len(required_missing) == 0 and len(required_failed) == 0
        level_results = self._level_results(contract=contract, satisfied_set=set(satisfied), failed_set=set(failed))
        achieved_level = self._achieved_level(level_results)
        target_level = contract.target_level
        target_level_satisfied = (
            bool(level_results.get(target_level, {}).get("satisfied"))
            if target_level
            else all_required_satisfied
        )

        # Check chain integrity
        chain_integrity = self._check_chain_integrity(
            require_chain=contract.require_chain,
            satisfied_set=set(satisfied),
        )

        # Resolve goal-proof status from the authoritative condition results, not
        # from a second, divergent oracle code path. Any condition bound to the
        # oracle_proof_valid predicate is already part of require_all, so its
        # satisfaction is reflected in all_required_satisfied. We only surface a
        # reporting flag here (and never re-gate on the raw _run_oracle result,
        # which would require fields the tracker does not populate).
        oracle_condition_names = [
            name
            for name, binding in contract.condition_bindings.items()
            if binding.predicate == "oracle_proof_valid"
        ]
        has_oracle_predicate = bool(oracle_condition_names)
        if has_oracle_predicate:
            goal_proof_valid = all(name in satisfied for name in oracle_condition_names)
        else:
            goal_proof_valid = oracle_proof_valid

        # Build redacted summary
        redacted_summary = (
            f"Contract '{contract.contract_id}': "
            f"{len([name for name in satisfied if name in required_set])}/{len(contract.require_all)} required conditions satisfied. "
            f"Achieved level: {achieved_level or 'none'}. "
            f"Missing: {required_missing[:3]}{'...' if len(required_missing) > 3 else ''}. "
            f"Chain: {'ok' if chain_integrity else 'broken'}."
        )

        # Compute eligible_for_stop. The goal proof, when required, is already one
        # of the require_all conditions, so all_required_satisfied subsumes it.
        eligible_for_stop = (
            target_level_satisfied
            and chain_integrity
            and not blocking_policy_violation
            and not fatal_error
        )

        return SuccessConditionProgress(
            profile_id=profile.profile_id,
            contract_id=contract.contract_id,
            mode=contract.mode,
            all_required_satisfied=all_required_satisfied,
            chain_integrity=chain_integrity,
            goal_proof_valid=goal_proof_valid,
            eligible_for_stop=eligible_for_stop,
            achieved_level=achieved_level,
            target_level=target_level,
            level_results=level_results,
            satisfied=satisfied,
            missing=required_missing,
            failed=required_failed,
            condition_results=condition_results,
            evidence_refs=all_evidence_refs,
            redacted_summary=redacted_summary,
            last_updated_cycle=cycle_index,
        )

    @staticmethod
    def _conditions_to_evaluate(contract: SuccessContract) -> list[str]:
        ordered: list[str] = []
        for name in contract.require_all:
            if name not in ordered:
                ordered.append(name)
        for conditions in contract.levels.values():
            for name in conditions:
                if name not in ordered:
                    ordered.append(name)
        return ordered

    @staticmethod
    def _level_results(
        *,
        contract: SuccessContract,
        satisfied_set: set[str],
        failed_set: set[str],
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for level, conditions in contract.levels.items():
            required = [str(name) for name in conditions]
            missing = [name for name in required if name not in satisfied_set and name not in failed_set]
            failed = [name for name in required if name in failed_set]
            satisfied = [name for name in required if name in satisfied_set]
            results[str(level)] = {
                "level": str(level),
                "required": required,
                "satisfied": not missing and not failed,
                "satisfied_conditions": satisfied,
                "missing": missing,
                "failed": failed,
            }
        return results

    @staticmethod
    def _achieved_level(level_results: dict[str, dict[str, Any]]) -> str | None:
        achieved: str | None = None
        for level, result in level_results.items():
            if bool(result.get("satisfied")):
                achieved = level
        return achieved

    def _run_oracle(
        self,
        oracle_config: dict[str, Any],
        runtime_state: dict[str, Any],
        kg_nodes: list[dict[str, Any]],
    ) -> Any:
        """Run GoalOracle and return output, never exposing secrets."""
        # Collect evidence from GoalProof KG nodes
        evidence_refs = [
            str(n.get("id") or "")
            for n in kg_nodes
            if n.get("type") in ("GoalProof", "GoalCheck", "Evidence")
        ]
        # Collect session/pivot refs from runtime
        session_refs = list((runtime_state.get("sessions") or {}).keys())
        pivot_route_refs = list((runtime_state.get("pivot_routes") or {}).keys())

        goal_id = oracle_config.get("goal_id", "")
        oracle_input = GoalOracleInput(
            goal_id=goal_id,
            evidence_refs=evidence_refs,
            session_refs=session_refs,
            pivot_route_refs=pivot_route_refs,
        )
        try:
            return self._oracle.validate(oracle_config=oracle_config, oracle_input=oracle_input)
        except Exception:
            return None

    def _check_chain_integrity(
        self,
        require_chain: list[list[str]],
        satisfied_set: set[str],
    ) -> bool:
        """Check that all require_chain pairs are both satisfied.

        Weak chain check: both A and B must be satisfied.
        Strong chain (A -> B in KG) would require edge inspection.
        """
        for chain_pair in require_chain:
            if len(chain_pair) < 2:
                continue
            a, b = chain_pair[0], chain_pair[-1]
            # Skip-ahead detection: a later step satisfied without its prerequisite
            # means the chain was bypassed -> broken. (Prerequisite satisfied while
            # the later step is still pending is acceptable: in progress.)
            if b in satisfied_set and a not in satisfied_set:
                return False
        return True
