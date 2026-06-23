"""Single execution-agent facade for capability-scoped execution rounds."""

from __future__ import annotations

from typing import Any

from src.core.runtime.tool_trace_fact_extractor import ToolTraceFactExtractor
from src.core.runtime.txt_trace_logger import TxtTraceLogger
from src.core.stage.agents import ExecutionStageAgent
from src.core.stage.llm_driven_stage_agent import LLMDrivenStageAgent
from src.core.stage.models import (
    ExtractedFact,
    RoundDirective,
    RoundResult,
    StageExecutionRequest,
    StageResult,
    normalize_stage_name,
)


# Capability -> legacy stage vocabulary. The agent and result adapter still speak
# the 5 canonical stage names (see LLMDrivenStageAgent guard + StageResultAdapter
# ROLE_BY_STAGE), so the request still carries a stage_type. The round capability
# itself is now propagated explicitly on StageResult.capability (stamped below),
# NOT reconstructed from stage_type. ``lateral`` folds into pivot, ``evidence``
# into goal, matching the 5 stage names.
CAPABILITY_TO_STAGE: dict[str, str] = {
    "recon": "RECON_STAGE",
    "analysis": "VULN_ANALYSIS_STAGE",
    "exploit": "EXPLOIT_STAGE",
    "pivot": "ACCESS_PIVOT_STAGE",
    "lateral": "ACCESS_PIVOT_STAGE",
    "goal": "GOAL_STAGE",
    "evidence": "GOAL_STAGE",
}


class ExecutionAgent:
    """Run one planner directive as a single bounded execution round."""

    agent_name = "execution_agent"

    def __init__(
        self,
        agent: LLMDrivenStageAgent,
        *,
        trace_logger_factory: type[TxtTraceLogger] = TxtTraceLogger,
    ) -> None:
        self._agent = agent
        self._trace_logger_factory = trace_logger_factory

    @classmethod
    def from_clients(
        cls,
        *,
        llm_client: Any = None,
        mcp_client: Any = None,
        default_timeout_seconds: int = 120,
        trace_logger_factory: type[TxtTraceLogger] = TxtTraceLogger,
    ) -> "ExecutionAgent":
        """Build the default single executor and wrap it."""

        return cls(
            ExecutionStageAgent(
                llm_client=llm_client,
                mcp_client=mcp_client,
                default_timeout_seconds=default_timeout_seconds,
            ),
            trace_logger_factory=trace_logger_factory,
        )

    def capability_summary(self) -> list[dict[str, str]]:
        agent = self._agent
        return [
            {
                "agent_name": agent.agent_name,
                "stage_type": normalize_stage_name(agent.stage_type),
                "context_builder": getattr(agent, "context_builder_name", "stage_context_builder"),
            }
        ]

    def run(
        self,
        directive: RoundDirective,
        *,
        graph_summary: dict[str, Any] | None = None,
        graph_history: dict[str, Any] | None = None,
        runtime_context: dict[str, Any] | None = None,
        policy_context: dict[str, Any] | None = None,
        mcp_tool_catalog: dict[str, Any] | None = None,
        pivot_routes: list[dict[str, Any]] | None = None,
        sessions: list[dict[str, Any]] | None = None,
    ) -> RoundResult:
        """Execute one capability round through the single execution agent."""

        stage_type = normalize_stage_name(CAPABILITY_TO_STAGE[directive.capability])
        agent = self._agent
        # Pass the FULL catalog (every in-scope tool stays callable); the planner's
        # allowed_tools are attached only as a focus hint. The real authorization
        # boundary is scope policy, not this list.
        catalog = dict(mcp_tool_catalog or {})
        if directive.allowed_tools:
            catalog["recommended_tool_names"] = list(directive.allowed_tools)
        request = StageExecutionRequest(
            operation_id=directive.operation_id,
            cycle_index=directive.cycle_index,
            agent_name=agent.agent_name,
            stage_type=stage_type,
            objective=directive.objective,
            target_refs=list(directive.target_refs),
            required_context={
                **dict(directive.required_context),
                "round_capability": directive.capability,
                "tool_hints": list(directive.tool_hints),
            },
            success_criteria=[directive.success_hint] if directive.success_hint else [],
            risk_level=directive.risk_level,
            max_steps=directive.max_tools,
            graph_summary=dict(graph_summary or {}),
            graph_history=dict(graph_history or {}),
            runtime_context=dict(runtime_context or {}),
            policy_context=dict(policy_context or {}),
            mcp_tool_catalog=catalog,
            allowed_tool_names=list(directive.allowed_tools),
            pivot_routes=list(pivot_routes or []),
            sessions=list(sessions or []),
        )
        stage_result = agent.run(request)
        # Capability is authoritative from the directive; stamp it on the result so
        # the result tier reads it directly instead of reconstructing it from stage_type.
        stage_result.capability = directive.capability
        round_result = self._round_result(directive=directive, stage_result=stage_result)
        self._write_round_log(directive=directive, stage_result=stage_result, round_result=round_result)
        if round_result.log_ref:
            stage_result.runtime_hints = {
                **dict(stage_result.runtime_hints),
                "round_log_ref": round_result.log_ref,
                "cycle_index": directive.cycle_index,
                "capability": directive.capability,
            }
        return round_result

    @staticmethod
    def _round_result(*, directive: RoundDirective, stage_result: StageResult) -> RoundResult:
        facts: list[ExtractedFact] = []
        extractor = ToolTraceFactExtractor()
        for extraction in extractor.extract_all(stage_result.tool_trace):
            for fact in extraction.facts:
                facts.append(
                    ExtractedFact(
                        fact_id=f"tool-fact-{extraction.trace_id}-{fact.entity_type}-{fact.label[:40]}",
                        entity_type=fact.entity_type,
                        label=fact.label,
                        properties=dict(fact.properties),
                        source_tool=fact.source_tool,
                        trace_id=extraction.trace_id,
                        confidence=fact.confidence,
                    )
                )
        return RoundResult(
            operation_id=stage_result.operation_id,
            cycle_index=directive.cycle_index,
            capability=directive.capability,
            tool_traces=list(stage_result.tool_trace),
            extracted_facts=facts,
            raw_summary=stage_result.summary,
            objective_met=stage_result.status in {"success", "succeeded"},
            stage_result=stage_result,
        )

    def _write_round_log(
        self,
        *,
        directive: RoundDirective,
        stage_result: StageResult,
        round_result: RoundResult,
    ) -> None:
        logger = self._trace_logger_factory(
            directive.operation_id,
            log_dir="var/runtime",
            filename=f"round-{directive.cycle_index}.txt",
            operation_subdir=True,
        )
        logger.write_block(
            "ROUND_DIRECTIVE",
            "execution directive",
            {
                "cycle_index": directive.cycle_index,
                "capability": directive.capability,
                "objective": directive.objective,
                "target_refs": [item.model_dump(mode="json") for item in directive.target_refs],
                "allowed_tools": list(directive.allowed_tools),
                "max_tools": directive.max_tools,
            },
        )
        logger.write_block(
            "ROUND_RESULT",
            "execution result",
            {
                "status": stage_result.status,
                "objective_met": round_result.objective_met,
                "summary": stage_result.summary,
                "tool_traces": [
                    {
                        "trace_id": trace.trace_id,
                        "step": trace.step,
                        "tool_name": trace.tool_name,
                        "success": trace.success,
                        "raw_output_ref": trace.raw_output_ref,
                        "summary": trace.summary,
                    }
                    for trace in stage_result.tool_trace
                ],
                "extracted_fact_count": len(round_result.extracted_facts),
            },
        )
        round_result.log_ref = str(logger.path)


__all__ = [
    "CAPABILITY_TO_STAGE",
    "ExecutionAgent",
]
