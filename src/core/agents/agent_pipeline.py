"""Minimal agent orchestration pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_protocol import (
    AgentContext,
    AgentExecutionResult,
    AgentInput,
    AgentKind,
    AgentOutput,
    BaseAgent,
    GraphRef,
)
from src.core.agents.kg_events import KGDeltaEvent, KGDeltaEventType, KGEventBatch
from src.core.agents.registry import AgentNotFoundError, AgentRegistry
from src.core.models.events import AgentTaskResult
from src.core.runtime.worker_result_adapter import WorkerResultAdapter


class PipelineStepResult(BaseModel):
    """One executed pipeline step."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    step_name: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    agent_kind: AgentKind
    success: bool
    agent_input: AgentInput
    agent_output: AgentOutput
    started_at: datetime
    finished_at: datetime
    duration_ms: int = Field(ge=0)

    @classmethod
    def from_execution(
        cls,
        *,
        step_name: str,
        agent_input: AgentInput,
        execution: AgentExecutionResult,
    ) -> "PipelineStepResult":
        return cls(
            step_name=step_name,
            agent_name=execution.agent_name,
            agent_kind=execution.agent_kind,
            success=execution.success,
            agent_input=agent_input,
            agent_output=execution.output,
            started_at=execution.started_at,
            finished_at=execution.finished_at,
            duration_ms=execution.duration_ms,
        )


class PipelineCycleResult(BaseModel):
    """One pipeline cycle result."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    cycle_name: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    success: bool
    steps: list[PipelineStepResult] = Field(default_factory=list)
    final_output: AgentOutput = Field(default_factory=AgentOutput)
    logs: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AgentPipeline:
    """Orchestrates agents via AgentInput/AgentOutput only.

    中文注释：
    pipeline 内部仍然使用通用 `AgentOutput` 传递 step 结果，
    但所有 worker 落地结果都会通过 `AgentResultAdapter` 统一收敛成
    `AgentTaskResult`，供 orchestrator / result applier 等 phase-two 组件消费。
    """

    def __init__(
        self,
        *,
        registry: AgentRegistry | None = None,
        agents: Sequence[BaseAgent] | None = None,
        event_sink: Callable[[list[dict[str, Any]]], None] | None = None,
        state_delta_sink: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> None:
        self.registry = registry or AgentRegistry()
        self._event_sink = event_sink
        self._state_delta_sink = state_delta_sink
        for agent in agents or []:
            try:
                self.registry.register(agent)
            except Exception:
                pass

    def resolve_agent(
        self,
        *,
        agent_name: str | None = None,
        agent_kind: AgentKind | None = None,
    ) -> BaseAgent:
        if agent_name:
            return self.registry.get(agent_name)
        if agent_kind is None:
            raise ValueError("resolve_agent requires agent_name or agent_kind")
        agents = self.registry.list_by_kind(agent_kind)
        if not agents:
            raise AgentNotFoundError(f"no agent registered for kind '{agent_kind.value}'")
        return agents[0]

    def run_planning_cycle(
        self,
        *,
        operation_id: str,
        graph_refs: Sequence[GraphRef],
        planner_payload: dict[str, Any],
        planner_agent: str | None = None,
        context: AgentContext | dict[str, Any] | None = None,
    ) -> PipelineCycleResult:
        steps: list[PipelineStepResult] = []
        planner_input = self._build_input(operation_id, graph_refs, planner_payload, context=context)
        planner_step = self._dispatch("planner", planner_input, agent_name=planner_agent, agent_kind=AgentKind.PLANNER)
        steps.append(planner_step)
        self._forward(planner_step)
        return self._cycle("planning", operation_id, steps)

    def worker_task_results(
        self,
        execution: PipelineCycleResult | Sequence[PipelineStepResult],
    ) -> list[AgentTaskResult]:
        """把 execution cycle 中的 worker step 统一转换为 canonical result。"""

        steps = execution.steps if isinstance(execution, PipelineCycleResult) else list(execution)
        results: list[AgentTaskResult] = []
        for step in steps:
            if step.agent_kind != AgentKind.WORKER:
                continue
            results.append(
                WorkerResultAdapter.to_task_result(
                    step.agent_output,
                    agent_input=step.agent_input,
                    agent_name=step.agent_name,
                    agent_kind=step.agent_kind,
                )
            )
        return results

    def run_feedback_cycle(
        self,
        *,
        operation_id: str,
        graph_refs: Sequence[GraphRef],
        worker_steps: Sequence[PipelineStepResult] | None = None,
        feedback_payload: dict[str, Any] | None = None,
        perception_agent: str | None = None,
        state_writer_agent: str | None = None,
        graph_projection_agent: str | None = None,
        critic_agent: str | None = None,
        context: AgentContext | dict[str, Any] | None = None,
    ) -> PipelineCycleResult:
        payload = dict(feedback_payload or {})
        steps: list[PipelineStepResult] = []
        worker_like_steps = list(worker_steps or [])
        for index, worker_step in enumerate(worker_like_steps, start=1):
            outcome = worker_step.agent_output.outcomes[0] if worker_step.agent_output.outcomes else None
            raw_result = worker_step.agent_output.evidence[0] if worker_step.agent_output.evidence else None
            if not outcome:
                continue
            perception_input = self._build_input(
                operation_id,
                worker_step.agent_input.graph_refs or graph_refs,
                {"outcome": outcome, "raw_result": raw_result},
                task_ref=worker_step.agent_input.task_ref,
                decision_ref=worker_step.agent_input.decision_ref,
                context=context or worker_step.agent_input.context,
            )
            step = self._dispatch(
                f"perception[{index}]",
                perception_input,
                agent_name=perception_agent,
                agent_kind=AgentKind.PERCEPTION,
            )
            steps.append(step)
            self._forward(step)

        observations = self._collect(steps, "observations")
        evidence = self._collect(steps, "evidence")
        if observations or evidence:
            writer_input = self._build_input(
                operation_id,
                graph_refs,
                {"observations": observations, "evidences": evidence, "kg_ref": payload.get("kg_ref")},
                context=context,
            )
            writer_step = self._dispatch(
                "state_writer",
                writer_input,
                agent_name=state_writer_agent,
                agent_kind=AgentKind.STATE_WRITER,
            )
            steps.append(writer_step)
            self._forward(writer_step)

            projection_input = self._build_input(
                operation_id,
                graph_refs,
                {
                    "kg_event_batch": self._kg_batch(writer_step).model_dump(mode="json"),
                    "goal_context": self._mapping(payload.get("goal_context")),
                    "policy_context": self._mapping(payload.get("policy_context")),
                },
                context=context,
            )
            projection_step = self._dispatch(
                "graph_projection",
                projection_input,
                agent_name=graph_projection_agent,
                agent_kind=AgentKind.GRAPH_PROJECTION,
            )
            steps.append(projection_step)
            self._forward(projection_step)

        if critic_agent is not None:
            critic_input = self._build_input(
                operation_id,
                graph_refs,
                {
                    "runtime_state": payload.get("runtime_state"),
                    "runtime_summary": payload.get("runtime_summary"),
                    "critic_context": payload.get("critic_context"),
                    "recent_outcomes": payload.get("recent_outcomes") or self._critic_recent_outcomes(worker_like_steps),
                },
                context=context,
            )
            critic_step = self._dispatch(
                "critic",
                critic_input,
                agent_name=critic_agent,
                agent_kind=AgentKind.CRITIC,
            )
            steps.append(critic_step)
            self._forward(critic_step)
        return self._cycle("feedback", operation_id, steps)

    def run_supervisor_cycle(
        self,
        *,
        operation_id: str,
        graph_refs: Sequence[GraphRef],
        supervisor_payload: dict[str, Any],
        supervisor_agent: str | None = None,
        context: AgentContext | dict[str, Any] | None = None,
    ) -> PipelineCycleResult:
        """Run the optional supervisor advisory cycle.

        This cycle is intentionally independent from the orchestrator main loop;
        callers may record the advice, but it does not dispatch workers or mutate graphs.
        """

        supervisor_input = self._build_input(operation_id, graph_refs, supervisor_payload, context=context)
        supervisor_step = self._dispatch(
            "supervisor",
            supervisor_input,
            agent_name=supervisor_agent,
            agent_kind=AgentKind.SUPERVISOR,
        )
        self._forward(supervisor_step)
        return self._cycle("supervisor", operation_id, [supervisor_step])

    def _dispatch(
        self,
        step_name: str,
        agent_input: AgentInput,
        *,
        agent_name: str | None = None,
        agent_kind: AgentKind | None = None,
    ) -> PipelineStepResult:
        agent = self.resolve_agent(agent_name=agent_name, agent_kind=agent_kind)
        execution = self.registry.dispatch(agent.name, agent_input)
        return PipelineStepResult.from_execution(
            step_name=step_name,
            agent_input=agent_input,
            execution=execution,
        )

    def _build_input(
        self,
        operation_id: str,
        graph_refs: Sequence[GraphRef],
        raw_payload: dict[str, Any] | None = None,
        *,
        task_ref: str | None = None,
        decision_ref: str | None = None,
        context: AgentContext | dict[str, Any] | None = None,
    ) -> AgentInput:
        return AgentInput(
            graph_refs=self._dedupe_refs(graph_refs),
            task_ref=task_ref,
            decision_ref=decision_ref,
            context=self._context(operation_id, context),
            raw_payload=dict(raw_payload or {}),
        )

    def _kg_batch(self, state_writer_step: PipelineStepResult) -> KGEventBatch:
        events: list[KGDeltaEvent] = []
        for delta in state_writer_step.agent_output.state_deltas:
            target_ref = GraphRef.model_validate(delta.get("target_ref"))
            patch = self._mapping(delta.get("patch"))
            delta_type = str(delta.get("delta_type") or "")
            events.append(
                KGDeltaEvent(
                    event_type=self._kg_event_type(delta_type, patch),
                    source_agent=state_writer_step.agent_name,
                    target_ref=target_ref,
                    patch=patch,
                    metadata={"state_delta_id": delta.get("id")},
                )
            )
        return KGEventBatch.from_events(events, metadata={"source_step": state_writer_step.step_name})

    def _forward(self, step: PipelineStepResult) -> None:
        if step.agent_output.emitted_events and self._event_sink is not None:
            self._event_sink(list(step.agent_output.emitted_events))
        if step.agent_output.state_deltas and self._state_delta_sink is not None:
            self._state_delta_sink(list(step.agent_output.state_deltas))

    def _cycle(
        self,
        cycle_name: str,
        operation_id: str,
        steps: Sequence[PipelineStepResult],
    ) -> PipelineCycleResult:
        final_output = self._merge(step.agent_output for step in steps)
        errors = [error for step in steps for error in step.agent_output.errors]
        return PipelineCycleResult(
            cycle_name=cycle_name,
            operation_id=operation_id,
            success=all(step.success for step in steps) and not errors,
            steps=list(steps),
            final_output=final_output,
            logs=[f"{cycle_name} cycle executed {len(steps)} step(s)"],
            errors=errors,
        )

    @staticmethod
    def _merge(outputs: Sequence[AgentOutput] | Any) -> AgentOutput:
        merged = AgentOutput()
        for output in outputs:
            merged.observations.extend(output.observations)
            merged.evidence.extend(output.evidence)
            merged.outcomes.extend(output.outcomes)
            merged.decisions.extend(output.decisions)
            merged.state_deltas.extend(output.state_deltas)
            merged.replan_requests.extend(output.replan_requests)
            merged.emitted_events.extend(output.emitted_events)
            merged.logs.extend(output.logs)
            merged.errors.extend(output.errors)
        return merged

    @staticmethod
    def _context(operation_id: str, context: AgentContext | dict[str, Any] | None) -> AgentContext:
        if isinstance(context, AgentContext):
            return context
        payload = dict(context or {})
        payload.setdefault("operation_id", operation_id)
        return AgentContext.model_validate(payload)

    @staticmethod
    def _extract_candidate_actions(decision: dict[str, Any]) -> list[str]:
        candidate = AgentPipeline._mapping(decision.get("payload")).get("planning_candidate")
        if isinstance(candidate, dict):
            return [str(item) for item in candidate.get("action_ids", [])]
        return []

    @staticmethod
    def _extract_task_candidates(decision: dict[str, Any]) -> list[dict[str, Any]]:
        candidate = AgentPipeline._mapping(decision.get("payload")).get("planning_candidate")
        if isinstance(candidate, dict):
            return [dict(item) for item in candidate.get("task_candidates", []) if isinstance(item, dict)]
        return []

    @staticmethod
    def _refs(value: Any) -> list[GraphRef]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        refs: list[GraphRef] = []
        for item in items:
            try:
                refs.append(item if isinstance(item, GraphRef) else GraphRef.model_validate(item))
            except Exception:
                continue
        return refs

    @staticmethod
    def _collect(steps: Sequence[PipelineStepResult], field_name: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for step in steps:
            items.extend(getattr(step.agent_output, field_name, []))
        return items

    @staticmethod
    def _kg_event_type(delta_type: str, patch: dict[str, Any]) -> KGDeltaEventType:
        operation = str(patch.get("operation") or "upsert").lower()
        if delta_type == "upsert_relation":
            return KGDeltaEventType.RELATION_UPDATED if operation == "update" else KGDeltaEventType.RELATION_ADDED
        if delta_type == "upsert_entity":
            return KGDeltaEventType.ENTITY_UPDATED if operation == "update" else KGDeltaEventType.ENTITY_ADDED
        if "confidence" in patch:
            return KGDeltaEventType.CONFIDENCE_CHANGED
        return KGDeltaEventType.ENTITY_UPDATED

    @staticmethod
    def _dedupe_refs(refs: Sequence[GraphRef]) -> list[GraphRef]:
        result: list[GraphRef] = []
        seen: set[tuple[str, str, str | None]] = set()
        for ref in refs:
            normalized = AgentPipeline._protocol_ref(ref)
            key = (normalized.graph.value, normalized.ref_id, normalized.ref_type)
            if key in seen:
                continue
            seen.add(key)
            result.append(normalized)
        return result

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _first_text(*values: Any) -> str | None:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    @staticmethod
    def _critic_recent_outcomes(steps: Sequence[PipelineStepResult]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for step in steps:
            for outcome in step.agent_output.outcomes:
                payload_ref = str(
                    outcome.get("raw_result_ref")
                    or f"runtime://outcomes/{outcome.get('task_id', 'unknown')}/{outcome.get('id', 'latest')}"
                )
                items.append(
                    {
                        "outcome_id": str(outcome.get("id") or payload_ref),
                        "task_id": str(outcome.get("task_id") or "unknown-task"),
                        "outcome_type": str(outcome.get("outcome_type") or "unknown"),
                        "summary": str(outcome.get("summary") or "worker outcome"),
                        "payload_ref": payload_ref,
                        "metadata": {
                            "source_agent": outcome.get("source_agent"),
                            "success": outcome.get("success"),
                        },
                    }
                )
        return items

    @staticmethod
    def _protocol_ref(ref: Any) -> GraphRef:
        if isinstance(ref, GraphRef):
            return ref
        graph = getattr(ref, "graph", None)
        ref_id = getattr(ref, "ref_id", None)
        ref_type = getattr(ref, "ref_type", None)
        if graph is not None and ref_id is not None:
            graph_value = graph.value if hasattr(graph, "value") else str(graph)
            if graph_value == "query":
                graph_value = "ag"
            return GraphRef(graph=graph_value, ref_id=str(ref_id), ref_type=ref_type)
        return GraphRef.model_validate(ref)


__all__ = ["AgentPipeline", "PipelineCycleResult", "PipelineStepResult"]
