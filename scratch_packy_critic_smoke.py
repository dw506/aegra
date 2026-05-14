from __future__ import annotations

import os
from pprint import pprint

from src.core.agents.agent_protocol import AgentContext, AgentInput, GraphRef, GraphScope
from src.core.agents.critic import CriticAgent
from src.core.agents.packy_critic_advisor import PackyCriticAdvisor
from src.core.models.ag import GraphRef as ModelGraphRef
from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntime, TaskRuntimeStatus
from src.core.models.tg import TaskGraph, TaskNode, TaskStatus, TaskType


def _llm_config_summary() -> str:
    api_key = os.getenv("AEGRA_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("AEGRA_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://www.packyapi.com/v1"
    model = os.getenv("AEGRA_LLM_MODEL") or "gpt-5.2"
    key_status = "set" if api_key else "unset"
    return f"key={key_status} base_url={base_url} model={model}"


def build_task_graph() -> TaskGraph:
    graph = TaskGraph()
    graph.add_node(
        TaskNode(
            id="task-packy-critic-1",
            label="Validate low-confidence service after failed dependency",
            task_type=TaskType.SERVICE_VALIDATION,
            status=TaskStatus.BLOCKED,
            source_action_id="action-packy-critic-1",
            target_refs=[ModelGraphRef(graph="kg", ref_id="svc-packy-critic", ref_type="Service")],
            input_bindings={"service_id": "svc-packy-critic", "host_id": "host-packy-critic"},
            estimated_cost=0.7,
            estimated_risk=0.6,
            estimated_noise=0.4,
            goal_relevance=0.1,
            reason="upstream dependency failed",
            gate_ids={"dependency-gate"},
        )
    )
    return graph


def build_runtime_state() -> RuntimeState:
    state = RuntimeState(
        operation_id="op-packy-critic-smoke",
        execution=OperationRuntime(operation_id="op-packy-critic-smoke"),
    )
    state.register_task(
        TaskRuntime(
            task_id="task-packy-critic-1",
            tg_node_id="task-packy-critic-1",
            status=TaskRuntimeStatus.FAILED,
            attempt_count=2,
            max_attempts=2,
        )
    )
    return state


def main() -> None:
    try:
        advisor = PackyCriticAdvisor.from_env()
    except ValueError as exc:
        raise SystemExit(
            "缺少 LLM 环境变量，请先设置 AEGRA_LLM_API_KEY / AEGRA_LLM_BASE_URL / AEGRA_LLM_MODEL，"
            "或兼容的 OPENAI_API_KEY / OPENAI_BASE_URL。"
            f"\n详细错误: {exc}"
        ) from exc

    graph = build_task_graph()
    runtime_state = build_runtime_state()
    agent = CriticAgent(llm_advisor=advisor)
    result = agent.run(
        AgentInput(
            graph_refs=[GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph")],
            context=AgentContext(
                operation_id="op-packy-critic-smoke",
                runtime_state_ref="runtime-packy-critic-smoke",
            ),
            raw_payload={
                "tg_graph": graph.to_dict(),
                "runtime_state": runtime_state.model_dump(mode="json"),
                "runtime_summary": {
                    "failed_task_ids": ["task-packy-critic-1"],
                    "remaining_budgets": {"operation_budget_remaining": 10},
                },
                "critic_context": {
                    "failure_threshold": 2,
                    "low_value_threshold": 0.8,
                },
                "recent_outcomes": [
                    {
                        "outcome_id": "outcome-packy-critic-1",
                        "task_id": "task-packy-critic-1",
                        "outcome_type": "failed_validation",
                        "summary": "service validation failed after dependency failure",
                        "payload_ref": "runtime://outcomes/task-packy-critic-1/latest",
                    }
                ],
            },
        )
    )

    print("=== Packy Critic Smoke ===")
    print("LLM config:", _llm_config_summary())
    print("Critic success:", result.success)
    print("Decision count:", len(result.output.decisions))
    print("Replan request count:", len(result.output.replan_requests))
    print("State delta count:", len(result.output.state_deltas))
    print()

    print("Logs:")
    for log in result.output.logs:
        print(f"- {log}")
    print()

    if result.output.errors:
        print("Errors:")
        pprint(result.output.errors)
        print()

    if not result.output.decisions:
        print("Critic 没有产出 recommendation decision。")
        return

    print("=== First Recommendation ===")
    recommendation = result.output.decisions[0]["payload"]["recommendation"]
    print("Type:", recommendation["recommendation_type"])
    print("Action:", recommendation["action"])
    print("Summary:", recommendation["summary"])
    print("Rationale:", recommendation["rationale"])
    print()

    finding = recommendation["metadata"].get("llm_review")
    llm_decision = recommendation["metadata"].get("llm_decision")
    llm_validation = recommendation["metadata"].get("llm_decision_validation")
    replan_proposal = recommendation["metadata"].get("llm_replan_proposal")

    print("=== LLM Review ===")
    if finding is None and llm_decision is None and llm_validation is None and replan_proposal is None:
        print("本次调用未返回可采纳的 LLM review，Critic 已按基线逻辑继续运行。")
        print("诊断提示：如果 key/base/model 已设置，通常是网关返回了非 JSON、finding_id 不匹配，或输出被 validator 拒绝。")
    else:
        print("Review:")
        pprint(finding)
        print("Decision:")
        pprint(llm_decision)
        print("Validation:")
        pprint(llm_validation)
        print("Replan proposal:")
        pprint(replan_proposal)

    if result.output.replan_requests:
        print()
        print("=== First Replan Request ===")
        pprint(result.output.replan_requests[0])


if __name__ == "__main__":
    main()
