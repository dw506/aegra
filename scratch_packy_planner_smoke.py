from __future__ import annotations

from pprint import pprint

from src.core.agents.agent_protocol import AgentContext, AgentInput, GraphRef, GraphScope
from src.core.agents.pipeline_builders import AgentPipelineAssemblyOptions, build_optional_agent_pipeline
from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.kg import DataAsset, Goal, Host, HostsEdge, Service, TargetsEdge
from src.core.models.kg_enums import EntityStatus


def build_agent_context(operation_id: str = "op-packy-planner-smoke") -> AgentContext:
    return AgentContext(operation_id=operation_id, runtime_state_ref="runtime-packy-planner-smoke")


def build_goal_focused_kg() -> KnowledgeGraph:
    # 中文注释：
    # 这里复用最小目标图，只验证“Packy advisor 能否接进 PlannerAgent”
    # 这条链，不引入 runtime / scheduler / worker 等其他变量。
    kg = KnowledgeGraph()
    kg.add_node(Host(id="host-1", label="Gateway", status=EntityStatus.VALIDATED, confidence=0.95))
    kg.add_node(Service(id="svc-1", label="SSH", confidence=0.8))
    kg.add_node(DataAsset(id="asset-1", label="Objective Data", confidence=0.85))
    kg.add_node(Goal(id="goal-1", label="Validate Objective", category="data", confidence=0.9))
    kg.add_edge(HostsEdge(id="edge-host-svc", label="hosts", source="host-1", target="svc-1"))
    kg.add_edge(TargetsEdge(id="edge-goal-asset", label="targets", source="goal-1", target="asset-1"))
    return kg


def build_planner_input() -> AgentInput:
    ag = AttackGraphProjector().project(build_goal_focused_kg())
    goal_node = ag.get_goal_nodes()[0]
    return AgentInput(
        graph_refs=[
            GraphRef(graph=GraphScope.AG, ref_id="ag-root", ref_type="graph"),
            GraphRef(graph=GraphScope.AG, ref_id=goal_node.id, ref_type="GoalNode"),
        ],
        context=build_agent_context(),
        raw_payload={
            "ag_graph": ag.to_dict(),
            "goal_refs": [
                GraphRef(graph=GraphScope.AG, ref_id=goal_node.id, ref_type="GoalNode").model_dump(mode="json")
            ],
            "planning_context": {"top_k": 1, "max_depth": 2},
        },
    )


def build_packy_planner_pipeline():
    # 中文注释：
    # 这里通过“可选 pipeline builder”显式启用 Packy planner advisor，
    # 而不是修改全局默认的 AgentPipeline 装配行为。
    return build_optional_agent_pipeline(
        options=AgentPipelineAssemblyOptions(enable_packy_planner_advisor=True)
    )


def main() -> None:
    try:
        pipeline = build_packy_planner_pipeline()
    except ValueError as exc:
        raise SystemExit(
            "缺少 LLM 环境变量，请先设置 AEGRA_LLM_API_KEY / AEGRA_LLM_BASE_URL / AEGRA_LLM_MODEL，"
            "或兼容的 OPENAI_API_KEY / OPENAI_BASE_URL。"
            f"\n详细错误: {exc}"
        ) from exc

    result = pipeline.run_planning_cycle(
        operation_id="op-packy-planner-smoke",
        graph_refs=build_planner_input().graph_refs,
        planner_payload=build_planner_input().raw_payload,
        context=build_planner_input().context,
    )

    print("=== Packy Planner Smoke ===")
    print("Planner success:", result.success)
    print("Decision count:", len(result.final_output.decisions))
    print("Log count:", len(result.logs))
    print()

    if result.errors:
        print("Errors:")
        pprint(result.errors)
        print()

    if not result.final_output.decisions:
        print("Planner 没有产出 decision。")
        return

    decision = result.final_output.decisions[0]
    candidate = decision["payload"]["planning_candidate"]
    llm_advice = candidate["metadata"].get("llm_advice")

    print("=== First Decision ===")
    print("Summary:", decision["summary"])
    print("Score:", decision["score"])
    print("Rationale:", decision["rationale"])
    print("Action IDs:", candidate["action_ids"])
    print()

    print("=== LLM Advice ===")
    if llm_advice is None:
        print("本次调用未返回 llm_advice，Planner 已按基线逻辑继续运行。")
    else:
        pprint(llm_advice)


if __name__ == "__main__":
    main()
