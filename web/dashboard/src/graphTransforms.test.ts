import { describe, expect, it } from "vitest";
import type { GraphState } from "./graphState";
import {
  buildDisplayName,
  buildKgAssetSummaries,
  buildOperationOverview,
  groupCycles,
  inferStepOrder,
} from "./graphTransforms";
import type { VisualNode } from "./types";

describe("graphTransforms", () => {
  it("builds human readable AG display names", () => {
    expect(buildDisplayName(agNode("s1", "HOST_KNOWN", { asset: "10.20.0.22" }))).toBe("发现主机：10.20.0.22");
    expect(buildDisplayName(agNode("svc1", "SERVICE_CONFIRMED", { asset: "10.20.0.22", port: 8080 }))).toBe("验证服务开放：10.20.0.22:8080");
    expect(buildDisplayName(agNode("tool1", "TOOL_CALL", { tool_name: "nmap" }))).toBe("工具调用：nmap");
  });

  it("infers AG step order", () => {
    expect(inferStepOrder(agNode("cycle1", "ATTACK_CYCLE"))).toBe(1);
    expect(inferStepOrder(agNode("result1", "STAGE_RESULT"))).toBe(5);
    expect(inferStepOrder(agNode("custom1", "CUSTOM", { step_order: 7 }))).toBe(7);
  });

  it("groups AG timeline nodes by cycle and sorts execution chain", () => {
    const graph = graphState([
      agNode("result2", "STAGE_RESULT", { cycle_index: 2, status: "blocked", stop_reason: "policy" }),
      agNode("plan1", "PLANNER_DECISION", { cycle_index: 1, selected_stage: "recon" }),
      agNode("cycle1", "ATTACK_CYCLE", { cycle_index: 1 }),
      agNode("agent1", "AGENT_EXECUTION", { cycle_index: 1, selected_agent: "recon_agent" }),
      agNode("tool1", "TOOL_CALL", { cycle_index: 1, tool_name: "nmap" }),
      agNode("result1", "STAGE_RESULT", { cycle_index: 1, status: "success", execution_success: true }),
    ]);

    const cycles = groupCycles(graph);

    expect(cycles).toHaveLength(2);
    expect(cycles[0].cycleIndex).toBe(1);
    expect(cycles[0].nodes.map((node) => node.id)).toEqual(["cycle1", "plan1", "agent1", "tool1", "result1"]);
    expect(cycles[0].selectedStage).toBe("recon");
    expect(cycles[0].selectedAgent).toBe("recon_agent");
    expect(cycles[0].executionSuccess).toBe("yes");
    expect(cycles[1].stopReason).toBe("policy");
  });

  it("builds operation overview from runtime metadata and AG fallback", () => {
    const runtime = graphState([
      runtimeNode("runtime:op-1", {
        operation_id: "op-1",
        operation_status: "running",
        target_count: 2,
        last_updated: "2026-06-03T00:00:00Z",
        metadata: {
          control_cycle_history: [{ cycle_index: 1 }, { cycle_index: 2 }],
          last_control_cycle: { cycle_index: 2 },
          goal_satisfied: false,
        },
      }),
    ]);
    const overview = buildOperationOverview(undefined, runtime, graphState([agNode("cycle1", "ATTACK_CYCLE")]));

    expect(overview.operationId).toBe("op-1");
    expect(overview.cycleCount).toBe(2);
    expect(overview.lastControlCycle.cycle_index).toBe(2);
  });

  it("builds KG asset summary with services, observations, evidence, and findings", () => {
    const kg = graphState([
      kgNode("host-1", "Host", { address: "10.20.0.22", scope: "in scope", goal_target: true }),
      kgNode("svc-1", "Service", { protocol: "tcp", port: 8080 }),
      kgNode("obs-1", "Observation", { summary: "HTTP service" }),
      kgNode("ev-1", "Evidence", { summary: "nmap scan" }),
      kgNode("finding-1", "Finding", { summary: "exposed service" }),
    ], [
      ["e1", "host-1", "svc-1", "HOSTS"],
      ["e2", "obs-1", "host-1", "OBSERVED_ON"],
      ["e3", "obs-1", "ev-1", "SUPPORTED_BY"],
      ["e4", "finding-1", "host-1", "AFFECTS"],
    ]);

    const [asset] = buildKgAssetSummaries(kg);

    expect(asset.title).toBe("10.20.0.22");
    expect(asset.goalTarget).toBe(true);
    expect(asset.services.map((node) => node.id)).toEqual(["svc-1"]);
    expect(asset.observations.map((node) => node.id)).toEqual(["obs-1"]);
    expect(asset.evidence.map((node) => node.id)).toEqual(["ev-1"]);
    expect(asset.findings.map((node) => node.id)).toEqual(["finding-1"]);
  });
});

function graphState(nodes: VisualNode[], edges: Array<[string, string, string, string]> = []): GraphState {
  return {
    version: 1,
    nodes: Object.fromEntries(nodes.map((node) => [node.id, node])),
    edges: Object.fromEntries(edges.map(([id, source, target, type]) => [id, { id, source, target, type, label: type, graph: "kg" as const, properties: {} }])),
    highlighted: {},
  };
}

function agNode(id: string, nodeType: string, properties: Record<string, unknown> = {}): VisualNode {
  return {
    id,
    label: nodeType,
    type: nodeType,
    graph: "ag",
    status: String(properties.status || ""),
    properties: { node_type: nodeType, kind: "process", ...properties },
  };
}

function kgNode(id: string, type: string, properties: Record<string, unknown>): VisualNode {
  return {
    id,
    label: String(properties.address || properties.summary || id),
    type,
    graph: "kg",
    properties,
  };
}

function runtimeNode(id: string, properties: Record<string, unknown>): VisualNode {
  return {
    id,
    label: String(properties.operation_id || id),
    type: "OperationRuntime",
    graph: "runtime",
    status: String(properties.operation_status || ""),
    properties,
  };
}
