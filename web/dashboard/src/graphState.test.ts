import { describe, expect, it } from "vitest";
import { applyDelta, emptyGraphState } from "./graphState";
import type { VisualGraphDelta } from "./types";

describe("applyDelta", () => {
  it("upserts and deletes nodes and edges", () => {
    const delta: VisualGraphDelta = {
      type: "graph_delta",
      operation_id: "op-1",
      graph: "kg",
      version: 1,
      timestamp: "2026-05-27T00:00:00Z",
      changes: [
        { operation: "upsert_node", entity_id: "host-1", entity_type: "Host", label: "host-1", properties: {} },
        { operation: "upsert_node", entity_id: "svc-1", entity_type: "Service", label: "ssh", properties: {} },
        { operation: "upsert_edge", entity_id: "edge-1", source: "host-1", target: "svc-1", edge_type: "HAS_SERVICE", properties: {} },
      ],
    };

    const state = applyDelta(emptyGraphState(), delta);
    expect(Object.keys(state.nodes)).toEqual(["host-1", "svc-1"]);
    expect(state.edges["edge-1"].source).toBe("host-1");

    const deleted = applyDelta(state, {
      ...delta,
      version: 2,
      changes: [{ operation: "delete_node", entity_id: "host-1" }],
    });
    expect(deleted.nodes["host-1"]).toBeUndefined();
    expect(deleted.edges["edge-1"]).toBeUndefined();
  });

  it("updates status without dropping existing properties", () => {
    const state = applyDelta(emptyGraphState(), {
      type: "graph_delta",
      operation_id: "op-1",
      graph: "ag",
      version: 1,
      timestamp: "2026-05-27T00:00:00Z",
      changes: [
        { operation: "upsert_node", entity_id: "execution-1", entity_type: "Execution", label: "scan", properties: { attempt: 1 } },
      ],
    });

    const updated = applyDelta(state, {
      type: "graph_delta",
      operation_id: "op-1",
      graph: "ag",
      version: 2,
      timestamp: "2026-05-27T00:00:01Z",
      changes: [
        { operation: "update_status", entity_id: "execution-1", status: "running", properties: { worker: "ExecutionAgent" } },
      ],
    });

    expect(updated.nodes["execution-1"].status).toBe("running");
    expect(updated.nodes["execution-1"].properties).toMatchObject({ attempt: 1, worker: "ExecutionAgent" });
  });
});
