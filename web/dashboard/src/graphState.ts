import type { GraphName, VisualEdge, VisualGraphDelta, VisualGraphSnapshot, VisualNode } from "./types";

export interface GraphState {
  version: number;
  nodes: Record<string, VisualNode>;
  edges: Record<string, VisualEdge>;
  highlighted: Record<string, number>;
}

export type DashboardGraphs = Record<GraphName, GraphState>;

export const emptyGraphState = (): GraphState => ({
  version: 0,
  nodes: {},
  edges: {},
  highlighted: {},
});

export const emptyDashboardGraphs = (): DashboardGraphs => ({
  kg: emptyGraphState(),
  ag: emptyGraphState(),
  tg: emptyGraphState(),
  runtime: emptyGraphState(),
});

export function snapshotToGraphs(snapshot: VisualGraphSnapshot): DashboardGraphs {
  const graphs = emptyDashboardGraphs();
  (Object.keys(snapshot.graphs) as GraphName[]).forEach((graphName) => {
    const graph = snapshot.graphs[graphName];
    graphs[graphName] = {
      version: graph.version,
      nodes: Object.fromEntries(graph.nodes.map((node) => [node.id, node])),
      edges: Object.fromEntries(graph.edges.map((edge) => [edge.id, edge])),
      highlighted: {},
    };
  });
  return graphs;
}

export function applyDelta(state: GraphState, delta: VisualGraphDelta): GraphState {
  const next: GraphState = {
    version: delta.version,
    nodes: { ...state.nodes },
    edges: { ...state.edges },
    highlighted: { ...state.highlighted },
  };
  const now = Date.now();

  delta.changes.forEach((change) => {
    if (change.operation === "upsert_node" || change.operation === "update_status") {
      const previous = next.nodes[change.entity_id];
      next.nodes[change.entity_id] = {
        id: change.entity_id,
        label: change.label || previous?.label || change.entity_id,
        type: change.entity_type ?? previous?.type,
        graph: delta.graph,
        status: change.status ?? String(change.properties?.status ?? previous?.status ?? ""),
        properties: { ...(previous?.properties || {}), ...(change.properties || {}) },
      };
      next.highlighted[change.entity_id] = now;
    }

    if (change.operation === "upsert_edge" && change.source && change.target) {
      next.edges[change.entity_id] = {
        id: change.entity_id,
        source: change.source,
        target: change.target,
        label: change.label || change.edge_type || change.entity_id,
        type: change.edge_type,
        graph: delta.graph,
        properties: change.properties || {},
      };
      next.highlighted[change.entity_id] = now;
    }

    if (change.operation === "delete_node") {
      delete next.nodes[change.entity_id];
      Object.keys(next.edges).forEach((edgeId) => {
        const edge = next.edges[edgeId];
        if (edge.source === change.entity_id || edge.target === change.entity_id) delete next.edges[edgeId];
      });
    }

    if (change.operation === "delete_edge") {
      delete next.edges[change.entity_id];
    }
  });

  return next;
}

export function filteredGraph(state: GraphState, typeFilter: string, statusFilter: string): GraphState {
  const nodes = Object.fromEntries(
    Object.entries(state.nodes).filter(([, node]) => {
      const typeOk = !typeFilter || node.type === typeFilter;
      const statusOk = !statusFilter || node.status === statusFilter;
      return typeOk && statusOk;
    }),
  );
  const nodeIds = new Set(Object.keys(nodes));
  const edges = Object.fromEntries(
    Object.entries(state.edges).filter(([, edge]) => nodeIds.has(edge.source) && nodeIds.has(edge.target)),
  );
  return { ...state, nodes, edges };
}
