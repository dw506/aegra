export type GraphName = "kg" | "ag" | "tg" | "runtime";

export type GraphOperation =
  | "upsert_node"
  | "upsert_edge"
  | "delete_node"
  | "delete_edge"
  | "update_status";

export interface VisualNode {
  id: string;
  label: string;
  type?: string | null;
  graph: GraphName;
  properties: Record<string, unknown>;
  status?: string | null;
}

export interface VisualEdge {
  id: string;
  source: string;
  target: string;
  label?: string | null;
  type?: string | null;
  graph: GraphName;
  properties: Record<string, unknown>;
}

export interface VisualGraphState {
  version: number;
  nodes: VisualNode[];
  edges: VisualEdge[];
}

export interface VisualGraphSnapshot {
  type: "graph_snapshot";
  operation_id: string;
  timestamp: string;
  graphs: Record<GraphName, VisualGraphState>;
}

export interface VisualGraphChange {
  operation: GraphOperation;
  entity_id: string;
  entity_type?: string | null;
  label?: string | null;
  source?: string | null;
  target?: string | null;
  edge_type?: string | null;
  status?: string | null;
  properties?: Record<string, unknown>;
}

export interface VisualGraphDelta {
  type: "graph_delta";
  operation_id: string;
  graph: GraphName;
  version: number;
  timestamp: string;
  changes: VisualGraphChange[];
}

export interface OperationSummary {
  operation_id: string;
  operation_status: string;
  last_updated: string;
  metadata: Record<string, unknown>;
}

export type ServerMessage = VisualGraphSnapshot | VisualGraphDelta;
