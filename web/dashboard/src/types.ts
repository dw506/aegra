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

export interface UnifiedOperation {
  id: string;
  target_scope: unknown[];
  status: string;
  current_round: number;
  goal_status: string;
  created_at: string | null;
  updated_at: string | null;
  metadata?: Record<string, unknown>;
}

export interface UnifiedOverview {
  asset_count: number;
  service_count: number;
  finding_count: number;
  verified_finding_count: number;
  evidence_count: number;
  access_count: number;
  current_agent: string | null;
  latest_decision: Record<string, unknown> | null;
  main_path_summary: string[];
  latest_evidence?: UnifiedEvidence | null;
  operation_status?: string;
  current_round?: number;
  goal_status?: string;
}

export interface UnifiedKgNode {
  id: string;
  type: string;
  display_name: string;
  summary: string;
  status: string;
  confidence: number;
  target: string | null;
  created_by: string | null;
  round: number | null;
  evidence_ids: string[];
  metadata: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
  linked_ag_node_ids?: string[];
}

export interface UnifiedKgEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  display_name: string;
  summary: string;
  evidence_ids: string[];
  confidence: number;
  metadata: Record<string, unknown>;
}

export interface UnifiedAgNode {
  id: string;
  type: string;
  round: number;
  agent: string;
  display_name: string;
  action_summary: string;
  result_summary: string;
  target: string | null;
  status: string;
  confidence: number;
  evidence_ids: string[];
  kg_node_ids: string[];
  is_main_path: boolean;
  started_at: string | null;
  ended_at: string | null;
  metadata: Record<string, unknown>;
  raw_type?: string;
}

export interface UnifiedAgEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  display_name: string;
  summary: string;
  evidence_ids: string[];
  confidence: number;
  metadata: Record<string, unknown>;
}

export interface UnifiedTimelineEvent {
  id: string;
  round: number;
  phase: "planner_decision" | "agent_execution" | "tool_call" | "extraction" | "result_apply" | "goal_check" | "error";
  agent: string | null;
  target: string | null;
  display_name: string;
  summary: string;
  status: string;
  tool_name: string | null;
  evidence_ids: string[];
  kg_updates: unknown[];
  ag_updates: unknown[];
  created_at: string | null;
  metadata: Record<string, unknown>;
}

export interface UnifiedEvidence {
  id: string;
  source: string;
  source_name: string;
  target: string | null;
  summary: string;
  structured_facts: unknown[];
  raw_output: string | null;
  linked_kg_node_ids: string[];
  linked_ag_node_ids: string[];
  created_by: string | null;
  round: number | null;
  created_at: string | null;
  metadata: Record<string, unknown>;
}

export interface UnifiedAgentTrace {
  id: string;
  round: number;
  planner_decision: {
    selected_agents: string[];
    decision_summary: string;
    reason: string;
    expected_outcome: string;
    blocked_by: unknown[];
    priority: "low" | "medium" | "high";
  };
  agent_states: Array<{
    agent: "ReconAgent" | "VulnAnalysisAgent" | "ExploitValidationAgent" | "AccessPivotAgent" | "GoalAgent" | "Other";
    state: "idle" | "selected" | "running" | "success" | "failed" | "skipped" | "blocked";
    summary: string;
  }>;
  created_at: string | null;
}

export interface UnifiedVisualization {
  operation: UnifiedOperation;
  overview: UnifiedOverview;
  kg: { nodes: UnifiedKgNode[]; edges: UnifiedKgEdge[] };
  ag: { nodes: UnifiedAgNode[]; edges: UnifiedAgEdge[] };
  timeline: UnifiedTimelineEvent[];
  evidence: UnifiedEvidence[];
  agent_trace: UnifiedAgentTrace[];
  attack_path: { nodes: UnifiedAgNode[]; edges: UnifiedAgEdge[] };
}
