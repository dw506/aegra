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
  discovered_hosts?: number;
  open_services?: number;
  open_ports?: number;
  identified_services?: number;
  vulnerability_tags?: string[];
  vulnerability_tag_count?: number;
  last_cycle_summary?: {
    cycle_index: number;
    title: string;
    summary: string;
    agent: string | null;
    status: string | null;
  } | null;
  domains?: number;
  identities?: number;
  attack_hypotheses?: number;
  service_count: number;
  finding_count: number;
  verified_finding_count: number;
  evidence_count: number;
  access_count: number;
  current_agent: string | null;
  current_phase?: string;
  current_goal?: string;
  current_best_path?: string;
  latest_decision: Record<string, unknown> | null;
  attack_chain_steps?: Array<Record<string, unknown>>;
  main_path_summary: string[];
  latest_evidence?: UnifiedEvidence | null;
  operation_status?: string;
  current_round?: number;
  goal_status?: string;
  progress_steps?: Array<{ label: string; status: "done" | "active" | "pending" | string }>;
  success_condition_progress?: Record<string, unknown>;
}

export interface UnifiedKgNode {
  id: string;
  type: string;
  display_name: string;
  summary: string;
  status: string;
  confidence: number;
  category?: string;
  role?: string | null;
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
  short_label?: string;
  compact_label?: string;
  semantic_role?: string;
  chain_importance?: "high" | "medium" | "low" | string;
  action_summary: string;
  result_summary: string;
  cycle_index?: number;
  capability?: string | null;
  intent?: string;
  target_summary?: string;
  planner_reason?: string | null;
  input_facts?: unknown[];
  target: string | null;
  status: string;
  confidence: number;
  evidence_ids: string[];
  finding_ids?: string[];
  kg_node_ids: string[];
  kg_delta?: Record<string, unknown>;
  tool_trace_refs?: string[];
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
  cycle_index?: number;
  phase: "planner_decision" | "agent_execution" | "tool_call" | "extraction" | "result_apply" | "goal_check" | "error";
  agent: string | null;
  capability?: string | null;
  intent?: string | null;
  target: string | null;
  target_summary?: string | null;
  display_name: string;
  summary: string;
  result_summary?: string;
  status: string;
  tool_name: string | null;
  tool_trace_refs?: string[];
  evidence_ids: string[];
  finding_ids?: string[];
  kg_updates: unknown[];
  ag_updates: unknown[];
  planner_reason?: string | null;
  input_facts?: unknown[];
  kg_delta?: Record<string, unknown>;
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
  cycle_index?: number;
  round: number;
  selected_agent?: string | null;
  selected_stage?: string | null;
  objective?: string | null;
  task_brief?: string | null;
  status?: string;
  summary?: string;
  tool_traces?: UnifiedToolTrace[];
  tool_count?: number;
  evidence_count?: number;
  finding_count?: number;
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

export interface UnifiedFinding {
  id: string;
  finding_id: string;
  title: string;
  summary: string;
  kind: string;
  severity: string;
  status: string;
  confidence: unknown;
  target: string | null;
  evidence_ids: string[];
  source_agent: string | null;
  stage_task_id: string | null;
  recorded_at: string | null;
  metadata: Record<string, unknown>;
}

export interface UnifiedToolTrace {
  id: string;
  round: number;
  agent: string | null;
  stage: string | null;
  step: number | null;
  server_id: string | null;
  tool_name: string;
  arguments: Record<string, unknown>;
  success: boolean | null;
  exit_code: unknown;
  summary: string;
  stdout_excerpt: string;
  stderr_excerpt: string;
  raw_output_ref: string | null;
  evidence_ids: string[];
  extracted_facts: Array<Record<string, unknown>>;
  writeback_status: string;
  created_at: string | null;
  timeline_event_ids: string[];
  metadata: Record<string, unknown>;
}

export interface UnifiedServiceMatrixRow {
  host_id: string;
  host: string;
  hostname?: string;
  role: string;
  role_guess?: string;
  status: string;
  confidence: number;
  services: Array<{
    service_id: string;
    name: string;
    port: string | null;
    protocol: string | null;
    product?: string | null;
    version?: string | null;
    status: string;
    confidence: number;
    evidence_ids: string[];
    summary: string;
    vulnerability_tags?: string[];
  }>;
  open_ports?: number;
  vulnerability_tags?: string[];
  first_seen?: string | null;
  last_seen?: string | null;
  related_cycles?: Array<{
    cycle_index: number;
    title: string;
    summary: string;
    agent: string | null;
    status: string | null;
  }>;
  evidence_ids: string[];
}

export interface UnifiedPlannerReasoning {
  id: string;
  cycle_index: number;
  input_summary: string[];
  selected_agent: string | null;
  required_capability: string | null;
  target_selector: string;
  decision: string | null;
  reason: string;
  expected_evidence: unknown[];
  status: string | null;
  tool_count: number;
  evidence_count: number;
}

export interface UnifiedVisualization {
  operation: UnifiedOperation;
  overview: UnifiedOverview;
  kg: { nodes: UnifiedKgNode[]; edges: UnifiedKgEdge[] };
  ag: { nodes: UnifiedAgNode[]; edges: UnifiedAgEdge[] };
  timeline: UnifiedTimelineEvent[];
  tool_trace: UnifiedToolTrace[];
  evidence: UnifiedEvidence[];
  findings: UnifiedFinding[];
  agent_trace: UnifiedAgentTrace[];
  attack_path: { nodes: UnifiedAgNode[]; edges: UnifiedAgEdge[] };
  risk_tags?: string[];
  service_matrix?: UnifiedServiceMatrixRow[];
  kg_groups?: {
    assets: UnifiedKgNode[];
    services: UnifiedKgNode[];
    domains_identities: UnifiedKgNode[];
    findings: UnifiedKgNode[];
    evidence: UnifiedKgNode[];
    other: UnifiedKgNode[];
  };
  planner_reasoning?: UnifiedPlannerReasoning[];
}
