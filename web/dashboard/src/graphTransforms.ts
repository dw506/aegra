import type { GraphState } from "./graphState";
import type { OperationSummary, VisualNode } from "./types";

export const AG_STEP_ORDER: Record<string, number> = {
  ATTACK_CYCLE: 1,
  PLANNER_DECISION: 2,
  AGENT_EXECUTION: 3,
  TOOL_CALL: 4,
  STAGE_RESULT: 5,
  HANDOFF_SUGGESTION: 6,
  BLOCKED_REASON: 6,
};

export interface CycleGroup {
  cycleIndex: number;
  title: string;
  nodes: VisualNode[];
  selectedStage: string;
  selectedAgent: string;
  executionSuccess: string;
  stopped: string;
  stopReason: string;
}

export interface OperationOverview {
  operationId: string;
  operationStatus: string;
  targetCount: number | string;
  cycleCount: number;
  lastUpdated: string;
  lastControlCycle: Record<string, unknown>;
  goalSatisfied: string;
  needsReplan: string;
}

export interface KgAssetSummary {
  host: VisualNode;
  title: string;
  scope: string;
  goalTarget: boolean;
  services: VisualNode[];
  observations: VisualNode[];
  evidence: VisualNode[];
  findings: VisualNode[];
  latestCycleIndex: string;
}

export function buildDisplayName(node: VisualNode): string {
  const type = nodeType(node);
  const meta = readMeta(node);
  const visualTitle = firstString(readValue(node, "visual_title"), readValue(node, "visual_label"), readValue(node, "display_name"), meta.visual_title, meta.visual_label, meta.display_name);
  if (visualTitle) return visualTitle;

  const asset = firstString(readValue(node, "asset"), readValue(node, "host"), readValue(node, "target"), meta.asset, meta.host, meta.target);
  const port = firstString(readValue(node, "port"), meta.port);
  const protocol = firstString(readValue(node, "protocol"), meta.protocol, "tcp");
  const service = firstString(readValue(node, "service"), readValue(node, "service_name"), meta.service, meta.service_name);
  const cycle = firstString(readValue(node, "cycle_index"), meta.cycle_index);
  const agent = firstString(readValue(node, "agent_name"), readValue(node, "selected_agent"), meta.agent_name, meta.selected_agent);
  const stage = firstString(readValue(node, "selected_stage"), readValue(node, "stage"), readValue(node, "capability"), meta.selected_stage, meta.stage, meta.capability);
  const tool = firstString(readValue(node, "tool_name"), meta.tool_name);
  const status = firstString(node.status, readValue(node, "status"), meta.status);
  const reason = firstString(readValue(node, "reason"), readValue(node, "stop_reason"), meta.reason, meta.stop_reason);
  const goal = firstString(readValue(node, "goal"), meta.goal);

  switch (type) {
    case "HOST_KNOWN":
      return `发现主机：${asset || "未知主机"}`;
    case "HOST_VALIDATED":
      return `验证主机可达：${asset || "未知主机"}`;
    case "SERVICE_KNOWN":
      return compactInstance(`发现服务：${asset || ""}${port ? `:${port}` : service ? ` ${service}` : protocol ? ` ${protocol}` : ""}`);
    case "SERVICE_VALIDATED":
    case "SERVICE_CONFIRMED":
      return compactInstance(`验证服务开放：${asset || ""}${port ? `:${port}` : service ? ` ${service}` : ""}`);
    case "MANAGED_SESSION":
    case "MANAGED_SESSION_AVAILABLE":
    case "SESSION_ACTIVE_ON_HOST":
      return `建立会话：${meta.user && asset ? `${meta.user}@${asset}` : firstString(meta.session_id, readValue(node, "session_id"), asset, "未知会话")}`;
    case "PRIVILEGE_VALIDATED":
      return `权限确认：${firstString(meta.principal, meta.privilege, readValue(node, "principal"), readValue(node, "privilege"), "未知权限")}`;
    case "GOAL_STATE_SATISFIED":
      return `目标达成：${goal || asset || "任务目标"}`;
    case "ATTACK_CYCLE":
      return `Cycle ${cycle || "?"}`;
    case "PLANNER_DECISION":
      return `规划决策：${stage || firstString(meta.decision, readValue(node, "decision"), "未命名决策")}`;
    case "AGENT_EXECUTION":
      return `执行 Agent：${agent || "未知 Agent"}`;
    case "TOOL_CALL":
      return `工具调用：${tool || "未知工具"}`;
    case "STAGE_RESULT":
      return `阶段结果：${status || "unknown"}`;
    case "BLOCKED_REASON":
      return `阻断原因：${reason || "未知原因"}`;
    case "HANDOFF_SUGGESTION":
      return `下一步建议：${firstString(meta.suggested_stage, meta.suggested_agent, readValue(node, "suggested_stage"), readValue(node, "suggested_agent"), "待规划")}`;
    case "Host":
      return firstString(readValue(node, "address"), readValue(node, "hostname"), node.label, node.id);
    case "Service":
      return `${protocol || "tcp"}${port ? `:${port}` : service ? `:${service}` : ""}`;
    case "Observation":
      return `观察：${firstString(readValue(node, "summary"), readValue(node, "observation_kind"), shortId(node.id))}`;
    case "Evidence":
      return `证据：${firstString(readValue(node, "summary"), readValue(node, "source_tool"), readValue(node, "evidence_kind"), shortId(node.id))}`;
    default:
      return firstString(node.label, type, "未命名节点");
  }
}

export function buildNodeSubtitle(node: VisualNode): string {
  const meta = readMeta(node);
  return firstString(
    readValue(node, "visual_subtitle"),
    readValue(node, "visual_summary"),
    readValue(node, "summary"),
    readValue(node, "output_summary"),
    meta.visual_subtitle,
    meta.visual_summary,
    meta.summary,
    meta.output_summary,
  );
}

export function inferStepOrder(node: VisualNode): number {
  const explicit = Number(firstString(readValue(node, "step_order"), readMeta(node).step_order));
  if (Number.isFinite(explicit) && explicit > 0) return explicit;
  return AG_STEP_ORDER[nodeType(node)] || 99;
}

export function groupCycles(graph: GraphState): CycleGroup[] {
  const nodes = Object.values(graph.nodes).filter((node) => node.graph === "ag");
  const processNodes = nodes.filter((node) => readValue(node, "kind") === "process" || AG_STEP_ORDER[nodeType(node)]);
  const groups = new Map<number, VisualNode[]>();

  for (const node of processNodes) {
    const cycle = inferCycleIndex(node);
    if (!groups.has(cycle)) groups.set(cycle, []);
    groups.get(cycle)?.push(node);
  }

  return Array.from(groups.entries())
    .sort(([a], [b]) => a - b)
    .map(([cycleIndex, items]) => {
      const sorted = items.sort((a, b) => inferStepOrder(a) - inferStepOrder(b) || a.id.localeCompare(b.id));
      const merged = mergeCycleProps(sorted);
      return {
        cycleIndex,
        title: `Cycle ${cycleIndex}`,
        nodes: sorted,
        selectedStage: firstString(merged.selected_stage, merged.stage, merged.capability, "n/a"),
        selectedAgent: firstString(merged.selected_agent, merged.agent_name, "n/a"),
        executionSuccess: formatBoolean(merged.execution_success ?? merged.success),
        stopped: formatBoolean(merged.stopped),
        stopReason: firstString(merged.stop_reason, merged.reason, "n/a"),
      };
    });
}

export function buildOperationOverview(
  operation: OperationSummary | undefined,
  runtime: GraphState,
  ag: GraphState,
): OperationOverview {
  const root = Object.values(runtime.nodes).find((node) => node.type === "OperationRuntime");
  const metadata = readMeta(root);
  const history = Array.isArray(metadata.control_cycle_history) ? metadata.control_cycle_history : [];
  const lastControlCycle = asRecord(metadata.last_control_cycle);
  return {
    operationId: firstString(operation?.operation_id, readValue(root, "operation_id"), ""),
    operationStatus: firstString(operation?.operation_status, root?.status, readValue(root, "operation_status"), "unknown"),
    targetCount: firstString(readValue(root, "target_count"), metadata.target_count, operation?.metadata?.target_count, "n/a"),
    cycleCount: history.length || Object.values(ag.nodes).filter((node) => nodeType(node) === "ATTACK_CYCLE").length,
    lastUpdated: firstString(operation?.last_updated, readValue(root, "last_updated"), "n/a"),
    lastControlCycle,
    goalSatisfied: formatBoolean(firstString(readValue(root, "goal_satisfied"), metadata.goal_satisfied, operation?.metadata?.goal_satisfied)),
    needsReplan: formatBoolean(firstString(readValue(root, "needs_replan"), metadata.needs_replan, operation?.metadata?.needs_replan)),
  };
}

export function buildStatePath(graph: GraphState): VisualNode[] {
  const order = ["HOST_KNOWN", "HOST_VALIDATED", "SERVICE_VALIDATED", "SERVICE_CONFIRMED", "MANAGED_SESSION", "MANAGED_SESSION_AVAILABLE", "PRIVILEGE_VALIDATED", "GOAL_STATE_SATISFIED"];
  return Object.values(graph.nodes)
    .filter((node) => order.includes(nodeType(node)))
    .sort((a, b) => order.indexOf(nodeType(a)) - order.indexOf(nodeType(b)) || a.id.localeCompare(b.id));
}

export function buildKgAssetSummaries(graph: GraphState, ag?: GraphState): KgAssetSummary[] {
  const nodes = Object.values(graph.nodes);
  const hosts = nodes.filter((node) => node.type === "Host");
  return hosts.map((host) => {
    const related = relatedNodes(graph, host.id);
    const services = related.filter((node) => node.type === "Service");
    const observations = related.filter((node) => node.type === "Observation");
    const evidence = related.filter((node) => node.type === "Evidence");
    const findings = related.filter((node) => node.type === "Finding" || node.type === "Vulnerability");
    const hostRefs = [host.id, host.label, readValue(host, "address"), readValue(host, "hostname")]
      .filter((value) => value !== undefined && value !== null && value !== "")
      .map(String);
    const agCycles = Object.values(ag?.nodes || {})
      .filter((node) => {
        const serialized = JSON.stringify(node.properties || {});
        return hostRefs.some((ref) => ref && serialized.includes(ref));
      })
      .map(inferCycleIndex)
      .filter((value) => value > 0);
    return {
      host,
      title: buildDisplayName(host),
      scope: firstString(readValue(host, "scope"), readValue(host, "scope_status"), host.status, "observed"),
      goalTarget: truthy(readValue(host, "goal_target")) || related.some((node) => node.type === "Goal"),
      services,
      observations,
      evidence,
      findings,
      latestCycleIndex: agCycles.length ? String(Math.max(...agCycles)) : firstString(readValue(host, "cycle_index"), "n/a"),
    };
  });
}

export function readValue(node: VisualNode | undefined | null, key: string): unknown {
  if (!node) return undefined;
  const direct = (node as unknown as Record<string, unknown>)[key];
  if (direct !== undefined && direct !== null && direct !== "") return direct;
  const props = node.properties || {};
  if (props[key] !== undefined && props[key] !== null && props[key] !== "") return props[key];
  const nested = props.properties;
  if (isRecord(nested)) return nested[key];
  return undefined;
}

export function nodeType(node: VisualNode): string {
  return firstString(readValue(node, "node_type"), readValue(node, "kind"), node.type, node.label);
}

function inferCycleIndex(node: VisualNode): number {
  const value = Number(firstString(readValue(node, "cycle_index"), readMeta(node).cycle_index));
  if (Number.isFinite(value) && value > 0) return value;
  const fromId = node.id.match(/cycle[-:_]*(\d+)/i)?.[1];
  return fromId ? Number(fromId) : 0;
}

function mergeCycleProps(nodes: VisualNode[]) {
  return nodes.reduce<Record<string, unknown>>((merged, node) => ({ ...merged, ...node.properties, ...asRecord(node.properties.properties) }), {});
}

function relatedNodes(graph: GraphState, nodeId: string): VisualNode[] {
  const seen = new Set<string>();
  for (const edge of Object.values(graph.edges)) {
    if (edge.source === nodeId) seen.add(edge.target);
    if (edge.target === nodeId) seen.add(edge.source);
  }
  for (const edge of Object.values(graph.edges)) {
    if (seen.has(edge.source)) seen.add(edge.target);
    if (seen.has(edge.target)) seen.add(edge.source);
  }
  const direct = Object.values(graph.nodes).filter((node) => {
    const serialized = JSON.stringify(node.properties || {});
    return serialized.includes(nodeId) || seen.has(node.id);
  });
  return direct.filter((node) => node.id !== nodeId);
}

function readMeta(node: VisualNode | undefined | null): Record<string, unknown> {
  return asRecord(readValue(node, "metadata"));
}

function asRecord(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function truthy(value: unknown) {
  return value === true || value === "true" || value === "yes" || value === 1 || value === "1";
}

function formatBoolean(value: unknown): string {
  if (value === undefined || value === null || value === "") return "n/a";
  if (value === true || value === "true") return "yes";
  if (value === false || value === "false") return "no";
  return String(value);
}

function firstString(...values: unknown[]) {
  for (const value of values) {
    if (value === undefined || value === null || value === "") continue;
    return String(value);
  }
  return "";
}

function compactInstance(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function shortId(id: string) {
  const parts = id.split(/[:/\\#-]+/);
  return parts[parts.length - 1] || id;
}
