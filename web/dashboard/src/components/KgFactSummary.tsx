import { Database, FileText, Network, Server } from "lucide-react";
import type { ReactNode } from "react";
import type { GraphState } from "../graphState";
import type { VisualNode } from "../types";

interface Props {
  graph: GraphState;
  onSelectNode: (node: VisualNode) => void;
}

interface FactItem {
  node: VisualNode;
  title: string;
  subtitle: string;
  evidenceCount: number;
}

export function KgFactSummary({ graph, onSelectNode }: Props) {
  const nodes = Object.values(graph.nodes);
  const hosts = uniqueFacts(nodes.filter((node) => node.type === "Host").map((node) => buildHostFact(node, graph)));
  const services = uniqueFacts(nodes.filter((node) => node.type === "Service").map((node) => buildServiceFact(node, graph)));
  const findings = nodes
    .filter((node) => ["Finding", "Vulnerability", "Observation"].includes(String(node.type || "")))
    .map((node) => buildGenericFact(node, graph));
  const evidence = nodes.filter((node) => node.type === "Evidence").slice(0, 4).map((node) => buildEvidenceFact(node));

  return (
    <section className="kgFactSummary" aria-label="KG fact summary">
      <SummaryColumn
        icon={<Network size={16} />}
        title="Assets"
        empty="No assets"
        items={hosts}
        onSelectNode={onSelectNode}
      />
      <SummaryColumn
        icon={<Server size={16} />}
        title="Services"
        empty="No services"
        items={services}
        onSelectNode={onSelectNode}
      />
      <SummaryColumn
        icon={<FileText size={16} />}
        title="Evidence And Findings"
        empty="No evidence"
        items={[...findings, ...evidence]}
        onSelectNode={onSelectNode}
      />
    </section>
  );
}

function SummaryColumn({
  icon,
  title,
  empty,
  items,
  onSelectNode,
}: {
  icon: ReactNode;
  title: string;
  empty: string;
  items: FactItem[];
  onSelectNode: (node: VisualNode) => void;
}) {
  return (
    <div className="kgFactColumn">
      <div className="kgFactColumnTitle">
        {icon}
        <span>{title}</span>
        <strong>{items.length}</strong>
      </div>
      <div className="kgFactList">
        {items.length ? (
          items.slice(0, 5).map((item) => (
            <button key={item.node.id} className="kgFactRow" onClick={() => onSelectNode(item.node)}>
              <span className="kgFactMain">{item.title}</span>
              <span className="kgFactSub">{item.subtitle}</span>
              <span className="kgEvidenceBadge">
                <Database size={13} />
                {item.evidenceCount}
              </span>
            </button>
          ))
        ) : (
          <div className="kgFactEmpty">{empty}</div>
        )}
      </div>
    </div>
  );
}

function buildHostFact(node: VisualNode, graph: GraphState): FactItem {
  const title = firstString(readProp(node, "address"), readProp(node, "hostname"), node.label, node.id);
  return {
    node,
    title,
    subtitle: firstString(node.status, readProp(node, "platform"), "observed host"),
    evidenceCount: countSupportingEvidence(node, graph),
  };
}

function buildServiceFact(node: VisualNode, graph: GraphState): FactItem {
  const service = firstString(readProp(node, "service_name"), readProp(node, "service"), readProp(node, "protocol"), "service");
  const port = firstString(readProp(node, "port"));
  const product = firstString(readProp(node, "product"), readProp(node, "banner"), readProp(node, "server"));
  return {
    node,
    title: port ? `${service}:${port}` : service,
    subtitle: compact(firstString(product, readProp(node, "host"), node.label), 96),
    evidenceCount: countSupportingEvidence(node, graph),
  };
}

function buildGenericFact(node: VisualNode, graph: GraphState): FactItem {
  return {
    node,
    title: compact(firstString(readProp(node, "summary"), node.label, node.type), 80),
    subtitle: firstString(node.type, node.status, "fact"),
    evidenceCount: countSupportingEvidence(node, graph),
  };
}

function buildEvidenceFact(node: VisualNode): FactItem {
  return {
    node,
    title: compact(firstString(readProp(node, "summary"), readProp(node, "source_tool"), node.label, "Evidence"), 80),
    subtitle: firstString(readProp(node, "source_tool"), readProp(node, "evidence_kind"), "evidence"),
    evidenceCount: 1,
  };
}

function uniqueFacts(items: FactItem[]) {
  const byTitle = new Map<string, FactItem>();
  for (const item of items) {
    const key = item.title.toLowerCase();
    const existing = byTitle.get(key);
    if (!existing) {
      byTitle.set(key, item);
      continue;
    }
    existing.evidenceCount += item.evidenceCount;
    if (existing.subtitle === "observed" || existing.subtitle === "observed host") {
      existing.subtitle = item.subtitle;
    }
  }
  return Array.from(byTitle.values());
}

function countSupportingEvidence(node: VisualNode, graph: GraphState) {
  const evidenceNodes = Object.values(graph.nodes).filter((item) => item.type === "Evidence");
  const bySourceRefs = evidenceNodes.filter((evidence) => {
    const refs = evidence.properties.source_refs;
    if (!Array.isArray(refs)) return false;
    return refs.some((ref) => isObject(ref) && ref.entity_id === node.id);
  }).length;
  const byEdges = Object.values(graph.edges).filter((edge) => {
    const type = String(edge.type || edge.label || "").toLowerCase();
    if (!type.includes("support") && !type.includes("evidence")) return false;
    return edge.source === node.id || edge.target === node.id;
  }).length;
  return bySourceRefs + byEdges;
}

function firstString(...values: unknown[]) {
  for (const value of values) {
    if (value === undefined || value === null || value === "") continue;
    return String(value);
  }
  return "";
}

function readProp(node: VisualNode, key: string) {
  const direct = node.properties[key];
  if (direct !== undefined && direct !== null && direct !== "") return direct;
  const nested = node.properties.properties;
  if (isObject(nested)) return nested[key];
  return undefined;
}

function compact(value: string, maxLength: number) {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, Math.max(0, maxLength - 1))}…`;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
