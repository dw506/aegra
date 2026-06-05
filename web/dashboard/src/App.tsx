import {
  Activity,
  Bot,
  Download,
  Eye,
  Filter,
  GitBranch,
  ListChecks,
  RefreshCw,
  Route,
  Search,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { fetchVisualization, listOperations } from "./api";
import { CytoscapeGraph } from "./components/CytoscapeGraph";
import type { GraphState } from "./graphState";
import type {
  OperationSummary,
  UnifiedAgEdge,
  UnifiedAgNode,
  UnifiedEvidence,
  UnifiedKgEdge,
  UnifiedKgNode,
  UnifiedTimelineEvent,
  UnifiedVisualization,
  VisualEdge,
  VisualNode,
} from "./types";

type ViewId = "overview" | "kg" | "ag" | "timeline" | "evidence" | "trace" | "attack-path";
type DetailItem =
  | { kind: "KG Node"; payload: UnifiedKgNode }
  | { kind: "AG Node"; payload: UnifiedAgNode }
  | { kind: "Timeline Event"; payload: UnifiedTimelineEvent }
  | { kind: "Evidence"; payload: UnifiedEvidence }
  | null;

const views: Array<{ id: ViewId; label: string; icon: typeof Activity }> = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "kg", label: "KG View", icon: GitBranch },
  { id: "ag", label: "AG View", icon: Route },
  { id: "timeline", label: "Timeline", icon: ListChecks },
  { id: "evidence", label: "Evidence", icon: Eye },
  { id: "trace", label: "Agent Trace", icon: Bot },
  { id: "attack-path", label: "Attack Path", icon: Search },
];

const kgTypeOptions = ["Host", "Service", "Network", "Finding", "Evidence", "Credential", "Session", "Goal", "Unknown"];
const kgStatusOptions = ["observed", "suspected", "verified", "rejected", "active", "failed", "blocked"];
const statusOptions = ["pending", "running", "success", "failed", "blocked", "skipped", "observed", "suspected", "verified", "active"];

export default function App() {
  const [operations, setOperations] = useState<OperationSummary[]>([]);
  const [operationId, setOperationId] = useState("");
  const [visualization, setVisualization] = useState<UnifiedVisualization | null>(null);
  const [activeView, setActiveView] = useState<ViewId>("overview");
  const [detail, setDetail] = useState<DetailItem>(null);
  const [typeFilter, setTypeFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [agentFilter, setAgentFilter] = useState("");
  const [roundFilter, setRoundFilter] = useState("");
  const [sourceFilter, setSourceFilter] = useState("");
  const [targetFilter, setTargetFilter] = useState("");
  const [mainPathOnly, setMainPathOnly] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    listOperations().then((items) => {
      setOperations(items);
      setOperationId((current) => current || items[0]?.operation_id || "");
    });
  }, []);

  useEffect(() => {
    if (!operationId) return;
    refreshVisualization(operationId);
  }, [operationId]);

  const refreshVisualization = async (id = operationId) => {
    if (!id) return;
    setLoading(true);
    try {
      setVisualization(await fetchVisualization(id));
      setDetail(null);
    } finally {
      setLoading(false);
    }
  };

  const kgNodes = useMemo(() => {
    const nodes = visualization?.kg.nodes || [];
    return nodes.filter((node) => matches(typeFilter, node.type) && matches(statusFilter, node.status));
  }, [visualization, typeFilter, statusFilter]);

  const agNodes = useMemo(() => {
    const nodes = visualization?.ag.nodes || [];
    return nodes.filter((node) => {
      const roundOk = !roundFilter || String(node.round) === roundFilter;
      const mainOk = !mainPathOnly || node.is_main_path;
      return roundOk && mainOk && matches(agentFilter, node.agent) && matches(statusFilter, node.status);
    });
  }, [visualization, roundFilter, mainPathOnly, agentFilter, statusFilter]);

  const evidenceItems = useMemo(() => {
    const items = visualization?.evidence || [];
    return items.filter((item) => {
      const roundOk = !roundFilter || String(item.round ?? "") === roundFilter;
      const sourceOk = !sourceFilter || item.source === sourceFilter || item.source_name === sourceFilter || item.created_by === sourceFilter;
      const targetOk = !targetFilter || item.target === targetFilter;
      return roundOk && sourceOk && targetOk;
    });
  }, [visualization, roundFilter, sourceFilter, targetFilter]);

  const rounds = useMemo(() => unique((visualization?.ag.nodes || []).map((node) => String(node.round))), [visualization]);
  const agents = useMemo(() => unique((visualization?.ag.nodes || []).map((node) => node.agent)), [visualization]);
  const evidenceSources = useMemo(() => unique((visualization?.evidence || []).flatMap((item) => [item.source, item.source_name, item.created_by || ""])), [visualization]);
  const targets = useMemo(() => unique((visualization?.evidence || []).map((item) => item.target || "")), [visualization]);

  const exportJson = () => {
    const blob = new Blob([JSON.stringify(visualization || {}, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${operationId || "aegra"}-visualization.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="appShell">
      <header className="topbar">
        <div>
          <h1>Aegra Automation Console</h1>
          <p>{visualization?.operation.id || operationId || "No operation selected"}</p>
        </div>
        <div className="topbarControls">
          <select value={operationId} onChange={(event) => setOperationId(event.target.value)} aria-label="Operation">
            {operations.map((operation) => (
              <option key={operation.operation_id} value={operation.operation_id}>
                {operation.operation_id}
              </option>
            ))}
          </select>
          <button className="iconButton" onClick={() => refreshVisualization()} title="Refresh visualization">
            <RefreshCw size={18} className={loading ? "spin" : ""} />
          </button>
          <button className="iconTextButton" onClick={exportJson}>
            <Download size={17} />
            JSON
          </button>
        </div>
      </header>

      <main className="consoleWorkspace">
        <section className="consoleSurface">
          <div className="tabs">
            {views.map((view) => {
              const Icon = view.icon;
              return (
                <button key={view.id} className={activeView === view.id ? "active" : ""} onClick={() => setActiveView(view.id)}>
                  <Icon size={16} />
                  {view.label}
                </button>
              );
            })}
          </div>

          {visualization ? (
            <>
              {activeView === "overview" && <OverviewView data={visualization} onSelectEvidence={(item) => setDetail({ kind: "Evidence", payload: item })} />}
              {activeView === "kg" && (
                <GraphPanel
                  title="Environment Facts"
                  filters={
                    <>
                      <FilterSelect value={typeFilter} onChange={setTypeFilter} label="Type" options={kgTypeOptions} />
                      <FilterSelect value={statusFilter} onChange={setStatusFilter} label="Status" options={kgStatusOptions} />
                    </>
                  }
                  graph={kgGraphState(kgNodes, visualization.kg.edges)}
                  onSelectNode={(node) => {
                    const payload = visualization.kg.nodes.find((item) => item.id === node.id);
                    if (payload) setDetail({ kind: "KG Node", payload });
                  }}
                  countLabel={`${kgNodes.length} nodes / ${visibleEdgeCount(kgNodes, visualization.kg.edges)} edges`}
                />
              )}
              {activeView === "ag" && (
                <GraphPanel
                  title="Attack Process"
                  filters={
                    <>
                      <FilterSelect value={roundFilter} onChange={setRoundFilter} label="Round" options={rounds} />
                      <FilterSelect value={agentFilter} onChange={setAgentFilter} label="Agent" options={agents} />
                      <FilterSelect value={statusFilter} onChange={setStatusFilter} label="Status" options={statusOptions} />
                      <label className="toggleControl">
                        <input type="checkbox" checked={mainPathOnly} onChange={(event) => setMainPathOnly(event.target.checked)} />
                        Main path
                      </label>
                    </>
                  }
                  graph={agGraphState(agNodes, visualization.ag.edges)}
                  onSelectNode={(node) => {
                    const payload = visualization.ag.nodes.find((item) => item.id === node.id);
                    if (payload) setDetail({ kind: "AG Node", payload });
                  }}
                  countLabel={`${agNodes.length} nodes / ${visibleEdgeCount(agNodes, visualization.ag.edges)} edges`}
                />
              )}
              {activeView === "timeline" && <TimelineView events={visualization.timeline} onSelect={(item) => setDetail({ kind: "Timeline Event", payload: item })} />}
              {activeView === "evidence" && (
                <EvidenceView
                  items={evidenceItems}
                  rounds={rounds}
                  sources={evidenceSources}
                  targets={targets}
                  roundFilter={roundFilter}
                  setRoundFilter={setRoundFilter}
                  sourceFilter={sourceFilter}
                  setSourceFilter={setSourceFilter}
                  targetFilter={targetFilter}
                  setTargetFilter={setTargetFilter}
                  onSelect={(item) => setDetail({ kind: "Evidence", payload: item })}
                />
              )}
              {activeView === "trace" && <AgentTraceView traces={visualization.agent_trace} roundFilter={roundFilter} setRoundFilter={setRoundFilter} rounds={rounds} />}
              {activeView === "attack-path" && (
                <GraphPanel
                  title="Current Main Attack Path"
                  filters={<div className="counts">Main chain only</div>}
                  graph={agGraphState(visualization.attack_path.nodes, visualization.attack_path.edges)}
                  onSelectNode={(node) => {
                    const payload = visualization.attack_path.nodes.find((item) => item.id === node.id);
                    if (payload) setDetail({ kind: "AG Node", payload });
                  }}
                  countLabel={`${visualization.attack_path.nodes.length} nodes / ${visualization.attack_path.edges.length} edges`}
                />
              )}
            </>
          ) : (
            <div className="emptyState">No visualization data</div>
          )}
        </section>

        <UnifiedDetailPanel detail={detail} onClose={() => setDetail(null)} />
      </main>
    </div>
  );
}

function OverviewView({ data, onSelectEvidence }: { data: UnifiedVisualization; onSelectEvidence: (item: UnifiedEvidence) => void }) {
  const overview = data.overview;
  return (
    <div className="consoleView">
      <section className="metricGrid">
        <Metric label="Operation ID" value={data.operation.id} />
        <Metric label="Target scope" value={String(data.operation.target_scope.length)} />
        <Metric label="Current round" value={data.operation.current_round} />
        <Metric label="Current agent" value={overview.current_agent || "Unknown"} />
        <Metric label="Goal status" value={data.operation.goal_status} />
        <Metric label="Discovered assets" value={overview.asset_count} />
        <Metric label="Services" value={overview.service_count} />
        <Metric label="Findings" value={overview.finding_count} />
        <Metric label="Verified findings" value={overview.verified_finding_count} />
        <Metric label="Evidence items" value={overview.evidence_count} />
        <Metric label="Active access" value={overview.access_count} />
        <Metric label="Status" value={data.operation.status} />
      </section>

      <section className="overviewGrid">
        <article className="plainPanel">
          <h2>Current Main Attack Path</h2>
          {overview.main_path_summary.length ? (
            <ol className="pathList">
              {overview.main_path_summary.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}
            </ol>
          ) : (
            <p>No main path available</p>
          )}
        </article>
        <article className="plainPanel">
          <h2>Latest Planner Decision</h2>
          <p>{stringValue(overview.latest_decision?.decision_summary) || "No planner decision recorded"}</p>
          <p>{stringValue(overview.latest_decision?.reason) || "No reason recorded"}</p>
        </article>
        <article className="plainPanel">
          <h2>Latest Evidence</h2>
          {overview.latest_evidence ? (
            <button className="listButton" onClick={() => onSelectEvidence(overview.latest_evidence as UnifiedEvidence)}>
              <strong>{overview.latest_evidence.summary}</strong>
              <span>{overview.latest_evidence.source_name} / {overview.latest_evidence.target || "Unknown target"}</span>
            </button>
          ) : (
            <p>No evidence</p>
          )}
        </article>
      </section>
    </div>
  );
}

function GraphPanel({
  title,
  filters,
  graph,
  countLabel,
  onSelectNode,
}: {
  title: string;
  filters: ReactNode;
  graph: GraphState;
  countLabel: string;
  onSelectNode: (node: VisualNode) => void;
}) {
  return (
    <div className="graphPanel">
      <div className="toolbar">
        <h2>{title}</h2>
        {filters}
        <div className="counts">{countLabel}</div>
      </div>
      <div className="graphViewport">
        <CytoscapeGraph graph={graph} onSelectNode={onSelectNode} />
      </div>
    </div>
  );
}

function TimelineView({ events, onSelect }: { events: UnifiedTimelineEvent[]; onSelect: (item: UnifiedTimelineEvent) => void }) {
  const groups = new Map<number, UnifiedTimelineEvent[]>();
  for (const event of events) {
    if (!groups.has(event.round)) groups.set(event.round, []);
    groups.get(event.round)?.push(event);
  }
  return (
    <div className="consoleView">
      {Array.from(groups.entries()).map(([round, items]) => (
        <details key={round} className="roundBlock" open>
          <summary>Round {round}</summary>
          <div className="timelineRail">
            {items.map((event) => (
              <button key={event.id} className={`timelineEvent ${event.status}`} onClick={() => onSelect(event)}>
                <span>{humanPhase(event.phase)}</span>
                <strong>{event.display_name}</strong>
                <small>{event.agent || "System"} / {event.target || "Unknown target"} / {event.created_at || "No timestamp"}</small>
                <p>{event.summary || "No summary available"}</p>
                <small>Evidence: {event.evidence_ids.length ? event.evidence_ids.join(", ") : "No evidence"}</small>
              </button>
            ))}
          </div>
        </details>
      ))}
      {!events.length && <div className="emptyState">No timeline events</div>}
    </div>
  );
}

function EvidenceView({
  items,
  rounds,
  sources,
  targets,
  roundFilter,
  setRoundFilter,
  sourceFilter,
  setSourceFilter,
  targetFilter,
  setTargetFilter,
  onSelect,
}: {
  items: UnifiedEvidence[];
  rounds: string[];
  sources: string[];
  targets: string[];
  roundFilter: string;
  setRoundFilter: (value: string) => void;
  sourceFilter: string;
  setSourceFilter: (value: string) => void;
  targetFilter: string;
  setTargetFilter: (value: string) => void;
  onSelect: (item: UnifiedEvidence) => void;
}) {
  return (
    <div className="consoleView">
      <div className="toolbar inlineToolbar">
        <FilterSelect value={sourceFilter} onChange={setSourceFilter} label="Source" options={sources} />
        <FilterSelect value={roundFilter} onChange={setRoundFilter} label="Round" options={rounds} />
        <FilterSelect value={targetFilter} onChange={setTargetFilter} label="Target" options={targets} />
        <div className="counts">{items.length} evidence items</div>
      </div>
      <div className="evidenceList">
        {items.map((item) => (
          <article key={item.id} className="evidenceItem">
            <button className="listButton" onClick={() => onSelect(item)}>
              <strong>{item.summary || "No summary available"}</strong>
              <span>{item.source_name} / {item.target || "Unknown target"} / Round {item.round ?? "Unknown"}</span>
            </button>
            <dl className="compactDl">
              <div><dt>KG</dt><dd>{item.linked_kg_node_ids.length ? item.linked_kg_node_ids.join(", ") : "No KG links"}</dd></div>
              <div><dt>AG</dt><dd>{item.linked_ag_node_ids.length ? item.linked_ag_node_ids.join(", ") : "No AG links"}</dd></div>
            </dl>
            <details>
              <summary>Structured facts</summary>
              <pre>{JSON.stringify(item.structured_facts, null, 2)}</pre>
            </details>
            <details>
              <summary>Raw output</summary>
              <pre>{maskSensitive(item.raw_output || "No raw output")}</pre>
            </details>
          </article>
        ))}
      </div>
    </div>
  );
}

function AgentTraceView({
  traces,
  rounds,
  roundFilter,
  setRoundFilter,
}: {
  traces: UnifiedVisualization["agent_trace"];
  rounds: string[];
  roundFilter: string;
  setRoundFilter: (value: string) => void;
}) {
  const visible = traces.filter((trace) => !roundFilter || String(trace.round) === roundFilter);
  return (
    <div className="consoleView">
      <div className="toolbar inlineToolbar">
        <FilterSelect value={roundFilter} onChange={setRoundFilter} label="Round" options={rounds} />
      </div>
      {visible.map((trace) => (
        <article key={trace.id} className="tracePanel">
          <div className="plannerHub">
            <strong>PlannerAgent</strong>
            <span>Round {trace.round}</span>
            <p>{trace.planner_decision.decision_summary || "No planner decision recorded"}</p>
            <p>{trace.planner_decision.reason || "No reason recorded"}</p>
            <p>{trace.planner_decision.expected_outcome || "No expected outcome recorded"}</p>
          </div>
          <div className="agentGrid">
            {trace.agent_states.map((agent) => (
              <div key={agent.agent} className={`agentTile ${agent.state}`}>
                <strong>{agent.agent}</strong>
                <span>{agent.state}</span>
                <p>{agent.summary || "No summary available"}</p>
              </div>
            ))}
          </div>
        </article>
      ))}
      {!visible.length && <div className="emptyState">No agent trace</div>}
    </div>
  );
}

function UnifiedDetailPanel({ detail, onClose }: { detail: DetailItem; onClose: () => void }) {
  const payload = detail?.payload;
  const record = (payload || {}) as unknown as Record<string, unknown>;
  const title = payload ? stringValue(record.display_name) || stringValue(record.summary) || "Detail" : "Detail";
  return (
    <aside className="detailPanel">
      <div className="detailHeader">
        <div>
          <h2>{title}</h2>
          <p>{detail?.kind || "No selection"}</p>
        </div>
        <button className="iconButton" onClick={onClose} title="Close details">x</button>
      </div>
      {payload ? (
        <>
          <dl className="summaryList">
            {Object.entries(payload)
              .filter(([key]) => !["metadata", "raw_output", "structured_facts"].includes(key))
              .slice(0, 18)
              .map(([key, value]) => (
                <div key={key}><dt>{key}</dt><dd>{formatValue(value)}</dd></div>
              ))}
          </dl>
          <pre>{JSON.stringify(payload, null, 2)}</pre>
        </>
      ) : (
        <div className="emptyState">Select an item</div>
      )}
    </aside>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="overviewMetric">
      <span>{label}</span>
      <strong>{String(value)}</strong>
    </div>
  );
}

function FilterSelect({ value, onChange, label, options }: { value: string; onChange: (value: string) => void; label: string; options: string[] }) {
  return (
    <select value={value} onChange={(event) => onChange(event.target.value)} aria-label={label}>
      <option value="">All {label.toLowerCase()}</option>
      {options.filter(Boolean).map((option) => <option key={option} value={option}>{option}</option>)}
    </select>
  );
}

function kgGraphState(nodes: UnifiedKgNode[], edges: UnifiedKgEdge[]): GraphState {
  const nodeIds = new Set(nodes.map((node) => node.id));
  return {
    version: 0,
    highlighted: {},
    nodes: Object.fromEntries(nodes.map((node) => [node.id, toVisualNode("kg", node)])),
    edges: Object.fromEntries(edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target)).map((edge) => [edge.id, toVisualEdge("kg", edge)])),
  };
}

function agGraphState(nodes: UnifiedAgNode[], edges: UnifiedAgEdge[]): GraphState {
  const nodeIds = new Set(nodes.map((node) => node.id));
  return {
    version: 0,
    highlighted: {},
    nodes: Object.fromEntries(nodes.map((node) => [node.id, toVisualNode("ag", node)])),
    edges: Object.fromEntries(edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target)).map((edge) => [edge.id, toVisualEdge("ag", edge)])),
  };
}

function toVisualNode(graph: "kg" | "ag", node: UnifiedKgNode | UnifiedAgNode): VisualNode {
  return {
    id: node.id,
    label: node.display_name,
    type: node.type,
    graph,
    status: node.status,
    properties: { ...node, label: node.display_name, display_name: node.display_name },
  };
}

function toVisualEdge(graph: "kg" | "ag", edge: UnifiedKgEdge | UnifiedAgEdge): VisualEdge {
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    label: edge.display_name || edge.type,
    type: edge.type,
    graph,
    properties: { ...edge },
  };
}

function visibleEdgeCount(nodes: Array<{ id: string }>, edges: Array<{ source: string; target: string }>) {
  const ids = new Set(nodes.map((node) => node.id));
  return edges.filter((edge) => ids.has(edge.source) && ids.has(edge.target)).length;
}

function matches(filter: string, value: string | null | undefined) {
  return !filter || value === filter;
}

function unique(values: string[]) {
  return Array.from(new Set(values.filter(Boolean))).sort();
}

function stringValue(value: unknown) {
  if (value === undefined || value === null) return "";
  return String(value);
}

function formatValue(value: unknown) {
  if (Array.isArray(value)) return value.length ? value.map(String).join(", ") : "None";
  if (typeof value === "object" && value !== null) return JSON.stringify(value);
  return value === undefined || value === null || value === "" ? "Unknown" : String(value);
}

function humanPhase(value: string) {
  return value.replace(/_/g, " ");
}

function maskSensitive(value: string) {
  return value
    .replace(/(token|password|private[_ -]?key|secret)(\s*[:=]\s*)([^\s]+)/gi, "$1$2***")
    .replace(/Bearer\s+[A-Za-z0-9._-]+/g, "Bearer ***");
}
