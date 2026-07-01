import { Activity, Database, Download, ListChecks, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { fetchVisualization, listOperations } from "./api";
import { CytoscapeGraph } from "./components/CytoscapeGraph";
import type { GraphState } from "./graphState";
import type {
  OperationSummary,
  UnifiedAgNode,
  UnifiedKgNode,
  UnifiedServiceMatrixRow,
  UnifiedTimelineEvent,
  UnifiedVisualization,
  VisualEdge,
  VisualNode,
} from "./types";

type ViewId = "overview" | "assets" | "cycles";
type AssetRow = UnifiedServiceMatrixRow & { vulnerability_tags: string[] };
type CycleRow = {
  cycle_index: number;
  title: string;
  agent: string;
  target: string;
  intent: string;
  result_summary: string;
  status: string;
  asset_updates: string[];
  service_updates: string[];
  vulnerability_tags: string[];
};

const views: Array<{ id: ViewId; label: string; icon: typeof Activity }> = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "assets", label: "Assets", icon: Database },
  { id: "cycles", label: "Attack Cycles", icon: ListChecks },
];

export default function App() {
  const [operations, setOperations] = useState<OperationSummary[]>([]);
  const [operationId, setOperationId] = useState("");
  const [visualization, setVisualization] = useState<UnifiedVisualization | null>(null);
  const [activeView, setActiveView] = useState<ViewId>("overview");
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    listOperations().then((items) => {
      const sorted = [...items].sort((left, right) => Date.parse(right.last_updated || "") - Date.parse(left.last_updated || ""));
      setOperations(sorted);
      setOperationId((current) => current || sorted[0]?.operation_id || "");
    });
  }, []);

  useEffect(() => {
    if (operationId) refreshVisualization(operationId);
  }, [operationId]);

  const refreshVisualization = async (id = operationId) => {
    if (!id) return;
    setLoading(true);
    setLoadError("");
    try {
      setVisualization(await fetchVisualization(id));
    } catch (error) {
      setVisualization(null);
      setLoadError(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  };

  const exportJson = () => {
    const blob = new Blob([JSON.stringify(visualization || {}, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${operationId || "aegra"}-visualization.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const assets = useMemo(() => buildAssets(visualization), [visualization]);
  const cycles = useMemo(() => buildCycles(visualization), [visualization]);
  const riskTags = useMemo(() => collectRiskTags(visualization, assets, cycles), [visualization, assets, cycles]);

  return (
    <div className="appShell">
      <header className="topbar">
        <div>
          <h1>Aegra Visualization</h1>
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

      <main className="focusedWorkspace">
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
              {activeView === "overview" && <OverviewView data={visualization} assets={assets} cycles={cycles} riskTags={riskTags} />}
              {activeView === "assets" && <AssetsView assets={assets} />}
              {activeView === "cycles" && <AttackCyclesView cycles={cycles} />}
            </>
          ) : (
            <div className="emptyState">{loadError ? `Visualization failed to load: ${loadError}` : "No visualization data"}</div>
          )}
        </section>
      </main>
    </div>
  );
}

function OverviewView({
  data,
  assets,
  cycles,
  riskTags,
}: {
  data: UnifiedVisualization;
  assets: AssetRow[];
  cycles: CycleRow[];
  riskTags: string[];
}) {
  const overview = data.overview;
  const latest = overview.last_cycle_summary || cycles[cycles.length - 1];
  const openPorts = overview.open_ports ?? assets.reduce((total, asset) => total + asset.services.length, 0);
  const serviceCount = overview.identified_services ?? overview.service_count ?? assets.reduce((total, asset) => total + asset.services.length, 0);
  const target = formatTarget(data.operation.target_scope);

  return (
    <div className="consoleView">
      <section className="overviewHeader">
        <div>
          <span>Operation</span>
          <strong>{data.operation.id}</strong>
        </div>
        <div>
          <span>Target</span>
          <strong>{target}</strong>
        </div>
        <div>
          <span>Status</span>
          <strong>{data.operation.status}</strong>
        </div>
      </section>

      <section className="metricGrid">
        <Metric label="Cycle Count" value={cycles.length || data.operation.current_round} />
        <Metric label="Discovered Hosts" value={overview.discovered_hosts ?? assets.length} />
        <Metric label="Open Ports" value={openPorts} />
        <Metric label="Identified Services" value={serviceCount} />
        <Metric label="Detected Vulnerability Tags / Risk Tags" value={riskTags.length} />
      </section>

      <section className="overviewGrid twoColumn">
        <article className="plainPanel">
          <h2>Latest</h2>
          {latest ? (
            <>
              <strong>{latest.title || `Cycle ${latest.cycle_index}`}</strong>
              <p>{("summary" in latest ? latest.summary : latest.result_summary) || "No summary available"}</p>
            </>
          ) : (
            <p>No cycle has been recorded</p>
          )}
        </article>
        <article className="plainPanel">
          <h2>Risk Tags</h2>
          <TagList tags={riskTags} />
        </article>
      </section>

      <section className="plainPanel relationPanel">
        <h2>Target to Attack Cycle Map</h2>
        <div className="relationshipGraph">
          <CytoscapeGraph graph={relationshipGraph(data, assets, cycles, riskTags, target)} onSelectNode={() => undefined} />
        </div>
      </section>
    </div>
  );
}

function AssetsView({ assets }: { assets: AssetRow[] }) {
  const [selectedId, setSelectedId] = useState("");
  const selected = assets.find((asset) => asset.host_id === selectedId) || assets[0];

  useEffect(() => {
    if (!selectedId && assets[0]) setSelectedId(assets[0].host_id);
    if (selectedId && assets.length && !assets.some((asset) => asset.host_id === selectedId)) setSelectedId(assets[0].host_id);
  }, [assets, selectedId]);

  return (
    <div className="consoleView">
      <div className="assetLayout">
        <div className="assetTableWrap">
          <table className="assetTable">
            <thead>
              <tr>
                <th>Host</th>
                <th>Hostname</th>
                <th>Role / Guess</th>
                <th>Open Ports</th>
                <th>Services</th>
                <th>Vulnerability Tags</th>
                <th>Last Seen</th>
              </tr>
            </thead>
            <tbody>
              {assets.map((asset) => (
                <tr key={asset.host_id} className={selected?.host_id === asset.host_id ? "selected" : ""} onClick={() => setSelectedId(asset.host_id)}>
                  <td>{asset.host}</td>
                  <td>{asset.hostname || "unknown"}</td>
                  <td>{asset.role_guess || asset.role || "unknown"}</td>
                  <td>{asset.open_ports ?? asset.services.length}</td>
                  <td>{unique(asset.services.map((service) => service.name)).join(", ") || "none"}</td>
                  <td><TagList tags={asset.vulnerability_tags} compact /></td>
                  <td>{formatCycleSeen(asset.last_seen)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!assets.length && <div className="emptyState compact">No assets discovered</div>}
        </div>

        {selected && <AssetDetail asset={selected} />}
      </div>
    </div>
  );
}

function AssetDetail({ asset }: { asset: AssetRow }) {
  return (
    <aside className="assetDetail">
      <h2>Host: {asset.host}</h2>
      <dl className="compactDl wide">
        <div><dt>Hostname</dt><dd>{asset.hostname || "unknown"}</dd></div>
        <div><dt>Role Guess</dt><dd>{asset.role_guess || asset.role || "unknown"}</dd></div>
        <div><dt>Confidence</dt><dd>{confidenceLabel(asset.confidence)}</dd></div>
        <div><dt>First Seen</dt><dd>{formatCycleSeen(asset.first_seen)}</dd></div>
        <div><dt>Last Updated</dt><dd>{formatCycleSeen(asset.last_seen)}</dd></div>
      </dl>

      <h3>Ports & Services</h3>
      <div className="serviceList">
        {asset.services.map((service) => (
          <div key={service.service_id} className="serviceRow">
            <code>{service.port || "?"}/{service.protocol || "tcp"}</code>
            <strong>{service.name || "unknown"}</strong>
            <span>{service.product || "unknown product"}{service.version ? ` ${service.version}` : ""}</span>
            <em>{service.status}</em>
          </div>
        ))}
        {!asset.services.length && <p>No open service recorded</p>}
      </div>

      <h3>Vulnerability / Risk Tags</h3>
      <TagList tags={asset.vulnerability_tags} />

      <h3>Related Cycles</h3>
      <ul className="cycleMiniList">
        {(asset.related_cycles || []).map((cycle) => (
          <li key={`${cycle.cycle_index}:${cycle.summary}`}>
            <strong>Cycle {cycle.cycle_index}</strong>
            <span>{cycle.summary || cycle.title}</span>
          </li>
        ))}
        {!(asset.related_cycles || []).length && <li><span>No related cycle recorded</span></li>}
      </ul>
    </aside>
  );
}

function AttackCyclesView({ cycles }: { cycles: CycleRow[] }) {
  return (
    <div className="consoleView">
      <div className="cycleCards">
        {cycles.map((cycle) => (
          <article key={cycle.cycle_index} className="cycleSummaryCard">
            <header>
              <div>
                <span>Cycle {cycle.cycle_index}</span>
                <h2>{cycle.title}</h2>
              </div>
              <strong className={`nodeStatus ${cycle.status}`}>{cycle.status}</strong>
            </header>
            <dl className="cycleFields">
              <div><dt>agent</dt><dd>{cycle.agent}</dd></div>
              <div><dt>target</dt><dd>{cycle.target}</dd></div>
              <div><dt>intent</dt><dd>{cycle.intent}</dd></div>
              <div><dt>result_summary</dt><dd>{cycle.result_summary}</dd></div>
              <div><dt>asset_updates</dt><dd>{cycle.asset_updates.join(", ") || "none"}</dd></div>
              <div><dt>service_updates</dt><dd>{cycle.service_updates.join(", ") || "none"}</dd></div>
              <div><dt>vulnerability_tags</dt><dd><TagList tags={cycle.vulnerability_tags} compact /></dd></div>
            </dl>
          </article>
        ))}
        {!cycles.length && <div className="emptyState">No attack cycles recorded</div>}
      </div>
    </div>
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

function TagList({ tags, compact = false }: { tags: string[]; compact?: boolean }) {
  const visible = unique(tags).filter(Boolean);
  if (!visible.length) return <span className="muted">none</span>;
  return (
    <div className={compact ? "tagList compact" : "tagList"}>
      {visible.map((tag) => <span key={tag}>{tag}</span>)}
    </div>
  );
}

function buildAssets(data: UnifiedVisualization | null): AssetRow[] {
  if (!data) return [];
  if (data.service_matrix?.length) {
    return data.service_matrix.map((row) => ({
      ...row,
      vulnerability_tags: unique(row.vulnerability_tags || row.services.flatMap((service) => service.vulnerability_tags || [])),
    }));
  }
  const hosts = data.kg.nodes.filter((node) => node.type === "Host");
  return hosts.map((host) => {
    const services = relatedServices(host, data);
    return {
      host_id: host.id,
      host: host.display_name,
      hostname: firstString(readMeta(host).hostname, readMeta(host).name, "unknown"),
      role: host.role || "unknown",
      role_guess: host.role || "unknown",
      status: host.status,
      confidence: host.confidence,
      services,
      open_ports: services.length,
      vulnerability_tags: collectNodeTags([host, ...services.map((service) => data.kg.nodes.find((node) => node.id === service.service_id)).filter(Boolean) as UnifiedKgNode[]]),
      first_seen: host.created_at,
      last_seen: host.updated_at || String(host.round || ""),
      related_cycles: relatedCycles(host, data.ag.nodes),
      evidence_ids: host.evidence_ids,
    };
  });
}

function relatedServices(host: UnifiedKgNode, data: UnifiedVisualization): AssetRow["services"] {
  const relatedIds = new Set<string>();
  for (const edge of data.kg.edges) {
    if (edge.source === host.id) relatedIds.add(edge.target);
    if (edge.target === host.id) relatedIds.add(edge.source);
  }
  return data.kg.nodes
    .filter((node) => node.type === "Service")
    .filter((node) => relatedIds.has(node.id) || firstString(readMeta(node).host, node.target) === host.display_name || firstString(readMeta(node).host) === host.id)
    .map((node) => {
      const meta = readMeta(node);
      return {
        service_id: node.id,
        name: firstString(meta.service_name, meta.service, node.display_name, "unknown"),
        port: nullableString(meta.port),
        protocol: nullableString(meta.protocol) || "tcp",
        product: nullableString(meta.product || meta.product_name || meta.technology),
        version: nullableString(meta.version || meta.product_version),
        status: node.status,
        confidence: node.confidence,
        evidence_ids: node.evidence_ids,
        summary: node.summary,
        vulnerability_tags: tagsFromRecord(meta),
      };
    });
}

function buildCycles(data: UnifiedVisualization | null): CycleRow[] {
  if (!data) return [];
  const groups = new Map<number, Array<UnifiedAgNode | UnifiedTimelineEvent>>();
  for (const node of data.ag.nodes) {
    const index = node.cycle_index || node.round || 0;
    if (!groups.has(index)) groups.set(index, []);
    groups.get(index)?.push(node);
  }
  for (const event of data.timeline) {
    const index = event.cycle_index || event.round || 0;
    if (!groups.has(index)) groups.set(index, []);
    groups.get(index)?.push(event);
  }

  return Array.from(groups.entries())
    .filter(([index]) => index > 0)
    .sort(([left], [right]) => left - right)
    .map(([cycleIndex, items]) => {
      const records = items.map((item) => item as unknown as Record<string, unknown>);
      const selected = records.find((item) => firstString(item.result_summary)) || records[records.length - 1] || {};
      const title = firstString(selected.title, selected.display_name, selected.intent, selected.action_summary, `Cycle ${cycleIndex}`);
      return {
        cycle_index: cycleIndex,
        title: title.startsWith("Cycle ") ? title : `Cycle ${cycleIndex} - ${title}`,
        agent: firstString(selected.agent, records.map((item) => item.agent).find(Boolean), "unknown"),
        target: firstString(selected.target_summary, selected.target, records.map((item) => item.target).find(Boolean), "unknown"),
        intent: firstString(selected.intent, selected.action_summary, selected.summary, "No intent recorded"),
        result_summary: firstString(selected.result_summary, selected.summary, "No result summary recorded"),
        status: firstString(selected.status, records.map((item) => item.status).find(Boolean), "unknown"),
        asset_updates: unique(records.flatMap((item) => extractUpdates(item, "Host"))),
        service_updates: unique(records.flatMap((item) => extractUpdates(item, "Service"))),
        vulnerability_tags: unique(records.flatMap((item) => tagsFromRecord(asRecord(item.metadata)))),
      };
    });
}

function relationshipGraph(data: UnifiedVisualization, assets: AssetRow[], cycles: CycleRow[], riskTags: string[], target: string): GraphState {
  const nodes: Record<string, VisualNode> = {};
  const edges: Record<string, VisualEdge> = {};
  const addNode = (id: string, label: string, type: string) => {
    nodes[id] = { id, label, type, graph: "kg", status: "observed", properties: { label, display_name: label } };
  };
  const addEdge = (source: string, targetId: string, label: string) => {
    const id = `${source}->${targetId}:${label}`;
    edges[id] = { id, source, target: targetId, label, type: label, graph: "kg", properties: {} };
  };

  addNode("target", target || data.operation.id, "Target");
  for (const asset of assets) {
    addNode(asset.host_id, asset.host, "Host");
    addEdge("target", asset.host_id, "contains");
    for (const service of asset.services) {
      addNode(service.service_id, serviceLabel(service), "Service");
      addEdge(asset.host_id, service.service_id, "exposes");
    }
  }
  for (const tag of riskTags.slice(0, 24)) {
    const id = `tag:${tag}`;
    addNode(id, tag, "Tag");
    for (const asset of assets.filter((item) => item.vulnerability_tags.includes(tag))) {
      addEdge(asset.host_id, id, "tagged");
      for (const service of asset.services.filter((item) => (item.vulnerability_tags || []).includes(tag))) {
        addEdge(service.service_id, id, "tagged");
      }
    }
  }
  for (const cycle of cycles.slice(-12)) {
    const id = `cycle:${cycle.cycle_index}`;
    addNode(id, `Cycle ${cycle.cycle_index}`, "AttackCycle");
    const matchedAssets = assets.filter((asset) => cycleMentionsAsset(cycle, asset));
    for (const asset of matchedAssets.length ? matchedAssets : assets.slice(0, 1)) {
      addEdge(asset.host_id, id, "observed_in");
    }
    for (const tag of cycle.vulnerability_tags) {
      if (nodes[`tag:${tag}`]) addEdge(`tag:${tag}`, id, "used_by");
    }
  }

  return { version: 0, highlighted: {}, nodes, edges };
}

function collectRiskTags(data: UnifiedVisualization | null, assets: AssetRow[], cycles: CycleRow[]) {
  return unique([
    ...(data?.risk_tags || []),
    ...(data?.overview.vulnerability_tags || []),
    ...assets.flatMap((asset) => asset.vulnerability_tags),
    ...cycles.flatMap((cycle) => cycle.vulnerability_tags),
  ]);
}

function collectNodeTags(nodes: UnifiedKgNode[]) {
  return unique(nodes.flatMap((node) => tagsFromRecord(readMeta(node))));
}

function relatedCycles(host: UnifiedKgNode, nodes: UnifiedAgNode[]) {
  const refs = [host.id, host.display_name, host.target].filter(Boolean).map(String);
  return nodes
    .filter((node) => refs.some((ref) => JSON.stringify(node).includes(ref)))
    .map((node) => ({
      cycle_index: node.cycle_index || node.round,
      title: `Cycle ${node.cycle_index || node.round} - ${node.intent || node.action_summary}`,
      summary: node.result_summary || node.action_summary,
      agent: node.agent,
      status: node.status,
    }));
}

function extractUpdates(item: Record<string, unknown>, type: "Host" | "Service") {
  const values: string[] = [];
  const delta = asRecord(item.kg_delta);
  for (const update of [...asArray(delta.updates), ...asArray(item.kg_updates), ...asArray(item.kg_node_ids)]) {
    if (typeof update === "string") values.push(update);
    if (isRecord(update) && firstString(update.type, update.entity_type).includes(type)) {
      values.push(firstString(update.id, update.ref_id, update.label));
    }
  }
  return values.filter(Boolean);
}

function cycleMentionsAsset(cycle: CycleRow, asset: AssetRow) {
  const text = JSON.stringify(cycle);
  return [asset.host_id, asset.host, asset.hostname, ...asset.services.map((service) => service.service_id)].filter(Boolean).some((ref) => text.includes(String(ref)));
}

function serviceLabel(service: AssetRow["services"][number]) {
  const port = service.port ? `${service.port}/${service.protocol || "tcp"}` : service.protocol || "";
  const product = [service.product, service.version].filter(Boolean).join(" ");
  return [port, service.name, product].filter(Boolean).join(" ");
}

function formatTarget(value: unknown[]) {
  if (!value?.length) return "unknown";
  return value.map((item) => typeof item === "string" ? item : JSON.stringify(item)).join(", ");
}

function formatCycleSeen(value: unknown) {
  const text = firstString(value);
  if (!text) return "unknown";
  if (/^\d+$/.test(text)) return `Cycle ${text}`;
  return text;
}

function confidenceLabel(value: number) {
  if (value >= 0.75) return "high";
  if (value >= 0.4) return "medium";
  if (value > 0) return "low";
  return "unknown";
}

function tagsFromRecord(record: Record<string, unknown>): string[] {
  const explicitKeys = ["vulnerability_tags", "risk_tags", "risk_labels", "vulnerability_labels", "detected_tags", "candidate_tags"];
  const parsedKeys = [...explicitKeys, "tags", "labels"];
  const tags = explicitKeys.flatMap((key) => normalizeTags(record[key]));
  const parsed = asRecord(record.parsed || record.parsed_output || record.tool_result);
  return unique([...tags, ...parsedKeys.flatMap((key) => normalizeTags(parsed[key]))]);
}

function normalizeTags(value: unknown): string[] {
  const items = typeof value === "string" ? value.replace(/;/g, ",").split(",") : asArray(value);
  return items
    .map((item) => isRecord(item) ? firstString(item.tag, item.label, item.name, item.kind, item.type) : firstString(item))
    .map((item) => item.trim().toLowerCase().replace(/[-\s]+/g, "_"))
    .filter((item) => item && !["none", "unknown", "n/a"].includes(item));
}

function readMeta(node: UnifiedKgNode) {
  return asRecord(node.metadata);
}

function nullableString(value: unknown): string | null {
  return firstString(value) || null;
}

function unique(values: string[]) {
  return Array.from(new Set(values.filter(Boolean))).sort();
}

function firstString(...values: unknown[]) {
  for (const value of values.flat()) {
    if (value === undefined || value === null || value === "") continue;
    return String(value);
  }
  return "";
}

function asRecord(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

function asArray(value: unknown): unknown[] {
  if (value === undefined || value === null) return [];
  return Array.isArray(value) ? value : [value];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
