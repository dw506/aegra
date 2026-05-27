import { Download, RefreshCw, Wifi, WifiOff } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchSnapshot, graphWsUrl, listOperations } from "./api";
import { CytoscapeGraph } from "./components/CytoscapeGraph";
import { NodeDetailPanel } from "./components/NodeDetailPanel";
import { TaskGraph } from "./components/TaskGraph";
import { applyDelta, emptyDashboardGraphs, filteredGraph, snapshotToGraphs, type DashboardGraphs } from "./graphState";
import type { GraphName, OperationSummary, ServerMessage, VisualNode } from "./types";

const tabs: { id: GraphName; label: string }[] = [
  { id: "kg", label: "KG" },
  { id: "ag", label: "AG" },
  { id: "tg", label: "TG" },
  { id: "runtime", label: "Runtime" },
];

export default function App() {
  const [operations, setOperations] = useState<OperationSummary[]>([]);
  const [operationId, setOperationId] = useState("");
  const [graphs, setGraphs] = useState<DashboardGraphs>(() => emptyDashboardGraphs());
  const [activeGraph, setActiveGraph] = useState<GraphName>("kg");
  const [selectedNode, setSelectedNode] = useState<VisualNode | null>(null);
  const [connected, setConnected] = useState(false);
  const [typeFilter, setTypeFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const reconnectAttempt = useRef(0);

  useEffect(() => {
    listOperations().then((items) => {
      setOperations(items);
      setOperationId((current) => current || items[0]?.operation_id || "");
    });
  }, []);

  useEffect(() => {
    if (!operationId) return;
    let stopped = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;

    const connect = async () => {
      try {
        const snapshot = await fetchSnapshot(operationId);
        if (!stopped) setGraphs(snapshotToGraphs(snapshot));
      } catch {
        if (!stopped) setGraphs(emptyDashboardGraphs());
      }

      socket = new WebSocket(graphWsUrl(operationId));
      socket.onopen = () => {
        reconnectAttempt.current = 0;
        setConnected(true);
      };
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data) as ServerMessage;
        if (message.type === "graph_snapshot") {
          setGraphs(snapshotToGraphs(message));
          return;
        }
        setGraphs((current) => ({
          ...current,
          [message.graph]: applyDelta(current[message.graph], message),
        }));
      };
      socket.onclose = () => {
        setConnected(false);
        if (stopped) return;
        const delay = Math.min(15000, 750 * 2 ** reconnectAttempt.current++);
        reconnectTimer = window.setTimeout(connect, delay);
      };
    };

    connect();
    return () => {
      stopped = true;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [operationId]);

  const activeState = graphs[activeGraph];
  const visibleState = useMemo(
    () => filteredGraph(activeState, typeFilter, statusFilter),
    [activeState, typeFilter, statusFilter],
  );
  const typeOptions = useMemo(() => unique(Object.values(activeState.nodes).map((node) => node.type || "")), [activeState]);
  const statusOptions = useMemo(() => unique(Object.values(activeState.nodes).map((node) => node.status || "")), [activeState]);

  const exportJson = () => {
    const blob = new Blob([JSON.stringify(activeState, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${operationId || "aegra"}-${activeGraph}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="appShell">
      <header className="topbar">
        <div>
          <h1>Aegra Graph Dashboard</h1>
          <p>{operationId || "No operation selected"}</p>
        </div>
        <div className="topbarControls">
          <select value={operationId} onChange={(event) => setOperationId(event.target.value)} aria-label="Operation">
            {operations.map((operation) => (
              <option key={operation.operation_id} value={operation.operation_id}>
                {operation.operation_id}
              </option>
            ))}
          </select>
          <button className="iconButton" onClick={() => operationId && fetchSnapshot(operationId).then((snapshot) => setGraphs(snapshotToGraphs(snapshot)))} title="Refresh snapshot">
            <RefreshCw size={18} />
          </button>
          <span className={`connection ${connected ? "online" : "offline"}`}>
            {connected ? <Wifi size={17} /> : <WifiOff size={17} />}
            {connected ? "Live" : "Offline"}
          </span>
        </div>
      </header>

      <main className="workspace">
        <section className="graphSurface">
          <div className="tabs">
            {tabs.map((tab) => (
              <button key={tab.id} className={activeGraph === tab.id ? "active" : ""} onClick={() => setActiveGraph(tab.id)}>
                {tab.label}
                <span>{graphs[tab.id].version}</span>
              </button>
            ))}
          </div>

          <div className="toolbar">
            <select value={typeFilter} onChange={(event) => setTypeFilter(event.target.value)} aria-label="Type filter">
              <option value="">All types</option>
              {typeOptions.map((value) => <option key={value} value={value}>{value}</option>)}
            </select>
            <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)} aria-label="Status filter">
              <option value="">All statuses</option>
              {statusOptions.map((value) => <option key={value} value={value}>{value}</option>)}
            </select>
            <button className="iconTextButton" onClick={exportJson}>
              <Download size={17} />
              JSON
            </button>
            <div className="counts">{Object.keys(visibleState.nodes).length} nodes / {Object.keys(visibleState.edges).length} edges</div>
          </div>

          <div className="graphViewport">
            {activeGraph === "tg" ? (
              <TaskGraph graph={visibleState} onSelectNode={setSelectedNode} />
            ) : (
              <CytoscapeGraph graph={visibleState} onSelectNode={setSelectedNode} />
            )}
          </div>
        </section>

        <NodeDetailPanel node={selectedNode} onClose={() => setSelectedNode(null)} />
      </main>
    </div>
  );
}

function unique(values: string[]) {
  return Array.from(new Set(values.filter(Boolean))).sort();
}
