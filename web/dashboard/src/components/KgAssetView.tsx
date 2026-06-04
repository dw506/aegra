import { Database, FileText, Server } from "lucide-react";
import type { ReactNode } from "react";
import type { GraphState } from "../graphState";
import { buildDisplayName, buildKgAssetSummaries } from "../graphTransforms";
import type { VisualNode } from "../types";
import { CytoscapeGraph } from "./CytoscapeGraph";

interface Props {
  kg: GraphState;
  ag: GraphState;
  selectedNode: VisualNode | null;
  onSelectNode: (node: VisualNode) => void;
}

export function KgAssetView({ kg, ag, selectedNode, onSelectNode }: Props) {
  const assets = buildKgAssetSummaries(kg, ag);
  const selectedAsset = assets.find((asset) => asset.host.id === selectedNode?.id) || assets[0];

  return (
    <div className="kgAssetView">
      <section className="kgAssetStats" aria-label="KG asset stats">
        <Metric label="Hosts" value={assets.length} />
        <Metric label="Services" value={assets.reduce((total, asset) => total + asset.services.length, 0)} />
        <Metric label="Observations" value={assets.reduce((total, asset) => total + asset.observations.length, 0)} />
        <Metric label="Evidence" value={assets.reduce((total, asset) => total + asset.evidence.length, 0)} />
        <Metric label="Findings" value={assets.reduce((total, asset) => total + asset.findings.length, 0)} />
      </section>

      <div className="kgAssetGrid">
        <aside className="assetList" aria-label="Host assets">
          {assets.length ? assets.map((asset) => (
            <button
              key={asset.host.id}
              className={`assetListItem ${selectedAsset?.host.id === asset.host.id ? "selected" : ""}`}
              onClick={() => onSelectNode(asset.host)}
            >
              <strong>{asset.title}</strong>
              <span>Scope: {asset.scope}</span>
              <span>Goal target: {asset.goalTarget ? "yes" : "no"}</span>
              <small>{asset.services.length} services / {asset.evidence.length} evidence</small>
            </button>
          )) : <div className="emptyState compact">No host assets</div>}
        </aside>

        <section className="kgGraphPane">
          <CytoscapeGraph graph={kg} onSelectNode={onSelectNode} />
        </section>

        <aside className="assetDetailPane">
          {selectedAsset ? (
            <>
              <h2>{selectedAsset.title}</h2>
              <dl className="summaryList">
                <div><dt>Scope</dt><dd>{selectedAsset.scope}</dd></div>
                <div><dt>Goal</dt><dd>{selectedAsset.goalTarget ? "yes" : "no"}</dd></div>
                <div><dt>Latest cycle</dt><dd>{selectedAsset.latestCycleIndex}</dd></div>
              </dl>
              <NodeList title="Services" icon={<Server size={15} />} nodes={selectedAsset.services} onSelectNode={onSelectNode} />
              <NodeList title="Observations" icon={<FileText size={15} />} nodes={selectedAsset.observations} onSelectNode={onSelectNode} />
              <NodeList title="Evidence" icon={<Database size={15} />} nodes={selectedAsset.evidence} onSelectNode={onSelectNode} />
              <NodeList title="Findings" icon={<FileText size={15} />} nodes={selectedAsset.findings} onSelectNode={onSelectNode} />
            </>
          ) : (
            <div className="emptyState">Select a host asset</div>
          )}
        </aside>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="overviewMetric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function NodeList({
  title,
  icon,
  nodes,
  onSelectNode,
}: {
  title: string;
  icon: ReactNode;
  nodes: VisualNode[];
  onSelectNode: (node: VisualNode) => void;
}) {
  return (
    <section className="assetNodeList">
      <h3>{icon}{title}<span>{nodes.length}</span></h3>
      {nodes.length ? nodes.map((node) => (
        <button key={node.id} onClick={() => onSelectNode(node)}>
          {buildDisplayName(node)}
        </button>
      )) : <p>None</p>}
    </section>
  );
}
