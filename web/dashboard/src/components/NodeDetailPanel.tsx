import { X } from "lucide-react";
import type { VisualNode } from "../types";

interface Props {
  node: VisualNode | null;
  onClose: () => void;
}

export function NodeDetailPanel({ node, onClose }: Props) {
  return (
    <aside className={`detailPanel ${node ? "open" : ""}`}>
      <div className="detailHeader">
        <div>
          <h2>{node?.label || "Node"}</h2>
          <p>{node?.type || node?.graph || ""}</p>
        </div>
        <button className="iconButton" onClick={onClose} title="Close details">
          <X size={18} />
        </button>
      </div>
      {node ? (
        <>
          <dl className="summaryList">
            <div><dt>ID</dt><dd>{node.id}</dd></div>
            <div><dt>Graph</dt><dd>{node.graph}</dd></div>
            <div><dt>Status</dt><dd>{node.status || "n/a"}</dd></div>
          </dl>
          <pre>{JSON.stringify(node.properties, null, 2)}</pre>
        </>
      ) : (
        <div className="emptyState">Select a node</div>
      )}
    </aside>
  );
}
