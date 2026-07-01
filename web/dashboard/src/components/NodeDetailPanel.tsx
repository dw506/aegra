import { X } from "lucide-react";
import { buildDisplayName, inferStepOrder, nodeType, readValue } from "../graphTransforms";
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
          <h2>{node ? buildDisplayName(node) : "Node"}</h2>
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
            <div><dt>Raw type</dt><dd>{nodeType(node) || "n/a"}</dd></div>
            <div><dt>Status</dt><dd>{node.status || "n/a"}</dd></div>
            <div><dt>Cycle</dt><dd>{String(readValue(node, "cycle_index") || "n/a")}</dd></div>
            <div><dt>Step</dt><dd>{node.graph === "ag" ? inferStepOrder(node) : "n/a"}</dd></div>
            <div><dt>Objective</dt><dd>{String(readValue(node, "objective") || readValue(node, "planner_objective") || "n/a")}</dd></div>
            <div><dt>Agent</dt><dd>{String(readValue(node, "selected_agent") || readValue(node, "agent_name") || "n/a")}</dd></div>
            <div><dt>Tool</dt><dd>{String(readValue(node, "tool_name") || "n/a")}</dd></div>
            <div><dt>Summary</dt><dd>{String(readValue(node, "summary") || "n/a")}</dd></div>
            <div><dt>Stop</dt><dd>{String(readValue(node, "stop_reason") || readValue(node, "reason") || "n/a")}</dd></div>
            <div><dt>Evidence</dt><dd>{formatList(readValue(node, "evidence_ids") || readValue(node, "evidence_refs"))}</dd></div>
            <div><dt>KG refs</dt><dd>{formatKgRefs(node)}</dd></div>
          </dl>
          <pre>{JSON.stringify({ metadata: readValue(node, "metadata"), properties: node.properties }, null, 2)}</pre>
        </>
      ) : (
        <div className="emptyState">Select a node</div>
      )}
    </aside>
  );
}

function formatList(value: unknown) {
  if (Array.isArray(value)) return value.length ? value.map(String).join(", ") : "n/a";
  return value ? String(value) : "n/a";
}

function formatKgRefs(node: VisualNode) {
  const refs = readValue(node, "refs") || readValue(node, "source_refs") || readValue(node, "subject_refs") || readValue(node, "created_from");
  if (!Array.isArray(refs)) return "n/a";
  const ids = refs.map((ref) => {
    if (typeof ref === "string") return ref;
    if (typeof ref === "object" && ref !== null) return String((ref as Record<string, unknown>).entity_id || (ref as Record<string, unknown>).id || "");
    return "";
  }).filter(Boolean);
  return ids.length ? ids.join(", ") : "n/a";
}
