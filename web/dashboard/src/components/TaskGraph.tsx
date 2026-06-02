import { Background, Controls, Handle, Position, ReactFlow, type Edge, type Node } from "@xyflow/react";
import type { GraphState } from "../graphState";
import type { VisualNode } from "../types";

interface Props {
  graph: GraphState;
  onSelectNode: (node: VisualNode) => void;
}

export function LegacyTaskGraph({ graph, onSelectNode }: Props) {
  const visualNodes = Object.values(graph.nodes);
  const nodes: Node[] = visualNodes.map((node, index) => ({
    id: node.id,
    type: "taskNode",
    position: { x: 80 + (index % 4) * 260, y: 70 + Math.floor(index / 4) * 125 },
    data: { node, fresh: graph.highlighted[node.id] > Date.now() - 4500 },
  }));
  const edges: Edge[] = Object.values(graph.edges).map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    label: edge.label || edge.type || "",
    animated: graph.highlighted[edge.id] > Date.now() - 4500,
  }));

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={{ taskNode: TaskNode }}
      fitView
      minZoom={0.2}
      onNodeClick={(_, node) => onSelectNode((node.data as { node: VisualNode }).node)}
    >
      <Background />
      <Controls />
    </ReactFlow>
  );
}

function TaskNode({ data }: { data: { node: VisualNode; fresh: boolean } }) {
  const node = data.node;
  const status = String(node.status || node.properties.status || "draft").toLowerCase();
  return (
    <div className={`taskNode ${status} ${data.fresh ? "fresh" : ""}`}>
      <Handle type="target" position={Position.Top} />
      <div className="taskTitle">{node.label}</div>
      <div className="taskMeta">{node.type || "Task"} · {status}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
