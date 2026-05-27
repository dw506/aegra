import cytoscape, { type Core } from "cytoscape";
import { useEffect, useRef } from "react";
import type { GraphState } from "../graphState";
import type { VisualNode } from "../types";

interface Props {
  graph: GraphState;
  onSelectNode: (node: VisualNode) => void;
}

export function CytoscapeGraph({ graph, onSelectNode }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      elements: [],
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            "border-color": "data(border)",
            "border-width": "2px",
            color: "#1f2937",
            label: "data(label)",
            "font-size": "11px",
            "text-wrap": "wrap",
            "text-max-width": "120px",
            "text-valign": "bottom",
            "text-margin-y": 8,
            width: "36px",
            height: "36px",
          },
        },
        {
          selector: "edge",
          style: {
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "line-color": "#9ca3af",
            "target-arrow-color": "#9ca3af",
            label: "data(label)",
            "font-size": "9px",
            color: "#4b5563",
          },
        },
        {
          selector: ".fresh",
          style: {
            "border-color": "#f59e0b",
            "line-color": "#f59e0b",
            "target-arrow-color": "#f59e0b",
          },
        },
      ],
      layout: { name: "cose", animate: false, padding: 32 },
    });
    cy.on("tap", "node", (event) => {
      const id = event.target.id();
      const node = graph.nodes[id];
      if (node) onSelectNode(node);
    });
    cyRef.current = cy;
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().remove();
    const freshCutoff = Date.now() - 4500;
    cy.add([
      ...Object.values(graph.nodes).map((node) => ({
        group: "nodes" as const,
        data: {
          id: node.id,
          label: node.label,
          color: colorFor(node.type || node.status || ""),
          border: graph.highlighted[node.id] > freshCutoff ? "#f59e0b" : "#ffffff",
        },
        classes: graph.highlighted[node.id] > freshCutoff ? "fresh" : "",
      })),
      ...Object.values(graph.edges).map((edge) => ({
        group: "edges" as const,
        data: { id: edge.id, source: edge.source, target: edge.target, label: edge.label || edge.type || "" },
        classes: graph.highlighted[edge.id] > freshCutoff ? "fresh" : "",
      })),
    ]);
    cy.layout({ name: "cose", animate: false, padding: 34 }).run();
    cy.fit(undefined, 36);
  }, [graph]);

  return <div className="cyGraph" ref={containerRef} />;
}

function colorFor(value: string) {
  if (value.includes("HOST") || value.includes("Asset")) return "#2563eb";
  if (value.includes("SERVICE")) return "#0f766e";
  if (value.includes("ACTION") || value.includes("action")) return "#7c3aed";
  if (value.includes("GOAL") || value.includes("goal")) return "#dc2626";
  if (value.includes("CREDENTIAL")) return "#b45309";
  return "#64748b";
}
