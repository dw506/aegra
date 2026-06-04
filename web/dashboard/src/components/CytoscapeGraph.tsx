import cytoscape, { type Core } from "cytoscape";
import { useEffect, useMemo, useRef } from "react";
import type { GraphState } from "../graphState";
import { buildDisplayName } from "../graphTransforms";
import type { VisualNode } from "../types";

interface Props {
  graph: GraphState;
  onSelectNode: (node: VisualNode) => void;
}

export function CytoscapeGraph({ graph, onSelectNode }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);
  const latestGraphRef = useRef(graph);
  const structureSignatureRef = useRef("");
  const structureSignature = useMemo(() => graphStructureSignature(graph), [graph]);

  latestGraphRef.current = graph;

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
            "font-size": "10px",
            "font-weight": 600,
            "text-wrap": "none",
            "text-max-width": "78px",
            "text-valign": "bottom",
            "text-margin-y": 7,
            width: "34px",
            height: "34px",
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
      const node = latestGraphRef.current.nodes[id];
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
    const freshCutoff = Date.now() - 4500;
    const structureChanged = structureSignatureRef.current !== structureSignature;
    structureSignatureRef.current = structureSignature;

    if (!structureChanged) {
      for (const node of Object.values(graph.nodes)) {
        const element = cy.getElementById(node.id);
        if (!element.length) continue;
        element.data({
          label: compactNodeLabel(node),
          fullLabel: node.label,
          color: colorFor(node.type || node.status || ""),
          border: graph.highlighted[node.id] > freshCutoff ? "#f59e0b" : "#ffffff",
        });
        element.toggleClass("fresh", graph.highlighted[node.id] > freshCutoff);
      }
      for (const edge of Object.values(graph.edges)) {
        const element = cy.getElementById(edge.id);
        if (!element.length) continue;
        element.data({ label: edge.label || edge.type || "" });
        element.toggleClass("fresh", graph.highlighted[edge.id] > freshCutoff);
      }
      return;
    }

    cy.elements().remove();
    cy.add([
      ...Object.values(graph.nodes).map((node) => ({
        group: "nodes" as const,
        data: {
          id: node.id,
          label: compactNodeLabel(node),
          fullLabel: node.label,
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
  }, [graph, structureSignature]);

  return <div className="cyGraph" ref={containerRef} />;
}

function graphStructureSignature(graph: GraphState) {
  const nodeIds = Object.keys(graph.nodes).sort();
  const edgeIds = Object.values(graph.edges)
    .map((edge) => `${edge.id}:${edge.source}->${edge.target}`)
    .sort();
  return `${nodeIds.join("|")}::${edgeIds.join("|")}`;
}

function colorFor(value: string) {
  if (value.includes("HOST") || value.includes("Asset")) return "#2563eb";
  if (value.includes("SERVICE")) return "#0f766e";
  if (value.includes("ACTION") || value.includes("action")) return "#7c3aed";
  if (value.includes("GOAL") || value.includes("goal")) return "#dc2626";
  if (value.includes("CREDENTIAL")) return "#b45309";
  return "#64748b";
}

function compactNodeLabel(node: VisualNode) {
  const displayName = buildDisplayName(node);
  if (displayName && displayName !== node.type && displayName !== node.label) return compactText(displayName, 28);
  const type = String(node.type || "").trim();
  const props = node.properties || {};
  const propLabel = firstString(
    props.address,
    props.hostname,
    props.url,
    props.service_name,
    props.service,
    props.action_type,
    props.node_type,
    props.evidence_kind,
    props.tool_name,
  );

  if (type === "Host") return firstString(props.address, props.hostname, node.label, shortId(node.id));
  if (type === "Service") {
    const port = firstString(props.port);
    const service = firstString(props.service_name, props.service, props.protocol, "svc");
    return port ? `${service}:${port}` : service;
  }
  if (type === "Evidence") return "Evidence";
  if (type === "Observation") return "Observation";
  if (type === "Finding") return "Finding";
  if (type === "ToolTrace") return firstString(props.tool_name, "Tool");
  if (type === "Goal") return "Goal";
  if (type === "NetworkZone") return "Scope";
  if (type.toLowerCase() === "action") return compactActionLabel(propLabel || node.label);
  if (type) return compactText(type, 18);
  return compactText(propLabel || node.label || shortId(node.id), 18);
}

function compactActionLabel(value: string) {
  return compactText(value.replace(/_/g, " ").toLowerCase(), 20);
}

function firstString(...values: unknown[]) {
  for (const value of values) {
    if (value === undefined || value === null || value === "") continue;
    return String(value);
  }
  return "";
}

function compactText(value: string, maxLength: number) {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, Math.max(0, maxLength - 1))}…`;
}

function shortId(id: string) {
  const parts = id.split("::");
  return compactText(parts[parts.length - 1] || id, 12);
}
