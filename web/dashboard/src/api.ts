import type { OperationSummary, UnifiedVisualization, VisualGraphSnapshot } from "./types";

export const API_BASE = import.meta.env.VITE_GRAPH_API_BASE || "http://127.0.0.1:8001";
export const WS_BASE =
  import.meta.env.VITE_GRAPH_WS_BASE ||
  "ws://127.0.0.1:8001";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<T>;
}

export const listOperations = () => request<OperationSummary[]>("/operations");

export const fetchSnapshot = (operationId: string) =>
  request<VisualGraphSnapshot>(`/operations/${encodeURIComponent(operationId)}/visual-graphs/snapshot`);

export const fetchVisualization = (operationId: string) =>
  request<UnifiedVisualization>(`/operations/${encodeURIComponent(operationId)}/visualization`);

export const graphWsUrl = (operationId: string) =>
  `${WS_BASE}/operations/${encodeURIComponent(operationId)}/visual-graphs/ws`;
