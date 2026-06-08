const state = {
  workspaceId: null,
  workspaces: [],
  evidence: [],
  findings: [],
  visualization: null,
  trace: null,
  details: new Map(),
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  const payload = text ? safeJson(text) : null;
  if (!response.ok) throw new Error(payload?.detail || text || response.statusText);
  return payload;
}

function safeJson(text) {
  try { return JSON.parse(text); } catch { return text; }
}

function currentOperation() {
  return state.workspaceId;
}

async function refreshAll() {
  await refreshWorkspaces();
  if (!currentOperation()) return;
  state.details.clear();
  await Promise.all([
    refreshAssets(),
    refreshPolicy(),
    refreshVisualization(),
    refreshFindings(),
    refreshApprovals(),
    refreshOperationTrace(),
  ]);
  renderVisualization();
}

async function refreshWorkspaces() {
  state.workspaces = await api("/workspaces");
  if (!state.workspaceId && state.workspaces.length) state.workspaceId = state.workspaces[0].id;
  const select = $("workspaceSelect");
  select.innerHTML = state.workspaces.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`).join("");
  if (state.workspaceId) select.value = state.workspaceId;
}

async function refreshAssets() {
  const rows = await api(`/workspaces/${encodeURIComponent(currentOperation())}/assets`);
  $("assetsTable").innerHTML = rows.map((item) => `
    <tr>
      <td>${escapeHtml(item.kind || "host")}</td>
      <td>${escapeHtml(item.value || item.address || item.hostname || item.url || "")}</td>
      <td>${escapeHtml(item.port || "")}</td>
      <td>${escapeHtml(item.protocol || "")}</td>
      <td><a href="${item.links.audit}">audit</a></td>
    </tr>`).join("");
}

async function refreshPolicy() {
  const op = currentOperation();
  const summary = await api(`/operations/${encodeURIComponent(op)}/summary`);
  const assets = await api(`/workspaces/${encodeURIComponent(op)}/assets`);
  $("policySummary").textContent = JSON.stringify({
    operation_status: summary.operation_status,
    scope: assets.map((item) => ({ value: item.value || item.address, port: item.port, protocol: item.protocol })),
    policy: summary.metadata?.runtime_policy || {},
    audit: `/operations/${op}/audit-report`,
  }, null, 2);
}

async function refreshVisualization() {
  const op = currentOperation();
  try {
    state.visualization = await api(`/operations/${encodeURIComponent(op)}/visualization`);
  } catch (error) {
    const graph = await api(`/operations/${encodeURIComponent(op)}/graph`);
    state.visualization = {
      operation: { id: op, status: "unknown", current_round: 0, goal_status: "unknown" },
      overview: {},
      kg: { nodes: graph.nodes || [], edges: graph.edges || [] },
      ag: { nodes: [], edges: [] },
      timeline: [],
      tool_trace: [],
      evidence: [],
      agent_trace: [],
    };
  }
}

async function refreshFindings() {
  const op = currentOperation();
  state.evidence = await api(`/operations/${encodeURIComponent(op)}/evidence`);
  state.findings = await api(`/operations/${encodeURIComponent(op)}/findings`);
}

async function refreshApprovals() {
  const op = currentOperation();
  const approvals = await api(`/operations/${encodeURIComponent(op)}/approvals`);
  $("approvalsTable").innerHTML = approvals.map((item) => `
    <tr>
      <td>${escapeHtml(item.approval_id)}</td>
      <td>${escapeHtml(item.task_id || "")}</td>
      <td class="${item.status === "denied" ? "denied" : ""}">${escapeHtml(item.status || "pending")}</td>
      <td>${escapeHtml(item.reason || "")}</td>
      <td><button data-approve="${escapeHtml(item.approval_id)}" data-task="${escapeHtml(item.task_id || "")}">Approve</button></td>
    </tr>`).join("") || `<tr><td colspan="5">No approval requests.</td></tr>`;
}

async function refreshOperationTrace() {
  const op = currentOperation();
  try {
    state.trace = await api(`/operations/${encodeURIComponent(op)}/trace`);
  } catch {
    state.trace = { operation_id: op, text: "", message: "Trace unavailable." };
  }
  renderOperationTrace();
}

function renderVisualization() {
  renderOverview();
  renderTimeline();
  renderAgentTrace();
  renderSemanticKgGraph($("kgGraph"), state.visualization?.kg?.nodes || [], state.visualization?.kg?.edges || []);
  renderAgentLaneAg($("agGraph"), state.visualization?.ag?.nodes || [], state.visualization?.ag?.edges || []);
  renderToolTrace();
  renderFindingsAndEvidence();
  renderOperationTrace();
}

function renderOverview() {
  const operation = state.visualization?.operation || {};
  const overview = state.visualization?.overview || {};
  const cards = [
    ["Operation Status", overview.operation_status || operation.status],
    ["Current Round", overview.current_round ?? operation.current_round],
    ["Goal Status", overview.goal_status || operation.goal_status],
    ["Current Agent", overview.current_agent],
    ["Asset Count", overview.asset_count],
    ["Service Count", overview.service_count],
    ["Finding Count", overview.finding_count],
    ["Verified Finding Count", overview.verified_finding_count],
    ["Evidence Count", overview.evidence_count],
    ["Access Count", overview.access_count],
    ["Latest Decision", overview.latest_decision?.decision_summary || overview.latest_decision?.decision, overview.latest_decision],
    ["Latest Evidence", overview.latest_evidence?.summary || overview.latest_evidence?.id, overview.latest_evidence],
  ];
  $("overviewGrid").innerHTML = cards.map(([title, value, detail]) => metricCard(title, value, detail)).join("");
}

function metricCard(title, value, detail) {
  const id = detail ? registerDetail(detail) : "";
  return `
    <button class="metric-card" ${id ? `data-detail="${id}"` : ""}>
      <span>${escapeHtml(title)}</span>
      <strong>${escapeHtml(value ?? "-")}</strong>
      ${detail?.summary ? `<small>${escapeHtml(detail.summary)}</small>` : ""}
    </button>`;
}

function renderTimeline() {
  const events = state.visualization?.timeline || [];
  const byRound = groupBy(events, (item) => item.round ?? 0);
  $("timelineList").innerHTML = Object.keys(byRound).sort(numberSort).map((round) => `
    <section class="timeline-cycle">
      <h3>Cycle ${escapeHtml(round)}</h3>
      ${byRound[round].map((event) => {
        const id = registerDetail(event);
        return `
          <button class="timeline-card" data-detail="${id}">
            <div><strong>${escapeHtml(event.display_name || event.phase)}</strong>${badge(event.status)}</div>
            <p>${escapeHtml(event.summary || "")}</p>
            <dl>
              <dt>Phase</dt><dd>${escapeHtml(event.phase || "")}</dd>
              <dt>Agent</dt><dd>${escapeHtml(event.agent || "-")}</dd>
              <dt>Tool</dt><dd>${escapeHtml(event.tool_name || "-")}</dd>
              <dt>Target</dt><dd>${escapeHtml(event.target || "-")}</dd>
              <dt>Evidence</dt><dd>${escapeHtml((event.evidence_ids || []).join(", ") || "-")}</dd>
              <dt>Created</dt><dd>${escapeHtml(event.created_at || "-")}</dd>
            </dl>
          </button>`;
      }).join("")}
    </section>`).join("") || empty("No timeline events yet.");
}

function renderAgentTrace() {
  const traces = state.visualization?.agent_trace || [];
  $("agentTraceList").innerHTML = traces.map((trace) => {
    const id = registerDetail(trace);
    return `
      <button class="graph-card agent-trace-card" data-detail="${id}">
        <div><strong>Cycle ${escapeHtml(trace.cycle_index ?? trace.round ?? "-")}: ${escapeHtml(trace.selected_agent || "No agent")}</strong>${badge(trace.status)}</div>
        <p>${escapeHtml(trace.summary || trace.planner_decision?.decision_summary || "")}</p>
        <dl>
          <dt>Stage</dt><dd>${escapeHtml(trace.selected_stage || "-")}</dd>
          <dt>Objective</dt><dd>${escapeHtml(trace.objective || "-")}</dd>
          <dt>Task</dt><dd>${escapeHtml(trace.task_brief || "-")}</dd>
          <dt>Tools</dt><dd>${escapeHtml(trace.tool_count ?? (trace.tool_traces || []).length)}</dd>
          <dt>Evidence</dt><dd>${escapeHtml(trace.evidence_count ?? 0)}</dd>
          <dt>Findings</dt><dd>${escapeHtml(trace.finding_count ?? 0)}</dd>
        </dl>
      </button>`;
  }).join("") || empty("No agent trace yet.");
}

function renderSemanticKgGraph(container, kgNodes, kgEdges) {
  const columns = [
    ["Network / Host", ["Network", "Host"]],
    ["Service / WebEndpoint / Technology", ["Service", "WebEndpoint", "Technology"]],
    ["Vulnerability / Finding", ["Vulnerability", "Finding"]],
    ["Evidence / Observation", ["Evidence", "Observation"]],
    ["Credential / Session / PivotRoute / Goal", ["Credential", "Session", "PivotRoute", "Goal"]],
  ];
  container.innerHTML = columns.map(([title, types]) => {
    const nodes = kgNodes.filter((node) => types.includes(node.type));
    return `
      <section class="kg-column">
        <h3>${escapeHtml(title)}</h3>
        ${nodes.map(renderKgNodeCard).join("") || `<p class="empty">No nodes.</p>`}
      </section>`;
  }).join("");
  const names = nodeNameMap(kgNodes);
  $("kgRelations").innerHTML = kgEdges.map((edge) => {
    const id = registerDetail(edge);
    return `
      <tr data-detail="${id}">
        <td>${escapeHtml(names.get(edge.source) || edge.source)}</td>
        <td>${escapeHtml(edge.display_name || edge.type)}</td>
        <td>${escapeHtml(names.get(edge.target) || edge.target)}</td>
        <td>${escapeHtml((edge.evidence_ids || []).join(", "))}</td>
      </tr>`;
  }).join("") || `<tr><td colspan="4">No KG relations.</td></tr>`;
}

function renderKgNodeCard(node) {
  const id = registerDetail(node);
  return `
    <button class="graph-card ${kgClass(node.type)}" data-detail="${id}">
      <strong>${escapeHtml(node.display_name || node.label || node.id)}</strong>
      <dl>
        <dt>Type</dt><dd>${escapeHtml(node.type || "-")}</dd>
        <dt>Status</dt><dd>${escapeHtml(node.status || "-")}</dd>
        <dt>Confidence</dt><dd>${escapeHtml(node.confidence ?? "-")}</dd>
        <dt>Target</dt><dd>${escapeHtml(node.target || "-")}</dd>
        <dt>Evidence</dt><dd>${escapeHtml((node.evidence_ids || []).length)}</dd>
      </dl>
    </button>`;
}

function renderAgentLaneAg(container, agNodes, agEdges) {
  const lanes = ["Planner", "ReconAgent", "VulnAnalysisAgent", "ExploitValidationAgent", "AccessPivotAgent", "GoalAgent", "Other"];
  container.innerHTML = lanes.map((lane) => {
    const nodes = agNodes
      .filter((node) => lane === "Planner" ? node.type === "ReplanStep" : (node.agent || "Other") === lane)
      .sort((a, b) => (a.round ?? 0) - (b.round ?? 0));
    return `
      <section class="agent-lane">
        <h3>${escapeHtml(lane)}</h3>
        ${nodes.map(renderAgNodeCard).join("") || `<p class="empty">No steps.</p>`}
      </section>`;
  }).join("");
  const names = nodeNameMap(agNodes);
  $("agRelations").innerHTML = agEdges.map((edge) => {
    const id = registerDetail(edge);
    return `
      <tr data-detail="${id}">
        <td>${escapeHtml(names.get(edge.source) || edge.source)}</td>
        <td>${escapeHtml(edge.display_name || edge.type)}</td>
        <td>${escapeHtml(names.get(edge.target) || edge.target)}</td>
        <td>${escapeHtml((edge.evidence_ids || []).join(", "))}</td>
      </tr>`;
  }).join("") || `<tr><td colspan="4">No AG process links.</td></tr>`;
}

function renderAgNodeCard(node) {
  const id = registerDetail(node);
  return `
    <button class="graph-card ag-node" data-detail="${id}">
      <div><strong>${escapeHtml(node.display_name || node.id)}</strong>${badge(node.status)}</div>
      <p>${escapeHtml(node.result_summary || node.action_summary || "")}</p>
      <dl>
        <dt>Round</dt><dd>${escapeHtml(node.round ?? "-")}</dd>
        <dt>Type</dt><dd>${escapeHtml(node.type || "-")}</dd>
        <dt>Agent</dt><dd>${escapeHtml(node.agent || "-")}</dd>
        <dt>Target</dt><dd>${escapeHtml(node.target || "-")}</dd>
        <dt>Evidence</dt><dd>${escapeHtml((node.evidence_ids || []).join(", ") || "-")}</dd>
      </dl>
    </button>`;
}

function renderToolTrace() {
  const traces = state.visualization?.tool_trace || [];
  $("toolTraceTable").innerHTML = traces.map((trace) => {
    const id = registerDetail(trace);
    return `
      <tr data-detail="${id}">
        <td>${escapeHtml(trace.round ?? "")}</td>
        <td>${escapeHtml(trace.agent || "")}</td>
        <td>${escapeHtml(trace.step ?? "")}</td>
        <td>${escapeHtml(trace.tool_name || "")}</td>
        <td>${badge(trace.success === true ? "success" : trace.success === false ? "failed" : "pending")}</td>
        <td>${escapeHtml(trace.exit_code ?? "")}</td>
        <td>${escapeHtml(trace.summary || "")}</td>
        <td>${escapeHtml(trace.raw_output_ref || "")}</td>
      </tr>`;
  }).join("") || `<tr><td colspan="8">No tool traces yet.</td></tr>`;
}

function renderFindingsAndEvidence() {
  const op = currentOperation();
  $("findingsTable").innerHTML = state.findings.map((item) => {
    const id = registerDetail(item);
    const evidenceRefs = item.evidence_refs || item.evidence_ids || [];
    return `
      <tr>
        <td>${escapeHtml(item.title || item.finding_id)}</td>
        <td>${escapeHtml(item.kind || item.finding_kind || "")}</td>
        <td>${escapeHtml(item.severity || "")}</td>
        <td>${escapeHtml(item.status || "")}</td>
        <td>${escapeHtml(item.confidence ?? "")}</td>
        <td>${escapeHtml(item.target || item.asset_id || "")}</td>
        <td>${evidenceRefs.map((evidenceId) => detailButton(evidenceId, findEvidence(evidenceId))).join(" ")}</td>
        <td>${escapeHtml(item.validation_status || item.validation?.status || "")}</td>
        <td><button data-detail="${id}">details</button> <a href="/operations/${op}/audit-report">audit</a></td>
      </tr>`;
  }).join("") || `<tr><td colspan="9">No findings yet.</td></tr>`;

  const visualizationEvidence = state.visualization?.evidence || [];
  const evidenceRows = mergeEvidence(visualizationEvidence, state.evidence);
  $("evidenceTable").innerHTML = evidenceRows.map((item) => {
    const id = registerDetail(item);
    return `
      <tr>
        <td>${escapeHtml(item.id || item.evidence_id)}</td>
        <td>${escapeHtml(item.kind || item.source || "")}</td>
        <td>${escapeHtml(item.source_name || item.source_tool || item.tool_name || "")}</td>
        <td>${escapeHtml(item.round ?? item.cycle_index ?? "")}</td>
        <td>${escapeHtml(item.created_by || item.agent || "")}</td>
        <td>${escapeHtml(item.summary || "")}</td>
        <td>${escapeHtml(relatedFinding(item) || "")}</td>
        <td><button data-detail="${id}">details</button></td>
      </tr>`;
  }).join("") || `<tr><td colspan="8">No evidence yet.</td></tr>`;
}

function renderOperationTrace() {
  if (!$("operationTraceText")) return;
  const text = state.trace?.text || state.trace?.message || "";
  const filter = ($("traceFilter")?.value || "").trim().toLowerCase();
  const lines = text.split(/\r?\n/).filter((line) => !filter || line.toLowerCase().includes(filter));
  $("operationTraceText").innerHTML = lines.map(highlightTraceLine).join("\n");
}

function highlightTraceLine(line) {
  const escaped = escapeHtml(line);
  const keywords = ["CYCLE_START", "PLANNER_DECISION", "LLM_DECISION", "TOOL_CALL", "TOOL_RESULT", "STAGE_FINISH", "ERROR"];
  return keywords.reduce((text, keyword) => text.replaceAll(keyword, `<mark>${keyword}</mark>`), escaped);
}

function showDetails(id) {
  const value = state.details.get(id);
  $("detailContent").textContent = JSON.stringify(value, null, 2);
  $("detailDrawer").classList.add("open");
}

function registerDetail(value) {
  const id = `detail-${state.details.size + 1}`;
  state.details.set(id, value);
  return id;
}

function detailButton(label, detail) {
  const id = registerDetail(detail || { id: label });
  return `<button data-detail="${id}">${escapeHtml(label)}</button>`;
}

function findEvidence(id) {
  return mergeEvidence(state.visualization?.evidence || [], state.evidence).find((item) => (item.id || item.evidence_id) === id);
}

function mergeEvidence(...groups) {
  const byId = new Map();
  groups.flat().forEach((item) => {
    const id = item?.id || item?.evidence_id;
    if (!id) return;
    byId.set(id, { ...(byId.get(id) || {}), ...item, id });
  });
  return Array.from(byId.values());
}

function relatedFinding(evidence) {
  const id = evidence.id || evidence.evidence_id;
  const finding = state.findings.find((item) => (item.evidence_refs || item.evidence_ids || []).includes(id));
  return finding?.title || finding?.finding_id;
}

function nodeNameMap(nodes) {
  return new Map(nodes.map((node) => [node.id, node.display_name || node.label || node.id]));
}

function groupBy(items, keyFn) {
  return items.reduce((groups, item) => {
    const key = keyFn(item);
    groups[key] = groups[key] || [];
    groups[key].push(item);
    return groups;
  }, {});
}

function numberSort(a, b) {
  return Number(a) - Number(b);
}

function kgClass(type) {
  return {
    Host: "kg-host",
    Network: "kg-host",
    Service: "kg-service",
    WebEndpoint: "kg-service",
    Technology: "kg-service",
    Finding: "kg-finding",
    Vulnerability: "kg-finding",
    Evidence: "kg-evidence",
    Observation: "kg-evidence",
    Session: "kg-session",
    Credential: "kg-credential",
  }[type] || "";
}

function badge(status) {
  const normalized = String(status ?? "pending").toLowerCase();
  const cls = ["success", "failed", "blocked", "running", "pending"].includes(normalized) ? normalized : "pending";
  return `<span class="badge ${cls}">${escapeHtml(status ?? "pending")}</span>`;
}

function empty(message) {
  return `<p class="empty">${escapeHtml(message)}</p>`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));
}

document.querySelectorAll(".tabs button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tabs button, .view").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    $(button.dataset.view).classList.add("active");
  });
});

$("workspaceSelect").addEventListener("change", async (event) => {
  state.workspaceId = event.target.value;
  await refreshAll();
});

$("createWorkspace").addEventListener("click", async () => {
  const id = $("workspaceId").value.trim();
  await api("/workspaces", {
    method: "POST",
    body: JSON.stringify({ id, name: $("workspaceName").value.trim() || id }),
  });
  state.workspaceId = id;
  await refreshAll();
});

$("addAsset").addEventListener("click", async () => {
  const op = currentOperation();
  await api(`/workspaces/${encodeURIComponent(op)}/assets`, {
    method: "POST",
    body: JSON.stringify({
      kind: "host",
      address: $("assetAddress").value.trim(),
      hostname: $("assetHostname").value.trim() || null,
      port: Number($("assetPort").value || 0) || null,
      protocol: $("assetProtocol").value.trim() || null,
      tags: ["authorized"],
    }),
  });
  await refreshAll();
});

$("startOperation").addEventListener("click", async () => {
  const op = currentOperation();
  $("runStatus").textContent = JSON.stringify(await api(`/operations/${op}/start`, { method: "POST" }), null, 2);
  await refreshAll();
});

$("runOperation").addEventListener("click", async () => {
  const op = currentOperation();
  $("runStatus").textContent = JSON.stringify(await api(`/operations/${op}/run`, {
    method: "POST",
    body: JSON.stringify({ max_cycles: 1, stop_when_quiescent: true }),
  }), null, 2);
  await refreshAll();
});

$("stopOperation").addEventListener("click", async () => {
  const op = currentOperation();
  $("runStatus").textContent = JSON.stringify(await api(`/operations/${op}/stop`, {
    method: "POST",
    body: JSON.stringify({ reason: "operator_stop" }),
  }), null, 2);
  await refreshAll();
});

$("closeDetails").addEventListener("click", () => $("detailDrawer").classList.remove("open"));
$("refreshTrace").addEventListener("click", refreshOperationTrace);
$("traceFilter").addEventListener("input", renderOperationTrace);

document.body.addEventListener("click", async (event) => {
  const source = event.target;
  if (!(source instanceof Element)) return;
  const target = source.closest("[data-detail], [data-approve]");
  if (!(target instanceof HTMLElement)) return;
  if (target.dataset.detail) showDetails(target.dataset.detail);
  if (target.dataset.approve) {
    const op = currentOperation();
    await api(`/operations/${op}/approve`, {
      method: "POST",
      body: JSON.stringify({
        approval_id: target.dataset.approve,
        task_id: target.dataset.task || null,
        decision: "approve",
        reason: "approved from product console",
      }),
    });
    await refreshAll();
  }
});

document.querySelectorAll(".exportReport").forEach((button) => {
  button.addEventListener("click", async () => {
    const op = currentOperation();
    const format = button.dataset.format;
    const response = await fetch(`/operations/${op}/report?format=${format}`);
    $("reportOutput").textContent = await response.text();
  });
});

refreshAll().catch((error) => {
  $("runStatus").textContent = error.message;
});

