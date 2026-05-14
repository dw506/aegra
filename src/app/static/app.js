const state = {
  workspaceId: null,
  workspaces: [],
  evidence: [],
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
  await Promise.all([
    refreshAssets(),
    refreshPolicy(),
    refreshTasks(),
    refreshGraph(),
    refreshFindings(),
    refreshApprovals(),
  ]);
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

async function refreshTasks() {
  const op = currentOperation();
  const payload = await api(`/operations/${encodeURIComponent(op)}/tasks`);
  renderGraph("taskGraph", payload.nodes || [], payload.edges || []);
  $("highRiskTable").innerHTML = (payload.high_risk_tasks || []).map((item) => `
    <tr>
      <td>${escapeHtml(item.label || item.task_id)}</td>
      <td class="risk">${escapeHtml(item.estimated_risk ?? "")}</td>
      <td>${escapeHtml(item.approval_status || "pending")}</td>
      <td><button data-evidence="all">Evidence</button></td>
      <td><a href="${payload.links.audit}">audit</a></td>
    </tr>`).join("") || `<tr><td colspan="5">No high-risk tasks awaiting approval.</td></tr>`;
}

async function refreshGraph() {
  const op = currentOperation();
  const graph = await api(`/operations/${encodeURIComponent(op)}/graph`);
  renderGraph("knowledgeGraph", graph.nodes || [], graph.edges || []);
}

async function refreshFindings() {
  const op = currentOperation();
  state.evidence = await api(`/operations/${encodeURIComponent(op)}/evidence`);
  const findings = await api(`/operations/${encodeURIComponent(op)}/findings`);
  $("findingsTable").innerHTML = findings.map((item) => `
    <tr>
      <td>${escapeHtml(item.title || item.finding_id)}</td>
      <td>${escapeHtml(item.severity || "")}</td>
      <td>${(item.evidence_refs || []).map((id) => `<button data-evidence="${escapeHtml(id)}">${escapeHtml(id)}</button>`).join(" ")}</td>
      <td><a href="/operations/${op}/audit-report">audit</a></td>
    </tr>`).join("") || `<tr><td colspan="4">No findings yet.</td></tr>`;
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

function renderGraph(containerId, nodes, edges) {
  const container = $(containerId);
  container.innerHTML = "";
  if (!nodes.length) {
    container.textContent = "No graph data.";
    return;
  }
  const widthStep = 210;
  const heightStep = 92;
  const points = new Map();
  nodes.forEach((node, index) => {
    const x = 24 + (index % 4) * widthStep;
    const y = 24 + Math.floor(index / 4) * heightStep;
    points.set(node.id, { x, y });
    const div = document.createElement("div");
    div.className = "node";
    div.style.left = `${x}px`;
    div.style.top = `${y}px`;
    div.innerHTML = `<strong>${escapeHtml(node.type || node.kind || "Task")}</strong>${escapeHtml(node.label || node.id)}`;
    container.appendChild(div);
  });
  edges.forEach((edge) => {
    const a = points.get(edge.source);
    const b = points.get(edge.target);
    if (!a || !b) return;
    const line = document.createElement("div");
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    line.className = "edge";
    line.style.left = `${a.x + 150}px`;
    line.style.top = `${a.y + 28}px`;
    line.style.width = `${Math.max(24, Math.hypot(dx, dy))}px`;
    line.style.transform = `rotate(${Math.atan2(dy, dx)}rad)`;
    container.appendChild(line);
  });
  container.style.minHeight = `${Math.max(320, 120 + Math.ceil(nodes.length / 4) * heightStep)}px`;
}

function showEvidence(id) {
  const items = id === "all" ? state.evidence : state.evidence.filter((item) => item.evidence_id === id);
  $("evidenceDetails").textContent = JSON.stringify(items, null, 2);
  $("evidenceDrawer").classList.add("open");
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

document.body.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  if (target.dataset.evidence) showEvidence(target.dataset.evidence);
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

$("closeEvidence").addEventListener("click", () => $("evidenceDrawer").classList.remove("open"));

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
