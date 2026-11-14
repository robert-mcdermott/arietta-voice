// Connections: MCP servers that give the assistant more tools. Add, edit,
// test, choose tools and choose when the assistant asks before acting.

import { $, $$, api, escapeHtml, guard, toast } from "./core.js";

const POLICY_LABELS = {
  ask: "Ask every time (say “always” to stop asking about that action)",
  first_use: "Ask the first time each action is used",
  never: "Never ask: I trust this server",
};
const POLICY_SHORT = { ask: "Asks before actions", first_use: "Asks once per action", never: "Never asks" };

const state = { payload: null, editing: null, dirty: false, autoTested: new Set(), testing: new Set() };

export function connectionsDirty() {
  return state.dirty;
}

const pathName = (name) => encodeURIComponent(name);

function server(name) {
  return (state.payload?.servers || []).find((item) => item.name === name);
}

function statusPill(item) {
  if (!item.enabled) return `<span class="pill">Off</span>`;
  if (state.testing.has(item.name)) return `<span class="pill">Testing…</span>`;
  const test = item.last_test;
  if (!test) return `<span class="pill">Not tested</span>`;
  if (!test.ok) return `<span class="pill bad">Not working</span>`;
  const on = test.tools.filter((tool) => !item.disabled_tools.includes(tool.name)).length;
  const count = on === test.tools.length ? `${on} tools` : `${on} of ${test.tools.length} tools on`;
  return `<span class="pill ok">Working · ${count}</span>`;
}

function toolRows(item) {
  const test = item.last_test;
  if (!test?.ok || !test.tools.length) return "";
  const rows = test.tools
    .map((tool) => {
      const name = escapeHtml(tool.name);
      const kind = tool.read_only
        ? `<span class="pill">Reads only</span>`
        : `<span class="pill warn">Changes things</span>`;
      const remembered = item.remembered.includes(tool.name)
        ? `<span class="pill accent">Approved, won't ask</span>`
        : "";
      return `<label class="mcp-tool">
          <input type="checkbox" class="mcp-tool-enabled" data-server="${escapeHtml(item.name)}" data-tool="${name}" ${tool.enabled ? "checked" : ""}>
          <span><strong>${escapeHtml(tool.title)}</strong> ${kind}${remembered}
          <span class="tool-desc">${escapeHtml(tool.description || "")}</span></span>
        </label>`;
    })
    .join("");
  return `<details class="fold mcp-tools"><summary>Tools the assistant may use</summary>
      <div class="mcp-tool-list">${rows}</div>
      <div class="row"><button type="button" class="btn small" data-mcp-action="save-tools" data-name="${escapeHtml(item.name)}">Save tool choices</button>
      <span class="hint">Fewer tools keep a small model accurate and quick.</span></div>
    </details>`;
}

function serverCard(item) {
  const name = escapeHtml(item.name);
  const where =
    item.transport === "stdio"
      ? `Runs <code>${escapeHtml([item.command, ...item.args].join(" "))}</code>`
      : `Connects to <code>${escapeHtml(item.url)}</code>`;
  const test = item.last_test;
  const problem = test && !test.ok ? `<div class="tool-error">${escapeHtml(test.error || "Failed")}</div>` : "";
  const info = test?.ok && test.server_info ? ` · ${escapeHtml(test.server_info)}` : "";
  const forget = item.remembered.length
    ? `<button type="button" class="btn small quiet" data-mcp-action="forget" data-name="${name}">Ask again about everything</button>`
    : "";
  return `<div class="tool-item" data-name="${name}">
      <label class="switch" title="${item.enabled ? "On" : "Off"}">
        <input class="mcp-enabled" type="checkbox" data-name="${name}" aria-label="Connect to ${name}" ${item.enabled ? "checked" : ""}>
        <span class="switch-track"></span>
      </label>
      <div>
        <div class="tool-title"><strong>${name}</strong>${statusPill(item)}<span class="pill">${escapeHtml(POLICY_SHORT[item.confirm] || item.confirm)}</span></div>
        <div class="tool-meta">${where}${info}</div>
        ${problem}
        ${toolRows(item)}
      </div>
      <div class="tool-actions">
        <button type="button" class="btn small" data-mcp-action="test" data-name="${name}">Test</button>
        <button type="button" class="btn small" data-mcp-action="edit" data-name="${name}">Edit</button>
        ${forget}
        <button type="button" class="btn small quiet" data-mcp-action="remove" data-name="${name}">Remove</button>
      </div>
    </div>`;
}

function render(payload) {
  state.payload = payload;
  const servers = payload.servers || [];
  const on = servers.filter((item) => item.enabled).length;
  $("#mcp-summary").innerHTML = servers.length
    ? `<span class="pill">${on} of ${servers.length} on</span>`
    : "";
  $("#mcp-file").innerHTML = `Saved in <code>${escapeHtml(payload.file)}</code>, which stays private.${
    payload.enabled ? "" : " MCP is turned off in your config ([mcp].enabled = false)."
  }`;
  $("#mcp-problems").innerHTML = (payload.problems || [])
    .map((problem) => `<div class="tool-error">${escapeHtml(problem)}</div>`)
    .join("");
  $("#mcp-list").innerHTML = servers.length
    ? `<div class="tool-list">${servers.map(serverCard).join("")}</div>`
    : `<p class="empty">No servers yet. Add one, or paste the snippet from a server's README below.</p>`;
  $("#mcp-programs-policy").textContent = payload.programs_allowed
    ? "A program server runs on this Mac with your permissions; add only ones you trust."
    : "Program servers run code on this Mac, so adding or changing one here needs [admin].allow_source_editing = true. URL servers can be added now.";
}

export async function loadConnections() {
  render(await api("/api/mcp"));
  // Check each server once per visit, so the page shows what works.
  for (const item of state.payload.servers) {
    if (item.enabled && !item.last_test && !state.autoTested.has(item.name)) {
      state.autoTested.add(item.name);
      guard(() => test(item.name, { quiet: true }))();
    }
  }
}

async function test(name, { quiet = false } = {}) {
  state.testing.add(name);
  render(state.payload);
  try {
    const result = await api(`/api/mcp/servers/${pathName(name)}/test`, { method: "POST" });
    const item = server(name);
    if (item) item.last_test = result;
    if (!quiet) toast(result.ok ? `${name} works · ${result.tools.length} tools` : `${name}: ${result.error}`, result.ok ? "ok" : "bad");
  } finally {
    state.testing.delete(name);
    render(state.payload);
  }
}

function payloadFor(item, changes = {}) {
  return {
    name: item.name,
    original_name: item.name,
    transport: item.transport,
    command: item.command,
    args: item.args,
    env: item.env,
    cwd: item.cwd,
    url: item.url,
    headers: item.headers,
    enabled: item.enabled,
    confirm: item.confirm,
    disabled_tools: item.disabled_tools,
    ...changes,
  };
}

async function save(body) {
  const tests = Object.fromEntries((state.payload?.servers || []).map((item) => [item.name, item.last_test]));
  const payload = await api("/api/mcp/servers", { method: "POST", body: JSON.stringify(body) });
  for (const item of payload.servers) item.last_test = item.last_test || tests[item.name] || null;
  render(payload);
  return payload;
}

function lines(text) {
  return text.split("\n").map((line) => line.trim()).filter(Boolean);
}

function pairs(text, separator) {
  const result = {};
  for (const line of lines(text)) {
    const at = line.indexOf(separator);
    if (at < 1) throw new Error(`Expected NAME${separator}value: ${line}`);
    result[line.slice(0, at).trim()] = line.slice(at + separator.length).trim();
  }
  return result;
}

function showTransport() {
  const stdio = $("#mcp-transport").value === "stdio";
  $("#mcp-stdio-fields").hidden = !stdio;
  $("#mcp-http-fields").hidden = stdio;
}

function openEditor(item = null) {
  state.editing = item?.name || null;
  $("#mcp-editor").hidden = false;
  $("#mcp-editor-title").textContent = item ? `Edit ${item.name}` : "Add a server";
  $("#mcp-name").value = item?.name || "";
  $("#mcp-transport").value = item?.transport || "stdio";
  $("#mcp-command").value = item?.command || "";
  $("#mcp-args").value = (item?.args || []).join("\n");
  $("#mcp-env").value = Object.entries(item?.env || {}).map(([k, v]) => `${k}=${v}`).join("\n");
  $("#mcp-url").value = item?.url || "";
  $("#mcp-headers").value = Object.entries(item?.headers || {}).map(([k, v]) => `${k}: ${v}`).join("\n");
  $("#mcp-confirm").value = item?.confirm || "ask";
  $("#mcp-enabled").checked = item ? item.enabled : true;
  showTransport();
  state.dirty = false;
  $("#mcp-editor").scrollIntoView({ behavior: "smooth", block: "start" });
  $("#mcp-name").focus();
}

function closeEditor() {
  state.editing = null;
  state.dirty = false;
  $("#mcp-editor").hidden = true;
}

async function saveEditor() {
  const previous = state.editing ? server(state.editing) : null;
  const body = {
    name: $("#mcp-name").value.trim(),
    original_name: state.editing,
    transport: $("#mcp-transport").value,
    command: $("#mcp-command").value.trim(),
    args: lines($("#mcp-args").value),
    env: pairs($("#mcp-env").value, "="),
    cwd: previous?.cwd || "",
    url: $("#mcp-url").value.trim(),
    headers: pairs($("#mcp-headers").value, ":"),
    enabled: $("#mcp-enabled").checked,
    confirm: $("#mcp-confirm").value,
    disabled_tools: previous?.disabled_tools || [],
  };
  await save(body);
  closeEditor();
  toast(`Saved ${body.name}`, "ok");
  if (body.enabled) await test(body.name, { quiet: false });
}

async function saveTools(name) {
  const item = server(name);
  const off = $$(`.mcp-tool-enabled[data-server="${CSS.escape(name)}"]`)
    .filter((input) => !input.checked)
    .map((input) => input.dataset.tool);
  await save(payloadFor(item, { disabled_tools: off }));
  const tested = server(name)?.last_test;
  if (tested) for (const tool of tested.tools) tool.enabled = !off.includes(tool.name);
  render(state.payload);
  toast(`Saved tool choices for ${name}`, "ok");
}

async function remove(name) {
  if (!confirm(`Remove ${name}? Its tools stop being available.`)) return;
  render(await api(`/api/mcp/servers/${pathName(name)}`, { method: "DELETE" }));
  toast(`Removed ${name}`, "ok");
}

async function forget(name) {
  const tests = Object.fromEntries(state.payload.servers.map((item) => [item.name, item.last_test]));
  const payload = await api(`/api/mcp/servers/${pathName(name)}/forget`, { method: "POST" });
  for (const item of payload.servers) item.last_test = tests[item.name] || null;
  render(payload);
  toast(`${name} will ask again before every action`, "ok");
}

async function importSnippet() {
  const text = $("#mcp-import-text").value.trim();
  if (!text) throw new Error("Paste a snippet first");
  const result = await api("/api/mcp/import", { method: "POST", body: JSON.stringify({ text }) });
  render(result);
  $("#mcp-import-text").value = "";
  const skipped = result.skipped.length ? ` · already there: ${result.skipped.join(", ")}` : "";
  toast(result.added.length ? `Added ${result.added.join(", ")}${skipped}` : `Nothing new${skipped}`, "ok");
  for (const name of result.added) await test(name);
}

export function initConnections() {
  $("#mcp-confirm").innerHTML = Object.entries(POLICY_LABELS)
    .map(([value, label]) => `<option value="${value}">${escapeHtml(label)}</option>`)
    .join("");
  $("#mcp-add").addEventListener("click", () => openEditor());
  $("#mcp-cancel").addEventListener("click", closeEditor);
  $("#mcp-save").addEventListener("click", guard(saveEditor));
  $("#mcp-transport").addEventListener("change", showTransport);
  $("#mcp-import").addEventListener("click", guard(importSnippet));
  $("#mcp-editor").addEventListener("input", () => {
    state.dirty = true;
  });
  $("#mcp-list").addEventListener("change", (event) => {
    const input = event.target;
    if (!input.classList.contains("mcp-enabled")) return;
    const item = server(input.dataset.name);
    guard(async () => {
      await save(payloadFor(item, { enabled: input.checked }));
      toast(`${item.name} ${input.checked ? "on" : "off"}`, "ok");
    })();
  });
  $("#mcp-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-mcp-action]");
    if (!button) return;
    const name = button.dataset.name;
    const action = button.dataset.mcpAction;
    if (action === "test") guard(() => test(name))();
    if (action === "edit") openEditor(server(name));
    if (action === "remove") guard(() => remove(name))();
    if (action === "forget") guard(() => forget(name))();
    if (action === "save-tools") guard(() => saveTools(name))();
  });
}
