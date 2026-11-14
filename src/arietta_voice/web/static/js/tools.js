// Tools: one list of built-in and user tools. Enable, reorder, test, validate,
// view source and (when allowed) create or edit user tools.

import {
  $,
  $$,
  api,
  escapeHtml,
  guard,
  readJsonTextarea,
  showResult,
  toast,
  writeJsonTextarea,
} from "./core.js";
import { loadOverview, runtimeAction } from "./home.js";

const tools = {
  payload: null,
  order: [],
  sourceEditing: false,
  dirty: false,
  testing: null,
  editor: { name: null, file: null, editable: false, dirty: false },
};

export function toolSourceDirty() {
  return tools.editor.dirty;
}

export function toolsBusy() {
  return tools.dirty || tools.editor.dirty;
}

const pathName = (name) => encodeURIComponent(name);

function record(name) {
  return (tools.payload?.tools || []).find((tool) => tool.name === name);
}

function exampleArguments(parameters) {
  const example = {};
  for (const [name, spec] of Object.entries(parameters?.properties || {})) {
    if (spec.enum) example[name] = spec.enum[0];
    else if (spec.type === "string") example[name] = "";
    else if (spec.type === "boolean") example[name] = false;
    else example[name] = 0;
  }
  return example;
}

function describeArguments(parameters) {
  const properties = Object.entries(parameters?.properties || {});
  if (!properties.length) return "no arguments";
  const required = new Set(parameters.required || []);
  return properties
    .map(([name, spec]) => `${name}${required.has(name) ? "" : "?"}: ${spec.type}`)
    .join(", ");
}

function setDirty(dirty) {
  tools.dirty = dirty;
  const button = $("#tools-save-enabled");
  button.classList.toggle("primary", !dirty);
  button.classList.toggle("attention", dirty);
  button.textContent = dirty ? "Save changes" : "Save";
}

function toolCard(tool, index, count) {
  const name = escapeHtml(tool.name);
  const enabled = Boolean(tool.enabled);
  const instant = (tool.triggers || []).length > 0;
  const missing = tool.origin === "missing";
  const pills = [
    tool.origin === "user" ? `<span class="pill accent">Yours</span>` : "",
    instant ? `<span class="pill">Instant answer</span>` : "",
    tool.model_callable !== false ? `<span class="pill">Model can call</span>` : "",
    tool.consequential ? `<span class="pill warn">Asks first</span>` : "",
    missing ? `<span class="pill warn">File missing</span>` : "",
    tool.error && !missing ? `<span class="pill bad">Error</span>` : "",
  ].join("");
  const triggers = instant
    ? `<div class="tool-meta">Triggers: ${tool.triggers.map((t) => `<code>${escapeHtml(t)}</code>`).join(" ")}</div>`
    : "";
  return `<div class="tool-item" data-name="${name}">
      <label class="switch" title="${enabled ? "Enabled" : "Disabled"}">
        <input class="tool-enabled" type="checkbox" data-tool="${name}" aria-label="Enable ${name}" ${enabled ? "checked" : ""} ${missing && !enabled ? "disabled" : ""}>
        <span class="switch-track"></span>
      </label>
      <div>
        <div class="tool-title"><strong>${name}</strong>${pills}</div>
        <div class="tool-desc">${tool.description ? `Use when ${escapeHtml(tool.description)}.` : `<span class="muted">No description found.</span>`}</div>
        <div class="tool-meta">Arguments: ${escapeHtml(describeArguments(tool.parameters))}</div>
        ${triggers}
        ${tool.error ? `<div class="tool-error">${escapeHtml(tool.error)}</div>` : ""}
      </div>
      <div class="tool-actions">
        <button type="button" class="btn small quiet" data-tool-action="up" data-tool-name="${name}" ${index === 0 ? "disabled" : ""} aria-label="Move ${name} up">↑</button>
        <button type="button" class="btn small quiet" data-tool-action="down" data-tool-name="${name}" ${index === count - 1 ? "disabled" : ""} aria-label="Move ${name} down">↓</button>
        <button type="button" class="btn small" data-tool-action="test" data-tool-name="${name}" ${missing ? "disabled" : ""}>Test</button>
        <button type="button" class="btn small" data-tool-action="validate" data-tool-name="${name}">Validate</button>
        <button type="button" class="btn small quiet" data-tool-action="source" data-tool-name="${name}" ${missing ? "disabled" : ""}>Source</button>
      </div>
    </div>`;
}

function renderList() {
  const byName = Object.fromEntries((tools.payload?.tools || []).map((t) => [t.name, t]));
  const list = tools.order.map((name) => byName[name]).filter(Boolean);
  $("#tools-content").innerHTML = list.length
    ? `<div class="tool-list">${list.map((tool, i) => toolCard(tool, i, list.length)).join("")}</div>`
    : `<p class="empty">No tools found.</p>`;
}

function render(payload) {
  tools.payload = payload;
  tools.order = (payload.tools || []).map((tool) => tool.name);
  tools.sourceEditing = Boolean(payload.source_editing_enabled);
  const enabled = (payload.enabled || []).length;
  const errors = (payload.errors || []).length;
  $("#tools-summary").innerHTML =
    `<span class="pill">${enabled} of ${tools.order.length} on</span>` +
    (errors ? `<span class="pill bad">${errors} with errors</span>` : "");
  $("#tools-directory").innerHTML = payload.tools_directory
    ? `Your tools live in <code>${escapeHtml(payload.tools_directory)}</code>`
    : "";
  renderList();
  $("#tools-create").disabled = !tools.sourceEditing;
  $("#source-editing-policy").textContent = tools.sourceEditing
    ? "Source editing is on for your own tools. Saving Python source can run code on this machine."
    : "Read-only. Set admin.allow_source_editing = true in your private config to create or edit your own tools here.";
  $$(".source-policy").forEach((element) => {
    element.textContent = tools.sourceEditing ? "" : "Source editing is off in your config.";
  });
  setDirty(false);
  updateSourceControls();
}

export async function loadTools() {
  render(await api("/api/tools"));
}

function move(name, step) {
  // Keep switches the user changed but has not saved.
  const states = Object.fromEntries(
    $$(".tool-enabled").map((input) => [input.dataset.tool, input.checked]),
  );
  const index = tools.order.indexOf(name);
  const target = index + step;
  if (index < 0 || target < 0 || target >= tools.order.length) return;
  [tools.order[index], tools.order[target]] = [tools.order[target], tools.order[index]];
  renderList();
  $$(".tool-enabled").forEach((input) => {
    if (input.dataset.tool in states) input.checked = states[input.dataset.tool];
  });
  setDirty(true);
}

async function saveEnabled() {
  const checked = new Set(
    $$(".tool-enabled")
      .filter((input) => input.checked)
      .map((input) => input.dataset.tool),
  );
  const enabled = tools.order.filter((name) => checked.has(name));
  await api("/api/tools/enabled", { method: "POST", body: JSON.stringify({ enabled }) });
  toast(`Saved · ${enabled.length} tools on`, "ok");
  await loadTools();
  await loadOverview();
}

async function probeEnabled() {
  const text = $("#tools-probe-text").value.trim();
  if (!text) {
    showResult("tools", "Which tool answers?", { error: "Enter a sample request first." });
    return;
  }
  const payload = await api("/api/tools/probe", { method: "POST", body: JSON.stringify({ text }) });
  const title = payload.matched
    ? payload.answers_directly
      ? `“${text}” is answered instantly by ${payload.name}`
      : `${payload.name} adds information, then the model answers`
    : `No instant answer for “${text}”; the model handles it`;
  showResult("tools", title, payload);
}

async function validate(name) {
  const payload = await api(`/api/tools/${pathName(name)}/validate`, { method: "POST" });
  showResult(
    "tools",
    payload.validation === "valid" ? `${name} is valid` : `${name}: ${payload.validation}`,
    payload,
  );
}

async function previewInstructions() {
  const payload = await api("/api/tools/instructions", { method: "POST" });
  showResult(
    "tools",
    "What the model is told about tools",
    payload.instruction_text || "No tools the model can call are enabled.",
  );
}

// ----- Test -----

function selectForTest(name) {
  const tool = record(name);
  tools.testing = name;
  $("#tool-test-meta").innerHTML = `Testing <strong>${escapeHtml(name)}</strong>`;
  $("#tool-test-run").disabled = false;
  $("#tool-test-text").value = $("#tools-probe-text").value || "";
  writeJsonTextarea("#tool-test-args", exampleArguments(tool?.parameters));
  $("#tool-test-result-box").hidden = true;
  $("#tool-test-run").scrollIntoView({ behavior: "smooth", block: "center" });
}

async function runTest() {
  const name = tools.testing;
  if (!name) return;
  const text = $("#tool-test-text").value.trim();
  let body;
  if (text) {
    body = { text };
  } else {
    try {
      body = { arguments: readJsonTextarea("#tool-test-args") };
    } catch (error) {
      showResult("tool-test", "Arguments are not valid JSON", error.message);
      return;
    }
  }
  const payload = await api(`/api/tools/${pathName(name)}/test`, {
    method: "POST",
    body: JSON.stringify(body),
  });
  let title = `Result from ${name}`;
  if (text && !payload.matched) title = `${name}'s triggers do not match “${text}”`;
  if (payload.error) title = `${name} failed: ${payload.error.kind}`;
  showResult("tool-test", title, payload);
}

// ----- Source and create -----

function updateSourceControls() {
  const { name, file, editable, dirty } = tools.editor;
  const canEdit = Boolean(name) && editable && tools.sourceEditing;
  $("#tool-source-save").disabled = !canEdit;
  $("#tool-source-validate").disabled = !name;
  $("#tool-source-content").readOnly = !canEdit;
  $("#tool-source-meta").innerHTML = name
    ? `<strong>${escapeHtml(name)}</strong>${editable ? "" : " · built in, read-only"}${dirty ? " · unsaved changes" : ""}${file ? ` · <code>${escapeHtml(file)}</code>` : ""}`
    : "Choose Source on a tool to view its code.";
}

async function openSource(name) {
  if (tools.editor.dirty && !confirm("Discard unsaved source edits and open another tool?")) return;
  const payload = await api(`/api/tools/${pathName(name)}/source`);
  tools.editor = { name, file: payload.file, editable: payload.editable, dirty: false };
  $("#tool-source-content").value = payload.content;
  $("#tool-source-result-box").hidden = true;
  updateSourceControls();
  $("#tool-source-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function saveSource() {
  const { name } = tools.editor;
  if (!name) return;
  const payload = await api(`/api/tools/${pathName(name)}/source`, {
    method: "PUT",
    body: JSON.stringify({ content: $("#tool-source-content").value }),
  });
  tools.editor.dirty = false;
  showResult("tool-source", `Saved ${name}`, payload.tool);
  await loadTools();
}

async function validateSource() {
  if (!tools.editor.name) return;
  const payload = await api(`/api/tools/${pathName(tools.editor.name)}/validate`, {
    method: "POST",
  });
  showResult("tool-source", `Validation: ${tools.editor.name}`, payload);
}

async function createTool() {
  const payload = await api("/api/tools/create", {
    method: "POST",
    body: JSON.stringify({
      name: $("#tools-create-name").value.trim(),
      description: $("#tools-create-description").value.trim(),
      trigger: $("#tools-create-trigger").value.trim(),
      response: $("#tools-create-response").value,
      enable: $("#tools-create-enable").checked,
    }),
  });
  toast(`Created ${payload.name}`, "ok");
  $("#tools-create-name").value = "";
  await loadTools();
  await loadOverview();
  await openSource(payload.name);
}

async function reloadForChat() {
  await api("/api/tools/reload", { method: "POST" });
  toast("Tools reloaded for console chat", "ok");
  await loadTools();
}

export function initTools() {
  $("#tools-save-enabled").addEventListener("click", guard(saveEnabled));
  $("#tools-probe-enabled").addEventListener("click", guard(probeEnabled));
  $("#tools-probe-text").addEventListener("keydown", (event) => {
    if (event.key === "Enter") guard(probeEnabled)();
  });
  $("#tools-instructions").addEventListener("click", guard(previewInstructions));
  $("#tools-create").addEventListener("click", guard(createTool));
  $("#tools-reload").addEventListener("click", guard(reloadForChat));
  $("#tools-restart-runtime").addEventListener(
    "click",
    guard(() => runtimeAction("restart")),
  );
  $("#tool-test-run").addEventListener("click", guard(runTest));
  $("#tools-content").addEventListener("change", (event) => {
    if (event.target.classList.contains("tool-enabled")) setDirty(true);
  });
  $("#tools-content").addEventListener("click", (event) => {
    const target = event.target.closest("[data-tool-action]");
    if (!target) return;
    const name = target.dataset.toolName;
    const action = target.dataset.toolAction;
    if (action === "up") move(name, -1);
    if (action === "down") move(name, 1);
    if (action === "test") selectForTest(name);
    if (action === "validate") guard(validate)(name);
    if (action === "source") guard(openSource)(name);
  });
  $("#tool-source-content").addEventListener("input", () => {
    if (!tools.editor.name || tools.editor.dirty) return;
    tools.editor.dirty = true;
    updateSourceControls();
  });
  $("#tool-source-save").addEventListener("click", guard(saveSource));
  $("#tool-source-validate").addEventListener("click", guard(validateSource));
  updateSourceControls();
}
