// Tools: enable, test, validate and (when allowed) edit instant and model tools.

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
  sourceEditing: false,
  dirty: { instant: false, model: false },
  editor: { kind: null, tool: null, file: null, dirty: false },
  argsFor: null,
};

const EMPTY = { enabled: [], available: [], tools: [], load_errors: [] };

export function toolSourceDirty() {
  return tools.editor.dirty;
}

export function toolsBusy() {
  return tools.dirty.instant || tools.dirty.model || tools.editor.dirty;
}

const pathName = (name) => encodeURIComponent(name);
const basePath = (kind) => (kind === "model" ? "/api/model-tools" : "/api/tools");

function modelToolRecord(name) {
  return (tools.payload?.model_tools?.tools || []).find((tool) => tool.configured_name === name);
}

function summary(group) {
  const enabled = (group.enabled || []).length;
  const available = (group.available || []).length;
  const errors = (group.load_errors || []).length;
  return `<span class="pill">${enabled} of ${available} on</span>${
    errors ? `<span class="pill bad">${errors} with errors</span>` : ""
  }`;
}

function setDirty(kind, dirty) {
  tools.dirty[kind] = dirty;
  const button = $(kind === "model" ? "#model-tools-save-enabled" : "#tools-save-enabled");
  button.classList.toggle("primary", !dirty);
  button.classList.toggle("attention", dirty);
  button.textContent = dirty ? "Save changes" : "Save";
}

function toolCard(tool, kind) {
  const enabled = Boolean(tool.enabled);
  const name = escapeHtml(tool.configured_name);
  const attribute = kind === "model" ? "data-model-tool-action" : "data-tool-action";
  const inputClass = kind === "model" ? "model-tool-enabled" : "tool-enabled";
  const canViewSource = tool.source === "local" && tool.available;
  const pills = [
    tool.error ? `<span class="pill bad">Error</span>` : "",
    tool.available ? "" : `<span class="pill warn">Configured, file missing</span>`,
  ].join("");
  const description =
    kind === "model"
      ? `<div class="tool-desc">Use when ${escapeHtml(tool.description || "no description is available")}.</div>`
      : "";
  const args =
    kind === "model"
      ? ` · arguments <code>${escapeHtml(JSON.stringify(tool.arguments_example || {}))}</code>`
      : "";
  const actions =
    kind === "model"
      ? `<button type="button" class="btn small" ${attribute}="invoke" data-tool-name="${name}" ${tool.available && !tool.error ? "" : "disabled"}>Invoke</button>`
      : `<button type="button" class="btn small" ${attribute}="probe" data-tool-name="${name}">Test</button>`;
  return `<div class="tool-item">
      <label class="switch" title="${enabled ? "Enabled" : "Disabled"}">
        <input class="${inputClass}" type="checkbox" data-tool="${name}" aria-label="Enable ${name}" ${enabled ? "checked" : ""} ${tool.error && !enabled ? "disabled" : ""}>
        <span class="switch-track"></span>
      </label>
      <div>
        <div class="tool-title"><strong>${name}</strong>${pills}</div>
        ${description}
        <div class="tool-meta"><code>${escapeHtml(tool.module || "—")}</code>${args}</div>
        ${tool.error ? `<div class="tool-error">${escapeHtml(tool.error)}</div>` : ""}
      </div>
      <div class="tool-actions">
        ${actions}
        <button type="button" class="btn small" ${attribute}="validate" data-tool-name="${name}">Validate</button>
        <button type="button" class="btn small quiet" ${attribute}="edit-source" data-tool-name="${name}" ${canViewSource ? "" : "disabled"}>Source</button>
      </div>
    </div>`;
}

function render(payload) {
  tools.payload = payload;
  tools.sourceEditing = Boolean(payload.source_editing_enabled);
  const instant = payload.assistant_tools || EMPTY;
  const model = payload.model_tools || EMPTY;

  $("#tools-summary").innerHTML = summary(instant);
  $("#tools-content").innerHTML = (instant.tools || []).length
    ? `<div class="tool-list">${instant.tools.map((tool) => toolCard(tool, "instant")).join("")}</div>`
    : `<p class="empty">No instant tools found.</p>`;
  $("#model-tools-summary").innerHTML = summary(model);
  $("#tools-model-content").innerHTML = (model.tools || []).length
    ? `<div class="tool-list">${model.tools.map((tool) => toolCard(tool, "model")).join("")}</div>`
    : `<p class="empty">No model tools found.</p>`;

  $("#tools-create").disabled = !tools.sourceEditing;
  $("#model-tools-create").disabled = !tools.sourceEditing;
  const policy = tools.sourceEditing
    ? "Source editing is on. Saving Python source can run code on this machine."
    : "Read-only. Set admin.allow_source_editing = true in your private config to create or edit tools here.";
  $("#source-editing-policy").textContent = policy;
  $$(".source-policy").forEach((element) => {
    element.textContent = tools.sourceEditing ? "" : "Source editing is off in your config.";
  });
  setDirty("instant", false);
  setDirty("model", false);
  updateSourceControls();
}

export async function loadTools() {
  render(await api("/api/tools"));
}

// ----- Enable / disable -----

async function saveEnabled(kind) {
  const selector = kind === "model" ? ".model-tool-enabled" : ".tool-enabled";
  const enabled = $$(selector)
    .filter((input) => input.checked)
    .map((input) => input.dataset.tool);
  const endpoint = `${basePath(kind)}/enabled`;
  await api(endpoint, { method: "POST", body: JSON.stringify({ enabled }) });
  toast(`Saved · ${enabled.length} ${kind === "model" ? "model" : "instant"} tools on`, "ok");
  await loadTools();
  await loadOverview();
}

// ----- Instant tools -----

async function probeEnabled() {
  const text = $("#tools-probe-text").value.trim();
  if (!text) {
    showResult("tools", "Test phrase", { error: "Enter a sample phrase first." });
    return;
  }
  const payload = await api("/api/tools/probe", { method: "POST", body: JSON.stringify({ text }) });
  showResult("tools", `Result for “${text}”`, payload);
}

async function probeOne(name) {
  const text = $("#tools-probe-text").value.trim();
  if (!text) {
    showResult("tools", `Test ${name}`, { error: "Enter a sample phrase first.", tool: name });
    $("#tools-probe-text").focus();
    return;
  }
  const payload = await api(`/api/tools/${pathName(name)}/probe`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });
  showResult("tools", `${name} on “${text}”`, payload);
}

async function validate(kind, name) {
  const payload = await api(`${basePath(kind)}/${pathName(name)}/validate`, { method: "POST" });
  showResult(kind === "model" ? "model-tools" : "tools", `Validation: ${name}`, payload);
  await loadTools();
}

async function createInstantTool() {
  const payload = await api("/api/tools/create", {
    method: "POST",
    body: JSON.stringify({
      module_name: $("#tools-create-name").value.trim(),
      trigger_phrase: $("#tools-create-trigger").value.trim(),
      response: $("#tools-create-response").value,
      enable: $("#tools-create-enable").checked,
    }),
  });
  showResult("tools", `Created ${payload.module_name}`, payload);
  $("#tools-create-name").value = "";
  await loadTools();
  await loadOverview();
}

// ----- Model tools -----

function argumentMismatch(tool, provided) {
  const expected = tool?.arguments_example || {};
  const expectedKeys = Object.keys(expected);
  const report = {
    tool: tool?.configured_name,
    expected_arguments: expected,
    provided_arguments: provided,
  };
  if (!expectedKeys.length && Object.keys(provided || {}).length) {
    return { ...report, error: "This tool expects no arguments. Use {} to invoke it." };
  }
  const missing = expectedKeys.filter((key) => !Object.prototype.hasOwnProperty.call(provided, key));
  if (missing.length) {
    return {
      ...report,
      missing_keys: missing,
      error: `This tool expects ${missing.join(", ")} in the arguments JSON.`,
    };
  }
  return null;
}

async function invokeModelTool(name, argsSelector, resultPrefix) {
  let args;
  try {
    args = readJsonTextarea(argsSelector);
  } catch (error) {
    showResult(resultPrefix, `Invoke ${name}`, { error: `Invalid arguments JSON: ${error.message}` });
    return;
  }
  const mismatch = argumentMismatch(modelToolRecord(name), args);
  if (mismatch) {
    showResult(resultPrefix, `Invoke ${name}`, mismatch);
    return;
  }
  const payload = await api(`/api/model-tools/${pathName(name)}/invoke`, {
    method: "POST",
    body: JSON.stringify({ arguments: args }),
  });
  showResult(resultPrefix, `Result from ${name}`, payload);
}

async function invokeFromList(name) {
  // The first click for a tool loads its example arguments; tools that take
  // arguments then wait for a second click so the placeholders can be edited.
  if (tools.argsFor !== name) {
    tools.argsFor = name;
    const example = modelToolRecord(name)?.arguments_example || {};
    writeJsonTextarea("#model-tools-arguments", example);
    if (Object.keys(example).length) {
      showResult(
        "model-tools",
        `Invoke ${name}`,
        "Example arguments are loaded above. Edit them, then choose Invoke again.",
      );
      $("#model-tools-arguments").focus();
      return;
    }
  }
  await invokeModelTool(name, "#model-tools-arguments", "model-tools");
}

async function createModelTool() {
  let argumentsExample;
  try {
    argumentsExample = readJsonTextarea("#model-tools-create-args");
  } catch (error) {
    showResult("model-tools", "Create tool", {
      error: `Invalid arguments example JSON: ${error.message}`,
    });
    return;
  }
  const payload = await api("/api/model-tools/create", {
    method: "POST",
    body: JSON.stringify({
      module_name: $("#model-tools-create-name").value.trim(),
      description: $("#model-tools-create-description").value.trim(),
      arguments_example: argumentsExample,
      followup_context: $("#model-tools-create-context").value,
      enable: $("#model-tools-create-enable").checked,
    }),
  });
  showResult("model-tools", `Created ${payload.module_name}`, payload);
  $("#model-tools-create-name").value = "";
  await loadTools();
  await loadOverview();
}

async function previewInstructions() {
  const payload = await api("/api/model-tools/instructions", { method: "POST" });
  showResult(
    "model-tools",
    "Instructions the model receives",
    payload.instruction_text || "No model tools are enabled.",
  );
}

// ----- Source -----

function updateSourceControls() {
  const { kind, tool, file, dirty } = tools.editor;
  const open = Boolean(kind && tool);
  $("#tool-source-save").disabled = !open || !tools.sourceEditing;
  $("#tool-source-content").readOnly = !tools.sourceEditing;
  $("#tool-source-validate").disabled = !open;
  $("#tool-source-test").disabled = !open;
  $("#tool-source-meta").innerHTML = open
    ? `${kind === "model" ? "Model tool" : "Instant tool"} <strong>${escapeHtml(tool)}</strong>${dirty ? " · unsaved changes" : ""}${file ? ` · <code>${escapeHtml(file)}</code>` : ""}`
    : "Choose Source on a tool to view its code.";
}

async function openSource(kind, name) {
  if (tools.editor.dirty && !confirm("Discard unsaved source edits and open another tool?")) return;
  const payload = await api(`${basePath(kind)}/${pathName(name)}/source`);
  tools.editor = { kind, tool: name, file: payload.file, dirty: false };
  $("#tool-source-content").value = payload.content;
  if (kind === "model") {
    writeJsonTextarea("#tool-source-test-args", payload.tool?.arguments_example || {});
  } else {
    $("#tool-source-test-text").value = $("#tools-probe-text").value || "";
  }
  $("#tool-source-result-box").hidden = true;
  updateSourceControls();
  $("#tool-source-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function saveSource() {
  const { kind, tool } = tools.editor;
  if (!kind || !tool) return;
  const payload = await api(`${basePath(kind)}/${pathName(tool)}/source`, {
    method: "PUT",
    body: JSON.stringify({ content: $("#tool-source-content").value }),
  });
  tools.editor.dirty = false;
  showResult("tool-source", `Saved ${tool}`, payload);
  await loadTools();
}

async function validateSource() {
  const { kind, tool } = tools.editor;
  if (!kind || !tool) return;
  const payload = await api(`${basePath(kind)}/${pathName(tool)}/validate`, { method: "POST" });
  showResult("tool-source", `Validation: ${tool}`, payload);
  await loadTools();
}

async function testSource() {
  const { kind, tool } = tools.editor;
  if (!kind || !tool) return;
  if (kind === "model") {
    await invokeModelTool(tool, "#tool-source-test-args", "tool-source");
    return;
  }
  const text = $("#tool-source-test-text").value.trim();
  if (!text) {
    showResult("tool-source", `Test ${tool}`, { error: "Enter a test phrase first." });
    return;
  }
  const payload = await api(`/api/tools/${pathName(tool)}/probe`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });
  showResult("tool-source", `${tool} on “${text}”`, payload);
}

async function reloadForChat() {
  await api("/api/tools/reload", { method: "POST" });
  await api("/api/model-tools/reload", { method: "POST" });
  toast("Tools reloaded for console chat", "ok");
  await loadTools();
}

export function initTools() {
  $("#tools-save-enabled").addEventListener("click", guard(() => saveEnabled("instant")));
  $("#model-tools-save-enabled").addEventListener("click", guard(() => saveEnabled("model")));
  $("#tools-probe-enabled").addEventListener("click", guard(probeEnabled));
  $("#tools-probe-text").addEventListener("keydown", (event) => {
    if (event.key === "Enter") guard(probeEnabled)();
  });
  $("#tools-create").addEventListener("click", guard(createInstantTool));
  $("#model-tools-create").addEventListener("click", guard(createModelTool));
  $("#model-tools-instructions").addEventListener("click", guard(previewInstructions));
  $("#tools-reload").addEventListener("click", guard(reloadForChat));
  $("#tools-restart-runtime").addEventListener(
    "click",
    guard(() => runtimeAction("restart")),
  );

  $("#tools-content").addEventListener("change", (event) => {
    if (event.target.classList.contains("tool-enabled")) setDirty("instant", true);
  });
  $("#tools-model-content").addEventListener("change", (event) => {
    if (event.target.classList.contains("model-tool-enabled")) setDirty("model", true);
  });
  $("#tools-content").addEventListener("click", (event) => {
    const target = event.target.closest("[data-tool-action]");
    if (!target) return;
    const name = target.dataset.toolName;
    const action = target.dataset.toolAction;
    if (action === "probe") guard(probeOne)(name);
    if (action === "validate") guard(validate)("instant", name);
    if (action === "edit-source") guard(openSource)("tool", name);
  });
  $("#tools-model-content").addEventListener("click", (event) => {
    const target = event.target.closest("[data-model-tool-action]");
    if (!target) return;
    const name = target.dataset.toolName;
    const action = target.dataset.modelToolAction;
    if (action === "invoke") guard(invokeFromList)(name);
    if (action === "validate") guard(validate)("model", name);
    if (action === "edit-source") guard(openSource)("model", name);
  });

  $("#tool-source-content").addEventListener("input", () => {
    if (!tools.editor.kind || tools.editor.dirty) return;
    tools.editor.dirty = true;
    updateSourceControls();
  });
  $("#tool-source-save").addEventListener("click", guard(saveSource));
  $("#tool-source-validate").addEventListener("click", guard(validateSource));
  $("#tool-source-test").addEventListener("click", guard(testSource));
  updateSourceControls();
}
