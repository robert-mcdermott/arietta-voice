// Persona and memory: the text the assistant is given before every answer.

import { $, api, escapeHtml, guard, toast } from "./core.js";

const persona = { payload: null, dirty: false, memoryDirty: false, saving: false, revision: 0, memoryRevision: null, edit: null };

export function personaDirty() {
  return persona.dirty || persona.memoryDirty || persona.edit !== null;
}

function render(payload) {
  persona.payload = payload;
  if (!persona.dirty) $("#persona-content").value = payload.content;
  $("#persona-file").textContent = payload.private
    ? `Your private persona: ${payload.file}`
    : `Using the default persona. Saving writes your own private copy; the default stays as it is.`;
  const memory = payload.memory;
  if (!persona.memoryDirty) {
    if ($("#memory-content").value !== memory.content) $("#memory-content").value = memory.content;
    persona.memoryRevision = memory.revision;
  }
  if (!persona.edit && $("#memory-entries").dataset.revision !== memory.revision) renderEntries(memory);
  $("#memory-file").textContent = memory.enabled
    ? `Memory is on. Say “Remember that…” to save a dated entry here. ${memory.file}`
    : `Memory is off in Settings, so this file is not used. ${memory.file}`;
  setButtons();
}

const origins = { chat: "Learned in Chat", voice: "Learned by voice", consolidated: "Consolidated memory", file: "Origin unrecorded" };

function renderEntries(memory) {
  const list = $("#memory-entries");
  list.dataset.revision = memory.revision;
  if (!memory.entries.length) {
    const empty = !memory.content.trim() || memory.content.trim() === "# Memory";
    list.innerHTML = empty ? '<p class="hint">No saved facts or dated memories yet.</p>'
      : '<p class="hint">Open the full memory file to review its free-form text.</p>';
    return;
  }
  list.innerHTML = memory.entries.map((entry) => `<div class="result-card" data-entry="${entry.id}">
    <div class="memory-value">${escapeHtml(entry.text)}</div>
    <div class="row"><span class="hint">${entry.saved_on
      ? `Saved ${escapeHtml(entry.saved_on)} · ${entry.origin === "voice" ? "By voice" : "In Chat"}`
      : origins[entry.origin]}${entry.edited ? " · corrected" : ""}</span>
    <button type="button" class="btn small quiet" data-memory-action="edit">Edit</button>
    <button type="button" class="btn small quiet" data-memory-action="forget">Forget</button></div></div>`).join("");
}

async function entryAction(event) {
  const button = event.target.closest("[data-memory-action]");
  if (!button || persona.saving) return;
  const action = button.dataset.memoryAction;
  const row = button.closest("[data-entry]");
  const entry = persona.payload.memory.entries.find((item) => item.id === Number(row.dataset.entry));
  if (action === "edit") {
    delete $("#memory-entries").dataset.revision;
    persona.edit = { id: entry.id, revision: persona.payload.memory.revision };
    row.innerHTML = `<label class="field">Correct saved ${entry.saved_on ? "memory" : "fact"}<input data-edit-value value="${escapeHtml(entry.text)}"></label>
      <div class="row"><button type="button" class="btn small primary" data-memory-action="apply">Save correction</button>
      <button type="button" class="btn small quiet" data-memory-action="cancel">Cancel</button></div>`;
    row.querySelector("input").focus();
    setButtons();
    return;
  }
  if (action === "cancel") {
    persona.edit = null;
    render(persona.payload);
    return;
  }
  if (action === "forget" && !confirm("Forget this memory in future conversations?")) return;
  const revision = persona.edit?.revision ?? persona.payload.memory.revision;
  const id = persona.edit?.id ?? entry.id;
  const url = `/api/memory/entries/${id}`;
  const input = row.querySelector("input");
  persona.saving = true;
  persona.revision++;
  setButtons();
  try {
    const payload = action === "forget"
      ? await api(`${url}?revision=${revision}`, { method: "DELETE" })
      : await api(url, { method: "PUT", body: JSON.stringify({ revision, content: input.value }) });
    persona.edit = null;
    render(payload);
    toast(action === "forget" ? "Memory forgotten" : "Correction saved", "ok");
  } finally {
    persona.saving = false;
    setButtons();
  }
}

function setButtons() {
  $("#memory-refresh").disabled = persona.saving;
  $("#memory-content").disabled = Boolean(persona.edit);
  for (const button of document.querySelectorAll("#memory-entries button")) {
    button.disabled = persona.saving || persona.memoryDirty ||
      (Boolean(persona.edit) && !["apply", "cancel"].includes(button.dataset.memoryAction));
  }
  const editing = document.querySelector("#memory-entries input");
  if (editing) editing.disabled = persona.saving;
  $("#persona-reset").disabled = persona.saving || !persona.payload ||
    persona.payload.content.trim() === persona.payload.default.trim();
  for (const [id, dirty] of [["#persona-save", persona.dirty], ["#memory-save", persona.memoryDirty]]) {
    const button = $(id);
    button.disabled = persona.saving || !persona.payload || (id === "#memory-save" && Boolean(persona.edit));
    button.classList.toggle("primary", !dirty);
    button.classList.toggle("attention", dirty);
    button.textContent = dirty ? "Save changes" : "Save";
  }
}

export async function loadPersona() {
  if (persona.saving) return;
  const revision = ++persona.revision;
  const payload = await api("/api/persona");
  if (revision === persona.revision) render(payload);
}

async function save(field, reset = false) {
  if (persona.saving || !persona.payload) return;
  if (reset && !confirm("Replace your persona with the default? Your current text will be lost.")) return;
  const input = $(field === "memory" ? "#memory-content" : "#persona-content");
  const content = input.value;
  persona.saving = true;
  persona.revision++;
  setButtons();
  try {
    const payload = reset
      ? await api("/api/persona/reset", { method: "POST" })
      : await api(`/api/${field}`, { method: "PUT", body: JSON.stringify(field === "memory" ? { content, revision: persona.memoryRevision } : { content }) });
    // Keep typing done during the save and any unfinished edits in the other editor.
    if (input.value === content) {
      if (field === "memory") persona.memoryDirty = false;
      else persona.dirty = false;
    }
    if (field === "memory") persona.memoryRevision = payload.memory.revision;
    render(payload);
    toast(reset ? "Default persona restored" : field === "memory" ? "Memory saved" :
      "Persona saved. It applies from the next answer.", "ok");
  } finally {
    persona.saving = false;
    setButtons();
  }
}

export function initPersona() {
  setButtons();
  $("#persona-save").addEventListener("click", guard(() => save("persona")));
  $("#persona-reset").addEventListener("click", guard(() => save("persona", true)));
  $("#memory-entries").addEventListener("click", guard(entryAction));
  $("#memory-refresh").addEventListener("click", guard(async () => {
    if ((persona.memoryDirty || persona.edit) && !confirm("Discard your unsaved memory edit and reload?")) return;
    persona.saving = true;
    persona.revision++;
    setButtons();
    $("#memory-content").disabled = true;
    try {
      const payload = await api("/api/persona");
      persona.memoryDirty = false;
      persona.edit = null;
      render(payload);
    } finally {
      persona.saving = false;
      setButtons();
    }
  }));
  $("#memory-save").addEventListener("click", guard(() => save("memory")));
  $("#persona-content").addEventListener("input", () => {
    persona.dirty = true;
    setButtons();
  });
  $("#memory-content").addEventListener("input", () => {
    persona.memoryDirty = true;
    setButtons();
  });
}
