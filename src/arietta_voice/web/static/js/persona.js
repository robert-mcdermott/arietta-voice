// Persona and memory: the text the assistant is given before every answer.

import { $, api, guard, toast } from "./core.js";

const persona = { payload: null, dirty: false, memoryDirty: false, saving: false, revision: 0 };

export function personaDirty() {
  return persona.dirty || persona.memoryDirty;
}

function render(payload) {
  persona.payload = payload;
  if (!persona.dirty) $("#persona-content").value = payload.content;
  $("#persona-file").textContent = payload.private
    ? `Your private persona: ${payload.file}`
    : `Using the default persona. Saving writes your own private copy; the default stays as it is.`;
  const memory = payload.memory;
  if (!persona.memoryDirty) $("#memory-content").value = memory.content;
  $("#memory-file").textContent = memory.enabled
    ? `Memory is on. The assistant adds facts here as you talk: ${memory.file}`
    : `Memory is off in Settings, so this file is not used. ${memory.file}`;
  setButtons();
}

function setButtons() {
  $("#persona-reset").disabled = persona.saving || !persona.payload ||
    persona.payload.content.trim() === persona.payload.default.trim();
  for (const [id, dirty] of [["#persona-save", persona.dirty], ["#memory-save", persona.memoryDirty]]) {
    const button = $(id);
    button.disabled = persona.saving || !persona.payload;
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
      : await api(`/api/${field}`, { method: "PUT", body: JSON.stringify({ content }) });
    // Keep typing done during the save and any unfinished edits in the other editor.
    if (input.value === content) {
      if (field === "memory") persona.memoryDirty = false;
      else persona.dirty = false;
    }
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
