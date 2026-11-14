// Persona and memory: the text the assistant is given before every answer.

import { $, api, guard, toast } from "./core.js";

const persona = { payload: null, dirty: false, memoryDirty: false };

export function personaDirty() {
  return persona.dirty || persona.memoryDirty;
}

function render(payload) {
  persona.payload = payload;
  $("#persona-content").value = payload.content;
  $("#persona-file").textContent = payload.private
    ? `Your private persona: ${payload.file}`
    : `Using the default persona. Saving writes your own private copy; the default stays as it is.`;
  $("#persona-reset").disabled = payload.content.trim() === payload.default.trim();
  const memory = payload.memory;
  $("#memory-content").value = memory.content;
  $("#memory-file").textContent = memory.enabled
    ? `Memory is on. The assistant adds facts here as you talk: ${memory.file}`
    : `Memory is off in Settings, so this file is not used. ${memory.file}`;
  persona.dirty = false;
  persona.memoryDirty = false;
  setButtons();
}

function setButtons() {
  for (const [id, dirty] of [["#persona-save", persona.dirty], ["#memory-save", persona.memoryDirty]]) {
    const button = $(id);
    button.classList.toggle("primary", !dirty);
    button.classList.toggle("attention", dirty);
    button.textContent = dirty ? "Save changes" : "Save";
  }
}

export async function loadPersona() {
  if (personaDirty()) return;
  render(await api("/api/persona"));
}

async function save() {
  const content = $("#persona-content").value;
  render(await api("/api/persona", { method: "PUT", body: JSON.stringify({ content }) }));
  toast("Persona saved. It applies from the next answer.", "ok");
}

async function reset() {
  if (!confirm("Replace your persona with the default? Your current text will be lost.")) return;
  render(await api("/api/persona/reset", { method: "POST" }));
  toast("Default persona restored", "ok");
}

async function saveMemory() {
  const content = $("#memory-content").value;
  render(await api("/api/memory", { method: "PUT", body: JSON.stringify({ content }) }));
  toast("Memory saved", "ok");
}

export function initPersona() {
  $("#persona-save").addEventListener("click", guard(save));
  $("#persona-reset").addEventListener("click", guard(reset));
  $("#memory-save").addEventListener("click", guard(saveMemory));
  $("#persona-content").addEventListener("input", () => {
    persona.dirty = true;
    setButtons();
  });
  $("#memory-content").addEventListener("input", () => {
    persona.memoryDirty = true;
    setButtons();
  });
}
