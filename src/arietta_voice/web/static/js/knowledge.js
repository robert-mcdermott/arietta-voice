// Knowledge: browse, edit and test the Markdown files the assistant can draw on.

import { $, api, escapeHtml, formatBytes, guard, relativeTime, toast } from "./core.js";

const knowledge = { files: [], current: null, dirty: false };

export function knowledgeDirty() {
  return knowledge.dirty;
}

export function knowledgeEditing() {
  return Boolean(knowledge.current);
}

function fileUrl(path) {
  return `/api/knowledge/files/${path.split("/").map(encodeURIComponent).join("/")}`;
}

function setStatus(text) {
  $("#knowledge-meta").textContent = text;
}

function setDirty(dirty) {
  knowledge.dirty = dirty;
  if (knowledge.current) {
    setStatus(dirty ? `${knowledge.current} · unsaved changes` : knowledge.current);
  }
}

function renderFiles() {
  const filter = $("#knowledge-filter").value.trim().toLowerCase();
  const files = knowledge.files.filter((file) => file.path.toLowerCase().includes(filter));
  $("#knowledge-count").textContent = `${knowledge.files.length} file${knowledge.files.length === 1 ? "" : "s"}`;
  $("#knowledge-file-list").innerHTML = files.length
    ? `<div class="file-list">${files
        .map(
          (file) => `<button type="button" class="file-item ${file.path === knowledge.current ? "active" : ""}" data-knowledge-file="${escapeHtml(file.path)}" title="${escapeHtml(file.path)}">
            <strong>${escapeHtml(file.path)}</strong>
            <span>${escapeHtml(formatBytes(file.size_bytes))} · ${escapeHtml(relativeTime(file.updated_at))}</span>
          </button>`,
        )
        .join("")}</div>`
    : `<p class="empty">${knowledge.files.length ? "No files match." : "No knowledge files yet."}</p>`;
}

export async function loadKnowledgeFiles() {
  const payload = await api("/api/knowledge/files");
  knowledge.files = payload.files || [];
  renderFiles();
}

async function openFile(path) {
  if (knowledge.dirty && !confirm("Discard unsaved edits and open another file?")) return;
  const payload = await api(fileUrl(path));
  knowledge.current = payload.path;
  $("#knowledge-file-path").value = payload.path;
  $("#knowledge-file-content").value = payload.content;
  setDirty(false);
  renderFiles();
}

function newFile() {
  const path = $("#knowledge-file-path").value.trim();
  if (!path) {
    setStatus("Enter a file path first, for example notes/example.md.");
    $("#knowledge-file-path").focus();
    return;
  }
  if (knowledge.dirty && !confirm("Discard unsaved edits and start a new file?")) return;
  knowledge.current = path;
  const title = path
    .split("/")
    .pop()
    .replace(/\.(md|markdown|txt)$/i, "")
    .replace(/[-_]+/g, " ");
  $("#knowledge-file-content").value = `---\ntitle: ${title}\ntags: \n---\n\n## Notes\n\n`;
  setDirty(true);
  renderFiles();
  $("#knowledge-file-content").focus();
}

async function saveFile() {
  const path = ($("#knowledge-file-path").value || knowledge.current || "").trim();
  if (!path) {
    setStatus("Enter a file path before saving.");
    return;
  }
  const payload = await api(fileUrl(path), {
    method: "PUT",
    body: JSON.stringify({ content: $("#knowledge-file-content").value }),
  });
  knowledge.current = payload.path;
  $("#knowledge-file-path").value = payload.path;
  setDirty(false);
  toast(`Saved ${payload.path}`, "ok");
  await loadKnowledgeFiles();
}

async function deleteFile() {
  const path = ($("#knowledge-file-path").value || knowledge.current || "").trim();
  if (!path) return;
  if (!confirm(`Delete ${path}? This cannot be undone here.`)) return;
  await api(fileUrl(path), { method: "DELETE" });
  knowledge.current = null;
  knowledge.dirty = false;
  $("#knowledge-file-path").value = "";
  $("#knowledge-file-content").value = "";
  setStatus("Open a file, or type a new path and choose New.");
  toast(`Deleted ${path}`, "ok");
  await loadKnowledgeFiles();
}

async function search() {
  const query = $("#knowledge-query").value.trim();
  if (!query) return;
  const payload = await api("/api/knowledge/search", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
  const matches = payload.matches || [];
  const summary = `<p class="hint">${matches.length} match${matches.length === 1 ? "" : "es"} using ${escapeHtml(payload.backend)} search</p>`;
  $("#knowledge-results").innerHTML =
    summary +
    matches
      .map(
        (match) => `<div class="result-card">
          <h4>${escapeHtml(match.title)}${match.section ? ` <span class="muted">· ${escapeHtml(match.section)}</span>` : ""}</h4>
          <div class="row"><span class="pill">${escapeHtml(String(match.source).split("/").pop())}</span><span class="pill accent">score ${escapeHtml(Number(match.score).toFixed(2))}</span></div>
          <p>${escapeHtml(match.snippet)}</p>
        </div>`,
      )
      .join("");
}

async function rebuildIndex() {
  const button = $("#knowledge-index");
  button.disabled = true;
  button.textContent = "Rebuilding…";
  try {
    const payload = await api("/api/knowledge/index", { method: "POST" });
    const count = payload.chunk_count ?? payload.chunks ?? payload.sections;
    toast(count == null ? "Index rebuilt" : `Index rebuilt · ${count} sections`, "ok");
  } finally {
    button.disabled = false;
    button.textContent = "Rebuild index";
  }
}

export function initKnowledge() {
  $("#knowledge-filter").addEventListener("input", renderFiles);
  $("#knowledge-file-list").addEventListener("click", (event) => {
    const target = event.target.closest("[data-knowledge-file]");
    if (target) guard(openFile)(target.dataset.knowledgeFile);
  });
  $("#knowledge-file-new").addEventListener("click", newFile);
  $("#knowledge-file-save").addEventListener("click", guard(saveFile));
  $("#knowledge-file-delete").addEventListener("click", guard(deleteFile));
  $("#knowledge-file-content").addEventListener("input", () => {
    if (!knowledge.dirty) setDirty(true);
  });
  $("#knowledge-file-content").addEventListener("keydown", (event) => {
    if (event.key === "s" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      guard(saveFile)();
    }
  });
  $("#knowledge-search").addEventListener("click", guard(search));
  $("#knowledge-query").addEventListener("keydown", (event) => {
    if (event.key === "Enter") guard(search)();
  });
  $("#knowledge-index").addEventListener("click", guard(rebuildIndex));
}
