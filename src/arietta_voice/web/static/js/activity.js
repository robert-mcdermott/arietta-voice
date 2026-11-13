// Activity: the event log, described in plain language with details on demand.

import { $, api, describeEvent, escapeHtml, guard, localDateTime, table, words } from "./core.js";

const log = { offset: 0, limit: 50 };
const HIDDEN_KEYS = new Set(["ts", "event", "content"]);

function queryString() {
  const params = new URLSearchParams({ limit: String(log.limit), offset: String(log.offset) });
  const event = $("#logs-event").value.trim();
  const query = $("#logs-search").value.trim();
  if (event) params.set("event", event);
  if (query) params.set("q", query);
  return params.toString();
}

function chipValue(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function details(item) {
  const chips = Object.entries(item)
    .filter(([key]) => !HIDDEN_KEYS.has(key))
    .map(([key, value]) => {
      const text = chipValue(value);
      return `<span class="chip" title="${escapeHtml(text)}"><b>${escapeHtml(words(key))}</b><span>${escapeHtml(text)}</span></span>`;
    });
  if (item.content && typeof item.content === "object") {
    for (const [key, value] of Object.entries(item.content)) {
      if (value === null || value === undefined || value === "") continue;
      chips.push(
        `<span class="chip" title="${escapeHtml(chipValue(value))}"><b>${escapeHtml(words(key))}</b><span>${escapeHtml(chipValue(value))}</span></span>`,
      );
    }
  }
  return chips.length ? `<div class="event-chips">${chips.join("")}</div>` : "";
}

export async function loadActivity({ reset = false } = {}) {
  if (reset) log.offset = 0;
  const payload = await api(`/api/logs?${queryString()}`);
  const rows = (payload.items || []).map((item) => [
    `<span class="muted" title="${escapeHtml(item.ts || "")}">${escapeHtml(localDateTime(item.ts))}</span>`,
    `<span class="event-name">${escapeHtml(describeEvent(item))}</span><br><code class="muted">${escapeHtml(item.event || "")}</code>`,
    details(item),
  ]);
  $("#logs-content").innerHTML = rows.length
    ? table(["When", "What happened", "Details"], rows)
    : `<p class="empty">No events match.</p>`;
  const start = payload.total ? payload.offset + 1 : 0;
  const end = payload.offset + payload.count;
  $("#logs-page-info").textContent = `${start}–${end} of ${payload.total} events, newest first`;
  $("#logs-prev").disabled = !payload.has_previous;
  $("#logs-next").disabled = !payload.has_next;
}

export function activityOnFirstPage() {
  return log.offset === 0;
}

export function initActivity() {
  const reload = guard(() => loadActivity({ reset: true }));
  $("#logs-apply").addEventListener("click", reload);
  $("#logs-clear").addEventListener("click", () => {
    $("#logs-search").value = "";
    $("#logs-event").value = "";
    reload();
  });
  for (const selector of ["#logs-search", "#logs-event"]) {
    $(selector).addEventListener("keydown", (event) => {
      if (event.key === "Enter") reload();
    });
  }
  $("#logs-prev").addEventListener("click", () => {
    log.offset = Math.max(0, log.offset - log.limit);
    guard(loadActivity)();
  });
  $("#logs-next").addEventListener("click", () => {
    log.offset += log.limit;
    guard(loadActivity)();
  });
}
