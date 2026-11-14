// Shared helpers: DOM access, API calls, formatting, notices.

export const $ = (selector) => document.querySelector(selector);
export const $$ = (selector) => [...document.querySelectorAll(selector)];

export async function api(path, options = {}) {
  const headers = Object.assign({ "Content-Type": "application/json" }, options.headers || {});
  const response = await fetch(path, Object.assign({}, options, { headers }));
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : response.statusText;
    const error = new Error(
      response.status === 401
        ? "Your session needs authentication. Reload this page to sign in."
        : `${detail} (${response.status})`,
    );
    error.status = response.status;
    throw error;
  }
  return response.json();
}

export function escapeHtml(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char],
  );
}

export function table(headers, rows) {
  const head = headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("");
  const body = rows
    .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
    .join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

export function setError(message) {
  $("#health-dot").classList.remove("ok");
  $("#health-dot").classList.add("bad");
  $("#error-message").textContent = message;
  $("#error-notice").hidden = false;
}

// Wrap an async handler so failures reach the error banner instead of the console.
export function guard(handler) {
  return (...args) => Promise.resolve(handler(...args)).catch((error) => setError(error.message));
}

export function toast(message, tone = "") {
  const element = document.createElement("div");
  element.className = `toast ${tone}`.trim();
  element.textContent = message;
  $("#toasts").append(element);
  setTimeout(() => element.remove(), tone === "bad" ? 7000 : 3500);
}

export function showResult(prefix, label, payload) {
  $(`#${prefix}-result-title`).textContent = label;
  $(`#${prefix}-result`).textContent =
    typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
  $(`#${prefix}-result-box`).hidden = false;
}

export function readJsonTextarea(selector) {
  const text = $(selector).value.trim();
  return text ? JSON.parse(text) : {};
}

export function writeJsonTextarea(selector, value) {
  $(selector).value = JSON.stringify(value || {}, null, 2);
}

export function clockTime(date = new Date()) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function formatDuration(seconds) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  if (value < 90) return `${value}s`;
  if (value < 3600) return `${Math.floor(value / 60)}m ${value % 60}s`;
  if (value < 86400) return `${Math.floor(value / 3600)}h ${Math.floor((value % 3600) / 60)}m`;
  return `${Math.floor(value / 86400)}d ${Math.floor((value % 86400) / 3600)}h`;
}

export function relativeTime(iso) {
  const then = Date.parse(iso || "");
  if (Number.isNaN(then)) return "";
  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 86400 * 14) return `${Math.floor(seconds / 86400)}d ago`;
  return new Date(then).toLocaleDateString([], { year: "numeric", month: "short", day: "numeric" });
}

export function localDateTime(iso) {
  const then = Date.parse(iso || "");
  if (Number.isNaN(then)) return iso || "";
  return new Date(then).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

export function words(value) {
  return String(value ?? "").replaceAll("_", " ");
}

const STATE_WORDS = {
  waiting_for_wake: "waiting for the wake word",
  waiting_for_audio: "waiting for an audio device",
};

function stateWords(value) {
  return STATE_WORDS[value] || words(value);
}

// Turn a logged event into a short sentence. Unknown events fall back to their name.
export function describeEvent(item) {
  const event = item.event || "event";
  switch (event) {
    case "state_change":
      return `Now ${stateWords(item.to_state)}`;
    case "wake_detected":
      return item.phrase ? `Wake word heard (“${item.phrase}”)` : "Wake word heard";
    case "wake_miss":
      return "Speech ignored (no wake word)";
    case "session_started":
      return "Conversation started";
    case "session_ended":
      return `Conversation ended${item.reason ? ` (${words(item.reason)})` : ""}`;
    case "user_transcript":
      return item.latency_ms == null
        ? "Request transcribed"
        : `Request transcribed in ${item.latency_ms} ms`;
    case "assistant_response":
      return "Answer composed";
    case "assistant_audio_complete":
      return item.interrupted ? "Speech interrupted" : "Finished speaking";
    case "tool_call":
      return `Used tool ${item.tool || ""}`.trim();
    case "alert":
      return item.repeat ? "Alert repeated" : `${item.count > 1 ? "Alerts" : "Alert"} announced`;
    case "alert_answered":
      return item.action === "snooze" ? "Alert snoozed" : "Alert stopped";
    case "alert_failed":
      return "Alert could not be announced";
    case "smart_turn_score":
      return "Checked whether the speaker finished";
    case "turn_failed":
      return `Turn failed${item.stage ? ` while ${words(item.stage)}` : ""}`;
    case "turn_cancelled":
      return "Turn cancelled";
    case "audio_connected":
      return item.recovery ? "Audio device recovered" : "Audio device connected";
    case "audio_disconnected":
      return "Audio device lost";
    case "audio_gap":
      return "Audio gap detected";
    case "utterance_limit":
      return "Utterance too long, discarded";
    case "startup_diagnostics":
      return "Startup checks run";
    case "api_chat":
      return "Console chat turn";
    case "api_error":
      return "Console request failed";
    case "api_runtime_start":
      return "Voice runtime started from console";
    case "api_runtime_stop":
      return "Voice runtime stopped from console";
    case "api_runtime_restart":
      return "Voice runtime restarted from console";
    default:
      return words(event).replace(/^./, (char) => char.toUpperCase());
  }
}
