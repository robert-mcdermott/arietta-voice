// Home: timers and reminders. The voice runtime says them; this card lists,
// adds and cancels them, and counts down between polls.

import { $, $$, api, escapeHtml, guard, toast } from "./core.js";

const TIMER_TOOLS = ["set_timer", "set_reminder", "set_alarm", "check_timers", "cancel_timer"];
const PLACEHOLDERS = {
  timer: ["10 minutes", "How long", "What for (optional)"],
  reminder: ["at 5 pm, in 20 minutes, tomorrow at 9", "When", "What to remind you, such as to call Mom"],
  alarm: ["7 am, tomorrow at 6:30", "When", "Name (optional)"],
};
const OUTCOMES = {
  dismissed: "Stopped",
  acknowledged: "Heard",
  unanswered: "Not answered",
  cancelled: "Cancelled",
};

const timers = { data: null, kind: "timer", offsetMs: 0, refreshing: false };

function nowSeconds() {
  return (Date.now() + timers.offsetMs) / 1000;
}

function countdown(seconds) {
  const value = Math.max(0, Math.ceil(seconds));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const secs = String(value % 60).padStart(2, "0");
  if (hours >= 24) return `${Math.floor(hours / 24)}d ${hours % 24}h`;
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}` : `${minutes}:${secs}`;
}

function capitalize(text) {
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

function title(alert) {
  return alert.kind === "reminder" ? `Reminder ${alert.name}` : capitalize(alert.name);
}

function activeItem(alert) {
  const ringing = alert.status === "ringing";
  const soon = alert.kind === "timer" || alert.remaining_s < 86400;
  const status = ringing
    ? `<span class="pill bad">Going off</span>`
    : `<span class="hint">${escapeHtml(alert.kind === "timer" ? `Ends ${alert.due_text}` : capitalize(alert.due_text))}</span>`;
  const count =
    !ringing && soon
      ? `<span class="timer-count" data-due="${alert.due_at}">${countdown(alert.remaining_s)}</span>`
      : "";
  return `<li class="timer-item${ringing ? " ringing" : ""}">
    <div><strong>${escapeHtml(title(alert))}</strong><div class="timer-sub">${status}</div></div>
    ${count}
    <button class="btn small" type="button" data-cancel="${escapeHtml(alert.id)}">${ringing ? "Stop" : "Cancel"}</button>
  </li>`;
}

function finishedItem(alert) {
  const outcome = OUTCOMES[alert.outcome] || "Finished";
  return `<li class="timer-item"><div><strong>${escapeHtml(title(alert))}</strong>
    <div class="timer-sub hint">${escapeHtml(outcome)}</div></div></li>`;
}

export function renderTimers(data) {
  timers.data = data;
  timers.offsetMs = data.now * 1000 - Date.now();
  const active = data.active || [];
  const finished = data.finished || [];
  $("#timers-list").innerHTML = active.length
    ? active.map(activeItem).join("")
    : `<li class="empty">No timers, reminders or alarms.</li>`;
  $("#timers-stop").hidden = !active.some((alert) => alert.status === "ringing");

  const notices = [];
  const missing = TIMER_TOOLS.filter((name) => !(data.tools_enabled || []).includes(name));
  if (missing.length) {
    notices.push(
      `Asking by voice needs ${missing.join(", ")} turned on. Turn ${missing.length === 1 ? "it" : "them"} on in Tools.`,
    );
  }
  if (active.length && !data.runtime_running) {
    notices.push(
      "The voice assistant is not running, so nothing will sound until it starts. Anything missed is announced then.",
    );
  }
  $("#timers-notice").hidden = !notices.length;
  $("#timers-notice").textContent = notices.join(" ");

  $("#timers-finished").hidden = !finished.length;
  $("#timers-finished-title").textContent = `Earlier (${finished.length})`;
  $("#timers-finished-list").innerHTML = finished.map(finishedItem).join("");
}

export async function loadTimers() {
  if (timers.refreshing) return;
  timers.refreshing = true;
  try {
    renderTimers(await api("/api/timers"));
  } finally {
    timers.refreshing = false;
  }
}

// Count down locally, and look again once something reaches zero.
function tick() {
  let due = false;
  $$(".timer-count").forEach((element) => {
    const left = Number(element.dataset.due) - nowSeconds();
    element.textContent = countdown(left);
    if (left <= 0) due = true;
  });
  if (due && !document.hidden) guard(loadTimers)();
}

function setKind(kind) {
  timers.kind = kind;
  $$(".segmented .seg").forEach((button) => {
    const active = button.dataset.kind === kind;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  const [whenHint, whenLabel, labelHint] = PLACEHOLDERS[kind];
  $("#timers-when").placeholder = whenHint;
  $("#timers-when").setAttribute("aria-label", whenLabel);
  $("#timers-label").placeholder = labelHint;
}

async function addTimer() {
  const when = $("#timers-when").value.trim();
  const label = $("#timers-label").value.trim();
  if (!when) {
    $("#timers-when").focus();
    return;
  }
  if (timers.kind === "reminder" && !label) {
    $("#timers-label").focus();
    toast("Say what the reminder is for.", "bad");
    return;
  }
  const body =
    timers.kind === "timer"
      ? { kind: "timer", duration: when, label }
      : { kind: timers.kind, when, label };
  const button = $("#timers-add");
  button.disabled = true;
  try {
    const payload = await api("/api/timers", { method: "POST", body: JSON.stringify(body) });
    renderTimers(payload);
    $("#timers-when").value = "";
    $("#timers-label").value = "";
    toast(`${title(payload.added)} added`, "ok");
  } catch (error) {
    toast(error.message.replace(/ \(400\)$/, ""), "bad");
  } finally {
    button.disabled = false;
  }
}

async function cancelTimer(id) {
  renderTimers(await api(`/api/timers/${encodeURIComponent(id)}/cancel`, { method: "POST" }));
}

export function initTimers() {
  $$(".segmented .seg").forEach((button) =>
    button.addEventListener("click", () => setKind(button.dataset.kind)),
  );
  $("#timers-add").addEventListener("click", guard(addTimer));
  for (const selector of ["#timers-when", "#timers-label"]) {
    $(selector).addEventListener("keydown", (event) => {
      if (event.key === "Enter") guard(addTimer)();
    });
  }
  $("#timers-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-cancel]");
    if (button) guard(cancelTimer)(button.dataset.cancel);
  });
  $("#timers-stop").addEventListener(
    "click",
    guard(async () => renderTimers(await api("/api/timers/stop", { method: "POST" }))),
  );
  setInterval(tick, 1000);
}
