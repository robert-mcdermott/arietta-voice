// Home: live assistant status, runtime controls, last exchange, recent activity.

import { $, api, clockTime, describeEvent, escapeHtml, relativeTime, toast } from "./core.js";
import { renderSystem } from "./system.js";

export const overview = {
  state: null,
  runtime: null,
  contentLogging: null,
  loading: false,
  actionPending: null,
};

const STEP_FOR_STAGE = {
  endpoint: "listen",
  transcribe: "transcribe",
  chat: "think",
  tts_start: "speak",
  tts_next: "speak",
};
const STEP_FOR_STATE = { listening: "listen", thinking: "think", speaking: "speak" };

function capitalize(value) {
  const text = String(value || "");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function describeRuntime(runtime, assistant) {
  const wake = capitalize(assistant?.wake_words?.[0] || assistant?.name || "the wake word");
  const wakeEnabled = assistant?.wake_enabled !== false;
  if (overview.actionPending) {
    const verb = { start: "Starting", stop: "Stopping", restart: "Restarting" }[
      overview.actionPending
    ];
    return { orb: "starting", title: `${verb}…`, sub: "This can take a moment while models load." };
  }
  if (!runtime.is_running) {
    if (runtime.lifecycle === "starting") {
      return { orb: "starting", title: "Starting up…", sub: runtime.note };
    }
    return { orb: "offline", title: "Not running", sub: runtime.note };
  }
  const turns = Number(runtime.session_turn_count || 0);
  const session = runtime.session_active
    ? `In a conversation · ${turns} turn${turns === 1 ? "" : "s"} so far.`
    : "";
  switch (runtime.state) {
    case "booting":
      return { orb: "starting", title: "Loading models…", sub: "Almost ready to listen." };
    case "waiting_for_wake":
      return {
        orb: "idle",
        title: `Waiting for “${wake}”`,
        sub: "Say the wake word to start a conversation.",
      };
    case "listening":
      return {
        orb: "listening",
        title: "Listening",
        sub: session || (wakeEnabled ? "" : "Wake word is off, so every utterance is heard."),
      };
    case "thinking":
      return { orb: "thinking", title: "Thinking…", sub: session };
    case "speaking":
      return { orb: "speaking", title: "Speaking", sub: session };
    case "waiting_for_audio":
      return {
        orb: "error",
        title: "Audio device unavailable",
        sub: "Retrying the microphone and speaker connection.",
      };
    case "stopping":
      return { orb: "offline", title: "Stopping…", sub: "" };
    default:
      return { orb: "idle", title: capitalize(String(runtime.state).replaceAll("_", " ")), sub: "" };
  }
}

function renderControls(runtime) {
  const start = $("#runtime-start");
  const stop = $("#runtime-stop");
  const restart = $("#runtime-restart");
  const managed = Boolean(runtime.managed_process_running);
  const starting = runtime.lifecycle === "starting" || (managed && !runtime.is_running);
  const canStart =
    !managed && (!runtime.is_running || runtime.is_stale || runtime.lifecycle === "stopped");
  const pending = Boolean(overview.actionPending);

  start.hidden = !canStart && !pending;
  stop.hidden = !managed;
  restart.hidden = !managed;
  start.disabled = pending || !canStart;
  stop.disabled = pending;
  restart.disabled = pending || starting;
  start.textContent = overview.actionPending === "start" ? "Starting…" : "Start listening";
  stop.textContent = overview.actionPending === "stop" ? "Stopping…" : "Stop";
  restart.textContent = overview.actionPending === "restart" ? "Restarting…" : "Restart";
  if (pending && !canStart) start.hidden = true;
}

function renderSteps(runtime) {
  const steps = $("#turn-steps");
  steps.hidden = !runtime.is_running;
  const stage = runtime.reliability?.turn_stage;
  const active = STEP_FOR_STAGE[stage] || STEP_FOR_STATE[runtime.state] || null;
  steps.querySelectorAll("li").forEach((item) => {
    item.classList.toggle("active", item.dataset.step === active);
  });
}

function renderWarnings(runtime) {
  const data = runtime.reliability || {};
  const audio = data.audio || {};
  const devices = data.devices || {};
  const warnings = [];
  if (runtime.is_running && runtime.state === "waiting_for_audio") {
    warnings.push(
      `Audio unavailable (${devices.last_error || "unknown"}). The interrupted turn was discarded; check any external action before repeating it.`,
    );
  } else if (runtime.last_error) {
    warnings.push(
      `Last error: ${runtime.last_error}. No automatic retry was made; check any external action before repeating it.`,
    );
  }
  if (audio.recording_limit_reached) {
    warnings.push("Recording reached its duration limit; voice interaction can continue.");
  }
  if (audio.recording_error) {
    warnings.push(`Recording stopped: ${audio.recording_error}. The recording may be incomplete.`);
  }
  if (audio.recording_dropped_chunks) {
    warnings.push("Recording lost audio under load; the saved file has gaps.");
  }
  if (runtime.is_running && !runtime.managed_process_running) {
    warnings.push(
      "This runtime was started from a terminal, so it can only be stopped or restarted there.",
    );
  }
  $("#runtime-warning").hidden = !warnings.length;
  $("#runtime-warning").textContent = warnings.join(" ");
}

function renderExchange(runtime, assistant) {
  const user = (runtime.last_user_transcript || "").trim();
  const reply = (runtime.last_assistant_text || "").trim();
  const name = assistant?.name || "Assistant";
  const hint = $("#exchange-hint");
  if (!user && !reply) {
    hint.textContent = "";
    $("#last-exchange").innerHTML =
      overview.contentLogging === false
        ? `<p class="muted">Conversation text is hidden. Set <code>logging.content_enabled = true</code> in your private config to show the current exchange here.</p>`
        : `<p class="muted">Nothing yet. The current exchange appears here during a conversation.</p>`;
    return;
  }
  hint.textContent = runtime.session_active ? "Live" : "";
  $("#last-exchange").innerHTML = `
    ${user ? `<div class="exchange-line"><span class="exchange-who">You</span><span>${escapeHtml(user)}</span></div>` : ""}
    ${reply ? `<div class="exchange-line assistant"><span class="exchange-who">${escapeHtml(name)}</span><span>${escapeHtml(reply)}</span></div>` : ""}`;
}

export function renderRuntime(runtime) {
  overview.runtime = runtime;
  const assistant = overview.state?.assistant;
  const view = describeRuntime(runtime, assistant);
  const orb = $("#orb");
  orb.dataset.state = view.orb;
  orb.style.setProperty(
    "--level",
    runtime.display_speaking_active ? String(runtime.display_speech_level || 0) : "0",
  );
  $("#hero-title").textContent = view.title;
  $("#hero-sub").textContent = view.sub || "";
  const origin = runtime.is_running
    ? runtime.managed_process_running
      ? " · managed here"
      : " · started in a terminal"
    : "";
  const since =
    runtime.is_running && runtime.started_at ? ` · up since ${relativeTime(runtime.started_at)}` : "";
  $("#hero-kicker").textContent = `Voice runtime${origin}${since}`;
  renderControls(runtime);
  renderSteps(runtime);
  renderWarnings(runtime);
  renderExchange(runtime, assistant);
}

function renderFacts(state) {
  const { assistant, models, knowledge, tools } = state;
  $("#brand-sub").textContent = `${assistant.name} · console`;
  $("#app-version").textContent = `v${state.api.version}`;
  $("#metric-assistant").textContent = assistant.name;
  $("#metric-wake").textContent = assistant.wake_enabled
    ? `Wake word: ${assistant.wake_words.join(", ")}`
    : "Wake word off";
  $("#metric-model").textContent = models.chat_model.split("/").pop();
  $("#metric-model").title = models.chat_model;
  $("#metric-provider").textContent =
    models.chat_provider === "local" ? "Runs on this Mac" : `Cloud · ${models.chat_provider}`;
  $("#metric-knowledge").textContent = knowledge.enabled
    ? `${knowledge.backend.charAt(0).toUpperCase()}${knowledge.backend.slice(1)} search`
    : "Off";
  $("#metric-index").textContent = !knowledge.enabled
    ? "Retrieval is turned off"
    : knowledge.loaded
      ? "Index loaded"
      : "Loads on first use";
  $("#metric-index").title = knowledge.directory;
  const enabled = (tools.enabled || []).length;
  $("#metric-tools").textContent = `${enabled} instant tool${enabled === 1 ? "" : "s"}`;
  $("#metric-tools-sub").textContent = tools.error_count
    ? `${tools.error_count} with errors`
    : "No tool errors";
}

function renderRecentEvents(items) {
  $("#recent-events").innerHTML = items.length
    ? items
        .map(
          (item) => `<li><span>${escapeHtml(describeEvent(item))}</span>
            <time datetime="${escapeHtml(item.ts || "")}" title="${escapeHtml(item.ts || "")}">${escapeHtml(relativeTime(item.ts))}</time></li>`,
        )
        .join("")
    : `<li class="muted">No activity recorded yet.</li>`;
}

function setHealth(health) {
  const dot = $("#health-dot");
  dot.classList.remove("ok", "bad");
  if (health.status === "ok") dot.classList.add("ok");
  if (health.status === "error") dot.classList.add("bad");
  $("#health-text").textContent =
    health.status === "ok" ? "Console connected" : `Status: ${health.status || "unknown"}`;
}

export async function loadOverview() {
  if (overview.loading) return;
  overview.loading = true;
  try {
    const [health, state, logs] = await Promise.all([
      api("/api/health"),
      api("/api/state"),
      api("/api/logs?limit=7"),
    ]);
    overview.state = state;
    setHealth(health);
    renderFacts(state);
    renderRuntime(state.runtime);
    renderRecentEvents(logs.items || []);
    renderSystem(state);
    $("#poll-status").textContent = `Updated ${clockTime()}`;
  } finally {
    overview.loading = false;
  }
}

// A lighter poll for the hero: just the runtime heartbeat.
export async function pollRuntime() {
  if (overview.loading || overview.actionPending || !overview.state) return;
  const runtime = await api("/api/runtime");
  overview.state.runtime = runtime;
  renderRuntime(runtime);
}

export async function loadContentLoggingFlag() {
  const config = await api("/api/config");
  overview.contentLogging = Boolean(config.logging?.content_enabled);
  if (overview.runtime) renderRuntime(overview.runtime);
}

export async function runtimeAction(action) {
  overview.actionPending = action;
  if (overview.runtime) renderRuntime(overview.runtime);
  try {
    const payload = await api(`/api/runtime/${action}`, { method: "POST" });
    toast(payload.message || `Runtime ${action} requested`, "ok");
  } finally {
    overview.actionPending = null;
    await loadOverview();
  }
}
