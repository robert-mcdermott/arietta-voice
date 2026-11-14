// System (health, audio and process detail) and Settings (effective configuration).

import {
  $,
  $$,
  api,
  escapeHtml,
  formatDuration,
  guard,
  localDateTime,
  table,
  toast,
  words,
} from "./core.js";

const STAGE_NAMES = {
  idle: "Ready",
  transcribe: "Transcribing",
  endpoint: "Checking turn end",
  chat: "Thinking",
  tts_start: "Preparing speech",
  tts_next: "Speaking",
  memory: "Updating memory",
  loading_models: "Loading models",
  waiting_for_audio: "Waiting for audio",
  stopped: "Stopped",
};

function meter(title, value, body) {
  return `<div class="card meter-card"><h4>${escapeHtml(title)}</h4>
    <div class="meter-value">${escapeHtml(value)}</div>${body}</div>`;
}

function note(text) {
  return `<div class="meter-note">${text}</div>`;
}

export function renderSystem(state) {
  const runtime = state.runtime;
  const data = runtime.reliability || {};
  const audio = data.audio || {};
  const devices = data.devices || {};
  const active = Boolean(runtime.is_running);
  const waitingForAudio = runtime.state === "waiting_for_audio";
  const elapsed = Number(data.turn_elapsed_s || 0);
  const limit = Number(data.stage_timeout_s || data.turn_timeout_s || 60);
  const stage = active
    ? waitingForAudio
      ? "Waiting for audio"
      : STAGE_NAMES[data.turn_stage] || "Starting"
    : "Offline";
  const queued = Math.max(0, Number(audio.queued_chunks) || 0);
  const capacity = Math.max(0, Number(audio.queue_max_chunks) || 0);
  const failures = Number(data.turn_failures ?? 0);

  $("#runtime-reliability").innerHTML = [
    meter(
      "Current turn",
      stage,
      `<progress aria-label="Turn time budget used" max="${limit}" value="${Math.min(elapsed, limit)}"></progress>` +
        note(elapsed ? `${elapsed}s of ${limit}s budget` : `Time limit ${limit}s`) +
        note(
          `Last turn ${data.last_turn_duration_ms == null ? "—" : `${escapeHtml(data.last_turn_duration_ms)} ms`}`,
        ),
    ),
    meter(
      "Microphone buffer",
      capacity ? `${queued} / ${capacity}` : "—",
      `<progress aria-label="Microphone buffer usage" max="${capacity || 1}" value="${queued}"></progress>` +
        note(
          `Overruns ${escapeHtml(audio.dropped_chunks ?? 0)} queue · ${escapeHtml(audio.input_overflows ?? 0)} input`,
        ) +
        note(`${escapeHtml(audio.discarded_chunks ?? 0)} chunks discarded by design`),
    ),
    meter(
      "Recovery",
      `${failures} failed turn${failures === 1 ? "" : "s"}`,
      note(
        `Timeouts ${escapeHtml(data.turn_timeouts ?? 0)} · Worker resets ${escapeHtml(data.worker_restarts ?? 0)}`,
      ) +
        note(`Oversized utterances ${escapeHtml(data.utterances_discarded ?? 0)}`) +
        note(`Recording drops ${escapeHtml(audio.recording_dropped_chunks ?? 0)}`),
    ),
    meter(
      "Audio devices",
      words(devices.connection || "Unknown").replace(/^./, (char) => char.toUpperCase()),
      note(`In: ${escapeHtml(devices.input || "not connected")}`) +
        note(`Out: ${escapeHtml(devices.output || "not connected")}`) +
        note(
          `Failures ${escapeHtml(devices.reconnect_attempts ?? 0)} · Recoveries ${escapeHtml(devices.recoveries ?? 0)}`,
        ),
    ),
  ].join("");

  const inference = state.api.inference || {};
  $("#api-inference-note").textContent =
    `Console chat ${inference.operation ? "is processing a request" : "is ready"} · ${inference.turn_timeout_s || limit}s turn limit`;

  const rows = [
    ["State", `<strong>${escapeHtml(words(runtime.state))}</strong>`],
    ["Lifecycle", escapeHtml(runtime.lifecycle)],
    ["Managed by console", runtime.managed_process_running ? "Yes" : "No"],
    ["Process ID", escapeHtml(runtime.process_id || "—")],
    ["Started", escapeHtml(runtime.started_at ? localDateTime(runtime.started_at) : "—")],
    [
      "Last heartbeat",
      runtime.last_heartbeat_at
        ? `${escapeHtml(localDateTime(runtime.last_heartbeat_at))} · ${escapeHtml(formatDuration(runtime.heartbeat_age_seconds))} ago`
        : "—",
    ],
    [
      "Session",
      runtime.session_active
        ? `Active · ${escapeHtml(runtime.session_turn_count || 0)} turns`
        : "Inactive",
    ],
    ["Console uptime", escapeHtml(formatDuration(state.api.uptime_seconds))],
    ["Console chat turns", escapeHtml(state.api.chat_turns)],
    ["Python", escapeHtml(state.host.python)],
    ["Platform", escapeHtml(state.host.platform)],
    ["Status file", `<code>${escapeHtml(runtime.status_file)}</code>`],
    ["Runtime log", `<code>${escapeHtml(runtime.managed_stdout_log || "—")}</code>`],
    ["Event log", `<code>${escapeHtml(state.logs.events_file)}</code>`],
    ["Note", escapeHtml(runtime.note)],
  ];
  $("#runtime-detail").innerHTML = table(["Field", "Value"], rows);
}

function renderDiagnostics(payload) {
  const checks = payload.checks || [];
  const failing = checks.filter((check) => !check.ok);
  $("#diagnostics-summary").textContent = checks.length
    ? failing.length
      ? `${failing.length} of ${checks.length} checks need attention`
      : `All ${checks.length} checks pass`
    : "";
  const rows = checks.map((check) => [
    `<span class="pill ${check.ok ? "ok" : "bad"}">${escapeHtml(check.status)}</span>`,
    `<strong>${escapeHtml(words(check.name))}</strong>`,
    escapeHtml(check.detail),
  ]);
  $("#diagnostics-table").innerHTML = rows.length
    ? table(["Status", "Check", "Detail"], rows)
    : `<p class="empty">No checks reported.</p>`;
}

export async function loadDiagnostics({ hardware = false } = {}) {
  const query = hardware ? "?include_hardware=true" : "";
  renderDiagnostics(await api(`/api/diagnostics${query}`));
}

const SECTION_TITLES = {
  assistant: "Assistant",
  wake_word: "Wake word",
  audio: "Audio",
  runtime: "Turn limits",
  models: "Models",
  prompts: "Persona and memory files",
  memory: "Memory",
  logging: "Logging",
  knowledge: "Knowledge",
  tools: "Instant tools",
  model_tools: "Model tools",
  admin: "Console",
};

function settingValue(value) {
  if (value === null || value === undefined || value === "") return `<span class="muted">—</span>`;
  if (typeof value === "boolean") {
    return `<span class="pill ${value ? "ok" : ""}">${value ? "On" : "Off"}</span>`;
  }
  if (Array.isArray(value)) {
    return value.length
      ? `<div class="row">${value.map((item) => `<span class="pill">${escapeHtml(item)}</span>`).join("")}</div>`
      : `<span class="muted">None</span>`;
  }
  if (typeof value === "object") return `<code>${escapeHtml(JSON.stringify(value))}</code>`;
  return escapeHtml(value);
}

// Editable settings: one card per group, saved to the private config.

const settingsState = { groups: [], saving: false };

function fieldInput(field, id) {
  const value = field.value;
  if (field.kind === "boolean") {
    return `<label class="check"><input id="${id}" type="checkbox" ${value ? "checked" : ""}> On</label>`;
  }
  if (field.kind === "choice") {
    return `<div class="segmented" role="group" aria-label="${escapeHtml(field.label)}" id="${id}">${field.choices
      .map(
        (choice) =>
          `<button type="button" class="seg${choice === value ? " active" : ""}" data-choice="${escapeHtml(choice)}" aria-pressed="${choice === value}">${escapeHtml(words(choice).replace(/^./, (c) => c.toUpperCase()))}</button>`,
      )
      .join("")}</div>`;
  }
  if (field.kind === "number" || field.kind === "integer") {
    const step = field.kind === "integer" ? "1" : "any";
    const bounds = `${field.minimum != null ? ` min="${field.minimum}"` : ""}${field.maximum != null ? ` max="${field.maximum}"` : ""}`;
    return `<input id="${id}" type="number" step="${step}"${bounds} value="${escapeHtml(value)}">`;
  }
  const text = Array.isArray(value) ? value.join(", ") : value ?? "";
  return `<input id="${id}" type="text" maxlength="${field.kind === "list" ? 400 : field.max_length}" value="${escapeHtml(text)}">`;
}

function fieldId(field) {
  return `setting-${field.section}-${field.key}`;
}

function readField(field) {
  const element = document.getElementById(fieldId(field));
  if (field.kind === "boolean") return element.checked;
  if (field.kind === "choice") return element.querySelector(".seg.active")?.dataset.choice;
  if (field.kind === "number" || field.kind === "integer") {
    return element.value === "" ? null : Number(element.value);
  }
  if (field.kind === "list") {
    return element.value.split(",").map((item) => item.trim()).filter(Boolean);
  }
  return element.value.trim();
}

function sameValue(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function groupChanges(group) {
  const changes = {};
  for (const field of group.fields) {
    const value = readField(field);
    if (!sameValue(value, field.value)) {
      (changes[field.section] ||= {})[field.key] = value;
    }
  }
  return changes;
}

function refreshSaveButtons() {
  settingsState.groups.forEach((group) => {
    const button = document.querySelector(`[data-save-group="${group.name}"]`);
    if (button) button.disabled = settingsState.saving || !Object.keys(groupChanges(group)).length;
  });
}

function settingsCard(group) {
  return `<div class="card settings-card" data-group="${escapeHtml(group.name)}">
        <div class="card-head"><h3>${escapeHtml(group.title)}</h3>
          <button class="btn small primary" type="button" data-save-group="${escapeHtml(group.name)}" disabled>Save</button></div>
        <div class="setting-list">${group.fields
          .map(
            (field) => `<div class="setting-row">
              <label for="${fieldId(field)}" class="setting-label">${escapeHtml(field.label)}${field.private ? ` <span class="pill">Yours</span>` : ""}</label>
              <div class="setting-input">${fieldInput(field, fieldId(field))}${field.help ? `<p class="hint">${escapeHtml(field.help)}</p>` : ""}</div>
            </div>`,
          )
          .join("")}</div>
      </div>`;
}

function renderSettingsForm(payload) {
  settingsState.groups = payload.groups;
  $("#settings-form").innerHTML = payload.groups.map(settingsCard).join("");
}

// After a save, redraw only that card, so edits waiting in other cards survive.
function replaceGroup(payload, name) {
  const fresh = payload.groups.find((group) => group.name === name);
  const index = settingsState.groups.findIndex((group) => group.name === name);
  if (!fresh || index < 0) return renderSettingsForm(payload);
  settingsState.groups[index] = fresh;
  document.querySelector(`.settings-card[data-group="${name}"]`).outerHTML = settingsCard(fresh);
}

async function saveGroup(name) {
  const group = settingsState.groups.find((item) => item.name === name);
  const changes = groupChanges(group);
  if (!Object.keys(changes).length) return;
  settingsState.saving = true;
  refreshSaveButtons();
  try {
    const payload = await api("/api/settings", {
      method: "PUT",
      body: JSON.stringify({ changes }),
    });
    replaceGroup(payload, name);
    toast(`${group.title} saved`, "ok");
    $("#settings-restart").hidden = !payload.runtime_running;
  } catch (error) {
    toast(error.message.replace(/ \(400\)$/, ""), "bad");
  } finally {
    settingsState.saving = false;
    refreshSaveButtons();
  }
}

export function initSettings(restartRuntime) {
  const form = $("#settings-form");
  form.addEventListener("input", refreshSaveButtons);
  form.addEventListener("change", refreshSaveButtons);
  form.addEventListener("click", (event) => {
    const choice = event.target.closest(".segmented .seg");
    if (choice) {
      choice.parentElement.querySelectorAll(".seg").forEach((button) => {
        const active = button === choice;
        button.classList.toggle("active", active);
        button.setAttribute("aria-pressed", String(active));
      });
      refreshSaveButtons();
      return;
    }
    const save = event.target.closest("[data-save-group]");
    if (save) guard(saveGroup)(save.dataset.saveGroup);
  });
  $("#settings-restart-runtime").addEventListener(
    "click",
    guard(async () => {
      $("#settings-restart").hidden = true;
      await restartRuntime();
    }),
  );
}

export function settingsDirty() {
  return settingsState.groups.some((group) => Object.keys(groupChanges(group)).length);
}

export async function loadSettings({ refreshForm = true } = {}) {
  if (refreshForm || !settingsState.groups.length) {
    renderSettingsForm(await api("/api/settings"));
  }
  const config = await api("/api/config");
  $("#config-json").textContent = JSON.stringify(config, null, 2);
  const general = {};
  const sections = [];
  for (const [key, value] of Object.entries(config)) {
    if (value && typeof value === "object" && !Array.isArray(value)) sections.push([key, value]);
    else general[key] = value;
  }
  if (Object.keys(general).length) sections.unshift(["general", general]);
  $("#settings-content").innerHTML = sections
    .map(([key, values]) => {
      const rows = Object.entries(values)
        .map(([name, value]) => `<dt>${escapeHtml(name)}</dt><dd>${settingValue(value)}</dd>`)
        .join("");
      const title = SECTION_TITLES[key] || words(key).replace(/^./, (char) => char.toUpperCase());
      return `<div class="card"><h3>${escapeHtml(title)}</h3><dl class="kv">${rows}</dl></div>`;
    })
    .join("");
}
