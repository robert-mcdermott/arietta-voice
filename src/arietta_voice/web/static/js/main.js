// Entry point: navigation, theme, polling, and wiring of each view.

import { activityOnFirstPage, initActivity, loadActivity } from "./activity.js";
import { chatPending, initChat, showChat } from "./chat.js";
import { connectionsDirty, initConnections, loadConnections } from "./connections.js";
import { $, $$, guard, setError } from "./core.js";
import {
  loadContentLoggingFlag,
  loadOverview,
  pollRuntime,
  runtimeAction,
  runtimeEventsLive,
  startRuntimeEvents,
  stopRuntimeEvents,
} from "./home.js";
import { initKnowledge, knowledgeDirty, knowledgeEditing, loadKnowledgeFiles } from "./knowledge.js";
import { initPersona, loadPersona, personaDirty } from "./persona.js";
import { initSettings, loadDiagnostics, loadSettings, settingsDirty } from "./system.js";
import { initTimers } from "./timers.js";
import { initTools, loadTools, toolsBusy, toolSourceDirty } from "./tools.js";

const VIEWS = {
  home: ["Home", "What your assistant is doing right now."],
  chat: ["Chat", "Type to your assistant. Uses the same answers, knowledge and tools as voice."],
  knowledge: ["Knowledge", "Notes your assistant can draw on when it answers."],
  persona: ["Persona", "Who your assistant is and what it remembers."],
  tools: ["Tools", "What your assistant can do beyond answering from its model."],
  connections: ["Connections", "MCP servers that give your assistant more tools."],
  activity: ["Activity", "A record of what the assistant heard, did and said."],
  system: ["System", "Health, audio and process detail for troubleshooting."],
  settings: ["Settings", "Change how your assistant sounds and behaves."],
};
const RUNTIME_POLL_MS = 1500;
const OVERVIEW_POLL_MS = 6000;
const VIEW_POLL_MS = 10000;

let currentView = "home";

// Routes look like "#/tools"; a bare "#tools" would make the browser scroll to that section.
function viewFromHash() {
  return location.hash.replace(/^#\/?/, "") || "home";
}
let viewLoading = false;

async function loadView(view) {
  if (view === "chat") showChat();
  if (view === "knowledge") await loadKnowledgeFiles();
  if (view === "persona") await loadPersona();
  if (view === "tools") await loadTools();
  if (view === "connections") await loadConnections();
  if (view === "activity") await loadActivity({ reset: true });
  if (view === "system") await loadDiagnostics();
  if (view === "settings") await loadSettings();
}

function switchView(view, { updateHash = true } = {}) {
  if (!Object.hasOwn(VIEWS, view)) view = "home";
  currentView = view;
  $("#section-picker").value = view;
  $$(".nav-button").forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  $$(".view").forEach((section) => section.classList.toggle("active", section.id === view));
  if (view === "home" && !document.hidden) startRuntimeEvents();
  else stopRuntimeEvents();
  $("#view-title").textContent = VIEWS[view][0];
  $("#view-sub").textContent = VIEWS[view][1];
  document.title = `${VIEWS[view][0]} · Arietta Voice`;
  if (updateHash && location.hash !== `#/${view}`) history.pushState(null, "", `#/${view}`);
  guard(loadView)(view);
}

// Refresh the open view in the background, but never under the user's edits.
async function refreshView() {
  if (viewLoading || document.hidden) return;
  viewLoading = true;
  try {
    if (currentView === "activity" && activityOnFirstPage()) await loadActivity();
    if (currentView === "tools" && !toolsBusy()) await loadTools();
    if (currentView === "knowledge" && !knowledgeEditing()) await loadKnowledgeFiles();
    if (currentView === "persona" && !personaDirty()) await loadPersona();
    if (currentView === "system") await loadDiagnostics();
  } catch (error) {
    setError(error.message);
  } finally {
    viewLoading = false;
  }
}

function initTheme() {
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("arietta-theme", next);
    } catch (error) {
      // Private browsing may refuse storage; the choice still applies to this page.
    }
  });
}

function init() {
  const picker = $("#section-picker");
  for (const [view, [title]] of Object.entries(VIEWS)) picker.add(new Option(title, view));
  picker.addEventListener("change", () => switchView(picker.value));
  initTheme();
  initChat();
  initKnowledge();
  initPersona();
  initTools();
  initConnections();
  initTimers();
  initSettings(() => runtimeAction("restart"));
  initActivity();

  $$(".nav-button").forEach((button) =>
    button.addEventListener("click", () => switchView(button.dataset.view)),
  );
  $$("[data-view-target]").forEach((button) =>
    button.addEventListener("click", () => switchView(button.dataset.viewTarget)),
  );
  window.addEventListener("hashchange", () =>
    switchView(viewFromHash(), { updateHash: false }),
  );

  for (const action of ["start", "stop", "restart"]) {
    $(`#runtime-${action}`).addEventListener(
      "click",
      guard(() => runtimeAction(action)),
    );
  }
  $("#diagnostics-run").addEventListener(
    "click",
    guard(() => loadDiagnostics({ hardware: true })),
  );
  $("#error-dismiss").addEventListener("click", () => {
    $("#error-notice").hidden = true;
  });

  window.addEventListener("beforeunload", (event) => {
    if (
      toolSourceDirty() ||
      knowledgeDirty() ||
      chatPending() ||
      settingsDirty() ||
      connectionsDirty() ||
      personaDirty()
    ) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopRuntimeEvents(); // Frees the server's connection while the tab is hidden.
      return;
    }
    if (currentView === "home") startRuntimeEvents();
    guard(loadOverview)();
    refreshView();
  });

  setInterval(() => {
    if (!document.hidden && currentView === "home" && !runtimeEventsLive()) guard(pollRuntime)();
  }, RUNTIME_POLL_MS);
  setInterval(() => {
    if (!document.hidden) guard(loadOverview)();
  }, OVERVIEW_POLL_MS);
  setInterval(refreshView, VIEW_POLL_MS);

  const initialView = viewFromHash();
  const view = Object.hasOwn(VIEWS, initialView) ? initialView : "home";
  history.replaceState(null, "", `#/${view}`);
  switchView(view, { updateHash: false });
  guard(loadOverview)();
  guard(loadContentLoggingFlag)();
}

init();
