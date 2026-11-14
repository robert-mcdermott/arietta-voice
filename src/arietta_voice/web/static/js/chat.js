// Chat: talk to the assistant by text through the same conversation service as voice.

import { $, api, escapeHtml, guard, setError, streamEvents } from "./core.js";
import { overview } from "./home.js";

const chat = { history: [], pending: false, requestId: null };
const IDLE_HINT = "Enter to send · Shift + Enter for a new line";

export function chatPending() {
  return chat.pending;
}

function assistantName() {
  return overview.state?.assistant?.name || "Assistant";
}

function messageMeta(message) {
  if (message.role !== "assistant" || !message.meta) return "";
  const parts = [];
  if (message.meta.source === "tool") parts.push("Answered by an instant tool");
  if (message.meta.used_tool) parts.push(`Tool: ${message.meta.used_tool}`);
  if (message.meta.model_tool) parts.push(`Model tool: ${message.meta.model_tool}`);
  if (message.meta.first != null) parts.push(`First words ${message.meta.first.toFixed(1)}s`);
  if (message.meta.seconds != null) parts.push(`${message.meta.seconds.toFixed(1)}s`);
  return parts.length
    ? `<div class="msg-meta">${parts.map((part) => `<span>${escapeHtml(part)}</span>`).join("<span>·</span>")}</div>`
    : "";
}

// The assistant asked before an action: answer it as you would by voice.
function answerButtons(message) {
  if (message.role !== "assistant" || !message.meta?.confirmation) return "";
  return `<div class="msg-answers row">
      <button type="button" class="btn small primary" data-chat-answer="Yes">Yes</button>
      <button type="button" class="btn small" data-chat-answer="Always">Always</button>
      <button type="button" class="btn small quiet" data-chat-answer="No">No</button>
    </div>`;
}

function render() {
  const log = $("#chat-log");
  if (!chat.history.length && !chat.pending) {
    log.innerHTML = `<div class="chat-empty"><strong>Chat with ${escapeHtml(assistantName())}</strong>
      Type a message to test answers, knowledge and tools without using the microphone.</div>`;
    return;
  }
  const last = chat.history[chat.history.length - 1];
  const messages = chat.history
    .filter((message) => message.content)
    .map(
      (message) => `<div class="msg ${message.role === "user" ? "user" : "assistant"}">
        <div class="msg-body">${escapeHtml(message.content)}</div>${messageMeta(message)}${
          message === last && !chat.pending ? answerButtons(message) : ""
        }</div>`,
    )
    .join("");
  const waiting = chat.pending && !(last?.role === "assistant" && last.content);
  const typing = waiting
    ? `<div class="msg assistant"><div class="msg-body"><span class="typing" aria-label="Thinking"><span></span><span></span><span></span></span></div></div>`
    : "";
  log.innerHTML = messages + typing;
  log.scrollTop = log.scrollHeight;
}

function setPending(pending) {
  chat.pending = pending;
  $("#chat-stop").hidden = !pending;
  $("#chat-stop").disabled = false;
  $("#chat-send").disabled = pending;
  $("#chat-clear").disabled = pending;
  $("#chat-message").disabled = pending;
}

function autosize() {
  const textarea = $("#chat-message");
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight + 2, 180)}px`;
}

async function send() {
  const textarea = $("#chat-message");
  const message = textarea.value.trim();
  if (!message || chat.pending) return;
  const history = chat.history.slice(-10).map(({ role, content }) => ({ role, content }));
  chat.requestId = Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  chat.history.push({ role: "user", content: message });
  textarea.value = "";
  autosize();
  setPending(true);
  $("#chat-status").textContent = "Thinking…";
  render();
  const started = performance.now();
  // The answer appears clause by clause, as it would be spoken.
  const reply = { role: "assistant", content: "", meta: null };
  chat.history.push(reply);
  try {
    let result = null;
    await streamEvents(
      "/api/chat/stream",
      { message, history, request_id: chat.requestId },
      (event) => {
        if (event.kind === "clause") {
          reply.content = reply.content ? `${reply.content} ${event.text}` : event.text;
          if (!reply.firstAt) reply.firstAt = performance.now();
          render();
        } else if (event.kind === "done") {
          result = event.result;
        } else if (event.kind === "error") {
          const error = new Error(`${event.detail} (${event.status})`);
          error.status = event.status;
          throw error;
        }
      },
    );
    if (!result) throw new Error("The answer ended before it finished. Try again.");
    reply.content = result.response;
    reply.meta = {
      source: result.source,
      used_tool: result.used_tool,
      model_tool: result.model_tool,
      confirmation: result.confirmation,
      seconds: (performance.now() - started) / 1000,
      first: reply.firstAt ? (reply.firstAt - started) / 1000 : null,
    };
    chat.history = chat.history.slice(-40);
    $("#chat-status").textContent = IDLE_HINT;
  } catch (error) {
    chat.history.pop(); // The partial answer.
    chat.history.pop();
    textarea.value = message;
    $("#chat-status").textContent = error.message;
    if (error.status !== 409) setError(error.message);
  } finally {
    chat.requestId = null;
    setPending(false);
    render();
    autosize();
    textarea.focus();
  }
}

async function stop() {
  const requestId = chat.requestId;
  if (!requestId) return;
  $("#chat-stop").disabled = true;
  try {
    const result = await api(`/api/chat/${requestId}/cancel`, { method: "POST" });
    if (chat.requestId === requestId) {
      $("#chat-status").textContent = result.cancelled
        ? "Stopping…"
        : "The turn has already finished";
    }
  } catch (error) {
    if (chat.requestId === requestId) $("#chat-stop").disabled = false;
    throw error;
  }
}

export function showChat() {
  render();
  $("#chat-message").focus();
}

export function initChat() {
  $("#chat-send").addEventListener("click", guard(send));
  $("#chat-stop").addEventListener("click", guard(stop));
  $("#chat-clear").addEventListener("click", () => {
    chat.history = [];
    render();
  });
  $("#chat-log").addEventListener("click", (event) => {
    const button = event.target.closest("[data-chat-answer]");
    if (!button || chat.pending) return;
    $("#chat-message").value = button.dataset.chatAnswer;
    guard(send)();
  });
  $("#chat-message").addEventListener("input", autosize);
  $("#chat-message").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      guard(send)();
    }
  });
  render();
}
