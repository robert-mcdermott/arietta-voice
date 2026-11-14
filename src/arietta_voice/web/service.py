"""Console application service: state, runtime control, knowledge, tools and chat, without HTTP."""

from __future__ import annotations

import json
import os
import platform
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arietta_voice import __version__
from arietta_voice.config import AppConfig, config_to_dict, load_config
from arietta_voice.conversation import ChatResult, ConversationService
from arietta_voice.diagnostics import run_startup_diagnostics
from arietta_voice.knowledge import KnowledgeRetriever, build_knowledge_index
from arietta_voice.llm import ChatModel, build_chat_model
from arietta_voice.logging_utils import EventLogger
from arietta_voice.prompts import PromptManager
from arietta_voice.runtime_status import read_runtime_status
from arietta_voice.runtime_supervisor import RuntimeSupervisor
from arietta_voice.timephrases import describe_when, parse_duration, parse_when
from arietta_voice.timers import (
    RINGING,
    TimerStore,
    alarm_name,
    reminder_parts,
    reminder_phrase,
    timer_name,
)
from arietta_voice.tool_admin import ToolAdmin
from arietta_voice.tools.registry import enabled_tool_names
from arietta_voice.turns import CancellationToken
from arietta_voice.web import settings as settings_module
from arietta_voice.web.schemas import ChatRequest, TimerCreateRequest, ToolCreateRequest
from arietta_voice.worker import (
    InferenceWorker,
    WorkerBusy,
    WorkerCancelled,
    WorkerTimeout,
)

# Retain the existing API field names while reporting the application version.
API_VERSION = __version__
AUTH_REALM = "Arietta Admin"


class AriettaApiService:
    def __init__(self, config_path: Path | None = None, *, config: AppConfig | None = None) -> None:
        self.config_path = config_path.resolve() if config_path is not None else None
        self.config = config or load_config(config_path)
        self.started_at = _now_iso()
        self.started_monotonic = time.monotonic()
        self.event_logger = EventLogger.from_config(self.config.logging)
        self.prompt_manager = PromptManager(self.config)
        self.knowledge = KnowledgeRetriever(self.config.knowledge)
        self.runtime_supervisor = RuntimeSupervisor(self.config, config_path=self.config_path)
        self.tool_admin = ToolAdmin(
            self.config,
            config_path=self.config_path,
            knowledge_retriever=self.knowledge,
        )
        self._llm: ChatModel | None = None
        self.worker = InferenceWorker(self.config)
        self._worker_model_pid: int | None = None
        self._chat_lock = threading.Lock()
        self._active_chat = None
        self._retired_chat_ids = deque(maxlen=128)
        self._conversation = ConversationService(
            self.config,
            prompt_manager=self.prompt_manager,
            knowledge=self.knowledge,
            generate=lambda *a, **kw: self._llm_generate(*a, **kw),
        )
        self.chat_turns = 0
        self.last_chat_at: str | None = None
        self.last_error: str | None = None

    def health(self) -> dict[str, Any]:
        try:
            diagnostics = run_startup_diagnostics(self.config, include_hardware=False)
            log_parent = self.config.logging.events_file.parent
            return {
                "status": diagnostics["overall_status"],
                "api_version": API_VERSION,
                "assistant": self.config.assistant.name,
                "generated_at": _now_iso(),
                "uptime_seconds": self.uptime_seconds(),
                "project_root": str(self.config.project_root),
                "events_file": str(self.config.logging.events_file),
                "events_file_exists": self.config.logging.events_file.exists(),
                "logs_writable": os.access(log_parent, os.W_OK),
                "runtime": read_runtime_status(self.config),
                "diagnostics": diagnostics,
            }
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {
                "status": "error",
                "api_version": API_VERSION,
                "generated_at": _now_iso(),
                "error": self.last_error,
            }

    def state(self) -> dict[str, Any]:
        tools = self.inspect_tools()
        event_count = self._event_file_line_count()
        runtime_status = self.runtime_status()
        return {
            "api": {
                "version": API_VERSION,
                "started_at": self.started_at,
                "uptime_seconds": self.uptime_seconds(),
                "chat_turns": self.chat_turns,
                "last_chat_at": self.last_chat_at,
                "last_error": self.last_error,
                "inference": {
                    "operation": self.worker.operation,
                    "worker_pid": self.worker.pid,
                    "worker_restarts": self.worker.restarts,
                    "turn_timeout_s": self.config.runtime.turn_timeout_s,
                },
            },
            "runtime": runtime_status,
            "assistant": {
                "name": self.config.assistant.name,
                "wake_words": list(self.config.wake_word.phrases),
                "wake_enabled": self.config.wake_word.enabled,
            },
            "models": {
                "chat_provider": self.config.models.chat_provider,
                "chat_model": self.config.models.chat_model,
                "llm_loaded": self.worker.pid is not None
                and self.worker.pid == self._worker_model_pid,
                "stt_language": self.config.models.stt_language,
            },
            "knowledge": {
                "enabled": self.config.knowledge.enabled,
                "backend": self.config.knowledge.backend,
                "directory": str(self.config.knowledge.directory),
                "index_directory": str(self.config.knowledge.index_directory),
                "loaded": self.knowledge.is_loaded(),
            },
            "tools": {
                "enabled": tools["enabled"],
                "inventory_mode": "static",
                "error_count": len(tools["errors"]),
            },
            "logs": {
                "events_file": str(self.config.logging.events_file),
                "event_count": event_count,
            },
            "host": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        }

    def config_payload(self) -> dict[str, Any]:
        return config_to_dict(self.config)

    def diagnostics(self, *, include_hardware: bool) -> dict[str, Any]:
        return run_startup_diagnostics(self.config, include_hardware=include_hardware)

    def runtime_status(self) -> dict[str, Any]:
        observed = read_runtime_status(self.config)
        supervisor_status = self.runtime_supervisor.status()
        if supervisor_status["managed_process_running"] and not observed["is_running"]:
            observed = {
                **observed,
                "state": "starting",
                "lifecycle": "starting",
                "note": "The API-managed voice runtime has started; waiting for its first heartbeat.",
            }
        return {
            **observed,
            **supervisor_status,
            "managed_by_api": bool(supervisor_status["managed_process_running"]),
        }

    def start_runtime(self) -> dict[str, Any]:
        payload = self.runtime_supervisor.start()
        self.event_logger.log(
            "api_runtime_start",
            changed=payload.get("changed"),
            message=payload.get("message"),
            process_id=payload.get("managed_process_id"),
        )
        return payload

    def stop_runtime(self) -> dict[str, Any]:
        payload = self.runtime_supervisor.stop()
        self.event_logger.log(
            "api_runtime_stop",
            changed=payload.get("changed"),
            message=payload.get("message"),
            process_id=payload.get("managed_process_id"),
            exit_code=payload.get("managed_exit_code"),
        )
        return payload

    def restart_runtime(self) -> dict[str, Any]:
        payload = self.runtime_supervisor.restart()
        self.event_logger.log(
            "api_runtime_restart",
            changed=payload.get("changed"),
            message=payload.get("message"),
            process_id=payload.get("managed_process_id"),
            exit_code=payload.get("managed_exit_code"),
        )
        return payload

    def read_events(
        self,
        *,
        limit: int,
        offset: int = 0,
        event: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)
        event_filter = None if event is None else event.strip()
        query_filter = "" if query is None else query.strip().lower()
        path = self.config.logging.events_file
        if not path.exists():
            return {
                "events_file": str(path),
                "items": [],
                "count": 0,
                "total": 0,
                "offset": offset,
                "limit": limit,
                "has_next": False,
                "has_previous": False,
                "event": event_filter,
                "query": query_filter,
            }

        matched: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {"event": "unparsed_log_line", "line_number": line_number, "raw": raw}
                if event_filter and payload.get("event") != event_filter:
                    continue
                if query_filter and query_filter not in json.dumps(payload, sort_keys=True).lower():
                    continue
                matched.append(payload)
        ordered = list(reversed(matched))
        page = ordered[offset : offset + limit]
        total = len(ordered)
        return {
            "events_file": str(path),
            "items": page,
            "count": len(page),
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_next": offset + limit < total,
            "has_previous": offset > 0,
            "event": event_filter,
            "query": query_filter,
        }

    def search_knowledge(self, query: str) -> dict[str, Any]:
        query = query.strip()
        result = self.knowledge.search(query)
        matches = (
            [] if result is None else [_knowledge_match_payload(match) for match in result.matches]
        )
        return {
            "query": query,
            "enabled": self.config.knowledge.enabled,
            "backend": self.config.knowledge.backend if result is None else result.backend,
            "directory": str(self.config.knowledge.directory),
            "matches": matches,
        }

    def build_index(self) -> dict[str, Any]:
        payload = build_knowledge_index(self.config.knowledge)
        self.worker.invalidate()
        self.knowledge = KnowledgeRetriever(self.config.knowledge)
        self.tool_admin.knowledge_retriever = self.knowledge
        self.event_logger.log("api_knowledge_index", **payload)
        return payload

    def list_knowledge_files(self) -> dict[str, Any]:
        directory = self.config.knowledge.directory
        directory.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(directory)
            if not _is_supported_knowledge_path(relative_path):
                continue
            stat = path.stat()
            files.append(
                {
                    "path": relative_path.as_posix(),
                    "name": path.name,
                    "size_bytes": stat.st_size,
                    "updated_at": datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    )
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z"),
                }
            )
        return {"directory": str(directory), "files": files}

    def read_knowledge_file(self, file_path: str) -> dict[str, Any]:
        path = _resolve_knowledge_file(self.config, file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(file_path)
        return {
            "path": path.relative_to(self.config.knowledge.directory).as_posix(),
            "content": path.read_text(encoding="utf-8"),
        }

    def write_knowledge_file(self, file_path: str, content: str) -> dict[str, Any]:
        path = _resolve_knowledge_file(self.config, file_path)
        created = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.worker.invalidate()
        self.knowledge = KnowledgeRetriever(self.config.knowledge)
        self.tool_admin.knowledge_retriever = self.knowledge
        payload = {
            "path": path.relative_to(self.config.knowledge.directory).as_posix(),
            "created": created,
            "size_bytes": path.stat().st_size,
        }
        self.event_logger.log("api_knowledge_file_saved", **payload)
        return payload

    def delete_knowledge_file(self, file_path: str) -> dict[str, Any]:
        path = _resolve_knowledge_file(self.config, file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(file_path)
        relative_path = path.relative_to(self.config.knowledge.directory).as_posix()
        path.unlink()
        self.worker.invalidate()
        self.knowledge = KnowledgeRetriever(self.config.knowledge)
        self.tool_admin.knowledge_retriever = self.knowledge
        payload = {"path": relative_path, "deleted": True}
        self.event_logger.log("api_knowledge_file_deleted", **payload)
        return payload

    def inspect_tools(self) -> dict[str, Any]:
        return {
            **self.tool_admin.inventory(),
            "source_editing_enabled": self.config.admin.allow_source_editing,
        }

    def probe_tools(self, text: str) -> dict[str, Any]:
        return self.tool_admin.probe(text)

    def set_enabled_tools(self, enabled: list[str]) -> dict[str, Any]:
        payload = self.tool_admin.set_enabled(enabled)
        self.worker.invalidate()
        self.event_logger.log("api_tools_enabled_updated", enabled=payload["enabled"])
        return payload

    def test_tool(self, name: str, *, text=None, arguments=None) -> dict[str, Any]:
        return self.tool_admin.test(name, text=text, arguments=arguments)

    def validate_tool(self, name: str) -> dict[str, Any]:
        return self.tool_admin.validate(name)

    def create_tool(self, payload: ToolCreateRequest) -> dict[str, Any]:
        result = self.tool_admin.create(
            name=payload.name,
            description=payload.description,
            trigger=payload.trigger,
            response=payload.response,
            enable=payload.enable,
        )
        self.worker.invalidate()
        self.event_logger.log("api_tool_created", name=result["name"], file=result["file"])
        return result

    def reload_tools(self) -> dict[str, Any]:
        self.worker.invalidate()
        payload = self.tool_admin.reload()
        self.event_logger.log("api_tools_reloaded", enabled=payload["enabled"])
        return payload

    def read_tool_source(self, name: str) -> dict[str, Any]:
        return self.tool_admin.read_source(name)

    def write_tool_source(self, name: str, content: str) -> dict[str, Any]:
        payload = self.tool_admin.write_source(name, content)
        self.worker.invalidate()
        self.event_logger.log(
            "api_tool_source_saved",
            name=payload["name"],
            file=payload["file"],
            validation=payload["tool"].get("validation"),
        )
        return payload

    def tool_instructions(self) -> dict[str, Any]:
        return {
            "enabled": enabled_tool_names(self.config),
            "instruction_text": self.tool_admin.instruction_text(),
        }

    # Settings the console can change (see web/settings.py).

    def settings(self) -> dict[str, Any]:
        return {
            **settings_module.describe(self.config, self._base_config_path()),
            "runtime_running": bool(self.runtime_status().get("is_running")),
        }

    def update_settings(self, changes: dict[str, dict[str, Any]]) -> dict[str, Any]:
        changed = settings_module.apply(self.config, self._base_config_path(), changes)
        if changed:
            # The console's own worker reads settings when it starts.
            self.worker.invalidate()
            self.event_logger.log("api_settings_changed", settings=changed)
        return {"changed": changed, **self.settings()}

    def _base_config_path(self) -> Path:
        return self.config_path or self.config.project_root / "config" / "arietta_voice.toml"

    # Timers and reminders. The voice runtime announces them; the console lists,
    # adds and cancels them through the same file.

    def timers(self) -> dict[str, Any]:
        store = TimerStore(self.config.timers.path)
        now = time.time()
        alerts = store.load()
        active = sorted((a for a in alerts if a.active), key=lambda a: a.due_at)
        finished = sorted(
            (a for a in alerts if not a.active), key=lambda a: a.finished_at or 0, reverse=True
        )
        return {
            "now": now,
            "active": [_alert_payload(alert, now) for alert in active],
            "finished": [_alert_payload(alert, now) for alert in finished[:20]],
            "runtime_running": bool(self.runtime_status().get("is_running")),
            "tools_enabled": [
                name
                for name in enabled_tool_names(self.config)
                if name
                in ("set_timer", "set_reminder", "set_alarm", "check_timers", "cancel_timer")
            ],
        }

    def add_timer(self, request: TimerCreateRequest) -> dict[str, Any]:
        store = TimerStore(self.config.timers.path)
        if request.kind == "timer":
            seconds = parse_duration(request.duration)
            if seconds is None:
                raise ValueError("Say how long, such as 10 minutes or 1 hour 30 minutes")
            alert = store.add_timer(seconds, request.label)
        else:
            now = datetime.now().astimezone()
            when = parse_when(request.when, now)
            if when is None:
                raise ValueError("Say when, such as in 20 minutes, at 5 pm or tomorrow at 9")
            if request.kind == "alarm":
                alert = store.add_alarm(when.due, " ".join(request.label.split()))
            else:
                form, body = reminder_parts(request.label)
                alert = store.add_reminder(body, when.due, form)
        return {"added": _alert_payload(alert, time.time()), **self.timers()}

    def cancel_timer(self, alert_id: str) -> dict[str, Any]:
        cancelled = TimerStore(self.config.timers.path).cancel([alert_id])
        if not cancelled:
            raise FileNotFoundError(alert_id)
        return {"cancelled": alert_id, **self.timers()}

    def stop_ringing(self) -> dict[str, Any]:
        store = TimerStore(self.config.timers.path)
        ringing = [alert.id for alert in store.active() if alert.status == RINGING]
        stopped = store.cancel(ringing, outcome="dismissed") if ringing else []
        return {"stopped": [alert.id for alert in stopped], **self.timers()}

    def chat(self, request: ChatRequest) -> ChatResult:
        token = CancellationToken()
        request_id = request.request_id or uuid.uuid4().hex
        with self._chat_lock:
            if request_id in self._retired_chat_ids:
                raise WorkerCancelled("Request ID already cancelled or completed; use a fresh ID")
            if self._active_chat is not None:
                raise WorkerBusy("Inference is busy")
            self._active_chat = (request_id, token)
        try:
            payload = request.model_dump(exclude={"request_id"})
            payload.update(
                enabled_tools=tuple(enabled_tool_names(self.config)),
            )
            response = self.worker.call(
                "chat", payload, timeout_s=self.config.runtime.turn_timeout_s, cancellation=token
            )
            with self._chat_lock:
                token.check()
                self._worker_model_pid = (
                    self.worker.pid if response.pop("model_loaded", False) else None
                )
                result = ChatResult(**response)
                self.last_error = None
                self._record_chat(
                    request.message,
                    result.response,
                    source=result.source,
                    used_tool=result.used_tool,
                    model_tool=result.model_tool,
                )
                self._retired_chat_ids.append(request_id)
                self._active_chat = None
                return result
        finally:
            with self._chat_lock:
                if self._active_chat is not None and self._active_chat[1] is token:
                    self._retired_chat_ids.append(request_id)
                    self._active_chat = None

    def chat_stream(self, request: ChatRequest):
        """Start a chat turn and return an iterator of its events as they happen.

        Starting happens now, so a busy worker or a reused request ID fails
        before anything is streamed. The iterator yields ``clause`` events as
        the answer is spoken by voice, then one ``done`` event with the result.
        Closing it early stops the turn.
        """
        token = CancellationToken()
        request_id = request.request_id or uuid.uuid4().hex
        with self._chat_lock:
            if request_id in self._retired_chat_ids:
                raise WorkerCancelled("Request ID already cancelled or completed; use a fresh ID")
            if self._active_chat is not None:
                raise WorkerBusy("Inference is busy")
            self._active_chat = (request_id, token)
        deadline = time.monotonic() + self.config.runtime.turn_timeout_s
        payload = request.model_dump(exclude={"request_id"})
        payload.update(enabled_tools=tuple(enabled_tool_names(self.config)))
        stream = ChatStream(self, request, request_id, token, deadline)
        try:
            self.worker.call(
                "chat_start",
                payload,
                timeout_s=self.config.runtime.turn_timeout_s,
                cancellation=token,
            )
        except BaseException:
            stream.close()
            raise
        return stream

    def _stream_events(self, request, token, deadline):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeout("Turn exceeded its time limit")
            event = self.worker.call("chat_next", timeout_s=remaining, cancellation=token)
            if not isinstance(event, dict) or event.get("kind") not in ("clause", "done"):
                raise ValueError("Invalid conversation event")
            if event["kind"] == "clause":
                yield {"kind": "clause", "text": event["text"]}
                continue
            with self._chat_lock:
                token.check()
            result = ChatResult(**event["result"])
            self.last_error = None
            if result.source == "model":
                self._worker_model_pid = self.worker.pid
            self._record_chat(
                request.message,
                result.response,
                source=result.source,
                used_tool=result.used_tool,
                model_tool=result.model_tool,
            )
            yield {"kind": "done", "result": asdict(result)}
            return

    def _finish_stream(self, request_id, token) -> None:
        with self._chat_lock:
            if self._active_chat is not None and self._active_chat[1] is token:
                self._retired_chat_ids.append(request_id)
                self._active_chat = None

    def cancel_chat(self, request_id: str) -> bool:
        with self._chat_lock:
            if self._active_chat is not None and self._active_chat[0] == request_id:
                self._active_chat[1].cancel()
                return True
            if request_id in self._retired_chat_ids:
                return False
            # Remember early cancellation if the HTTP stop request overtakes chat.
            self._retired_chat_ids.append(request_id)
            return True

    def _chat_in_process(self, request: ChatRequest, *, record: bool = True) -> ChatResult:
        # In-process test/embedding entry point; production calls run in the worker.
        result = self._conversation.chat(**request.model_dump(exclude={"request_id"}))
        if record:
            self._record_chat(
                request.message,
                result.response,
                source=result.source,
                used_tool=result.used_tool,
                model_tool=result.model_tool,
            )
        return result

    def uptime_seconds(self) -> int:
        return int(time.monotonic() - self.started_monotonic)

    def _llm_generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None,
        temperature: float | None,
    ) -> str:
        if self._llm is None:
            self._llm = build_chat_model(self.config.models)
        return self._llm.generate(
            messages,
            max_tokens=max_tokens or self.config.models.max_tokens,
            temperature=temperature if temperature is not None else self.config.models.temperature,
        )

    def _record_chat(
        self,
        message: str,
        response: str,
        *,
        source: str,
        used_tool: str | None,
        model_tool: str | None = None,
    ) -> None:
        self.chat_turns += 1
        self.last_chat_at = _now_iso()
        self.event_logger.log(
            "api_chat",
            source=source,
            used_tool=used_tool,
            model_tool=model_tool,
            message_chars=len(message),
            response_chars=len(response),
        )

    def _event_file_line_count(self) -> int:
        path = self.config.logging.events_file
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class ChatStream:
    """A console chat turn's events; closing it, early or late, always frees the turn.

    The HTTP layer closes it from a background task as well as after the last
    event, because a client that disconnects before the first event means the
    event generator never starts, and an unstarted generator's cleanup never
    runs. ``close`` is safe to call more than once.
    """

    def __init__(self, service, request, request_id, token, deadline) -> None:
        self.service, self.request_id, self.token = service, request_id, token
        self._events = service._stream_events(request, token, deadline)
        self._closed = False
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._events)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._events.close()
        finally:
            try:
                if self.service.worker.pid is not None:
                    self.service.worker.call("chat_close", timeout_s=2)
            except Exception:
                pass  # A failed worker is replaced on the next call.
            self.service._finish_stream(self.request_id, self.token)


def _alert_payload(alert, now: float) -> dict[str, Any]:
    due = datetime.fromtimestamp(alert.due_at).astimezone()
    return {
        "id": alert.id,
        "kind": alert.kind,
        "status": alert.status,
        "label": alert.label,
        "name": {
            "timer": timer_name,
            "alarm": alarm_name,
        }.get(alert.kind, reminder_phrase)(alert),
        "duration_s": alert.duration_s,
        "due_at": alert.due_at,
        "due_iso": due.isoformat(timespec="seconds"),
        "due_text": describe_when(due, datetime.fromtimestamp(now).astimezone()),
        "remaining_s": max(0.0, alert.due_at - now),
        "rings": alert.rings,
        "outcome": alert.outcome,
        "finished_at": alert.finished_at,
    }


def _knowledge_match_payload(match: Any) -> dict[str, Any]:
    return {
        "source": str(match.source_path),
        "title": match.title,
        "section": match.section_title,
        "score": match.score,
        "snippet": match.snippet,
    }


def _is_supported_knowledge_path(path: Path) -> bool:
    if any(part.startswith(".") for part in path.parts):
        return False
    return path.suffix.lower() in {".md", ".markdown", ".txt"}


def _resolve_knowledge_file(config: AppConfig, file_path: str) -> Path:
    normalized = Path(file_path)
    if not str(file_path).strip():
        raise ValueError("Knowledge file path is required.")
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError("Knowledge file path must stay inside the knowledge directory.")
    if any(part.startswith(".") for part in normalized.parts):
        raise ValueError("Hidden knowledge files are not supported.")
    if normalized.suffix.lower() not in {".md", ".markdown", ".txt"}:
        raise ValueError("Knowledge files must use .md, .markdown, or .txt.")

    root = config.knowledge.directory.resolve()
    target = (root / normalized).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Knowledge file path must stay inside the knowledge directory.") from exc
    return target
